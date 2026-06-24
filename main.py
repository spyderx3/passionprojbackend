import os
import shutil
import uuid
import time
import threading

import cv2
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Pose Landmark Video Processor")

# Allow the website (frontend) to talk to this backend from any domain.
# Once you know your real frontend URL, you can replace "*" with it for tighter security.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FILE_SIZE_MB = 100

# In-memory job tracker: job_id -> status dict
# (Fine for ~20-40 users; would need a real database at much larger scale)
jobs = {}


def process_video_with_pose(job_id: str, input_path: str, output_path: str):
    """Runs in a background thread so the upload request returns immediately."""
    try:
        jobs[job_id]["status"] = "processing"

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file. It may be corrupted or unsupported.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frames_processed = 0
        frames_with_pose = 0

        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                success, frame = cap.read()
                if not success:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = pose.process(frame_rgb)

                if results.pose_landmarks:
                    frames_with_pose += 1
                    mp_drawing.draw_landmarks(
                        frame,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                writer.write(frame)
                frames_processed += 1
                jobs[job_id]["progress"] = round((frames_processed / total_frames) * 100, 1)

        cap.release()
        writer.release()

        if not os.path.exists(output_path):
            raise RuntimeError("Output video was not created.")

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["frames_with_pose"] = frames_with_pose
        jobs[job_id]["total_frames"] = total_frames

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)


def cleanup_old_files():
    """Background loop: deletes files older than 1 hour to avoid filling up disk."""
    while True:
        time.sleep(600)  # check every 10 minutes
        cutoff = time.time() - 3600
        for folder in (UPLOAD_DIR, OUTPUT_DIR):
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath) and os.path.getmtime(fpath) < cutoff:
                    os.remove(fpath)


threading.Thread(target=cleanup_old_files, daemon=True).start()


@app.get("/")
def root():
    return {"message": "Pose Landmark Video Processor is running."}


@app.post("/process-video/")
async def process_video(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    job_id = uuid.uuid4().hex
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_pose.mp4")

    size = 0
    try:
        with open(input_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_FILE_SIZE_MB * 1024 * 1024:
                    raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_FILE_SIZE_MB}MB.")
                buffer.write(chunk)
    finally:
        await file.close()

    jobs[job_id] = {"status": "queued", "progress": 0, "output_path": output_path}

    thread = threading.Thread(
        target=process_video_with_pose,
        args=(job_id, input_path, output_path),
        daemon=True,
    )
    thread.start()

    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "status": job["status"],
        "progress": job.get("progress", 0),
        "error": job.get("error"),
    }


@app.get("/download/{job_id}")
def download(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail=f"Job is not finished yet (status: {job['status']}).")
    output_path = job["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file no longer available.")
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=f"pose_{job_id}.mp4",
    )
