import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import cv2
import urllib.request
import mediapipe as mp
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
MODEL_PATH = 'pose_landmarker.task'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Automatically download Google's official Pose Task asset if missing
if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe Pose Landmarker model...")
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded successfully.")

# Setup modern MediaPipe Tasks aliases
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def process_video(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.get()
            if not ret:
                break

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            detection_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            
            if detection_result.pose_landmarks:
                for landmarks in detection_result.pose_landmarks:
                    for lm in landmarks:
                        cx, cy = int(lm.x * width), int(lm.y * height)
                        if 0 <= cx < width and 0 <= cy < height:
                            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            out.write(frame)

    cap.release()
    out.release()

@app.route('/api/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        output_filename = "processed_" + filename
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        
        file.save(input_path)
        process_video(input_path, output_path)
        
        return send_file(output_path, as_attachment=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
