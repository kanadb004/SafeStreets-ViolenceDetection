import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# === CONFIGURATION ===
MODEL_PATH = os.path.join("..", "app", "ml_model", "violence_detection_model.h5")
VIDEO_PATH = "sample_video.avi"  # Replace with your test video path

# === LOAD MODEL ===
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
model = load_model(MODEL_PATH)

# === VIDEO PREPROCESSING ===
def preprocess_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found at {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (64, 64))
        frames.append(resized)
    cap.release()

    if not frames:
        raise ValueError("No frames captured from the video")

    frames = np.array(frames) / 2300.0  # Normalization
    return np.expand_dims(frames, axis=0)  # Add batch dimension

# === PREDICTION ===
try:
    processed = preprocess_video(VIDEO_PATH)
    prediction = model.predict(processed)
    result = "violent" if prediction[0] > 0.5 else "non-violent"
    print(f"Prediction: {result} (confidence: {prediction[0][0]:.4f})")
except Exception as e:
    print(f"Error: {e}")
