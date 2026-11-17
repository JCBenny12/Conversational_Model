import cv2
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model
import streamlit as st

# Load the saved emotion recognition model
model = load_model("../video_integration/emotion_recognition_model.h5")

# Define emotion labels
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def extract_frames(video_path, frames_per_second):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frames_per_second)

    extracted_frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            extracted_frames.append(frame)

        frame_count += 1

    cap.release()
    return extracted_frames

def predict_emotion_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotion_label = "Neutral"  # Default emotion
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized.astype('float32') / 255.0
        face_input = np.reshape(face_normalized, (1, 48, 48, 1))

        prediction = model.predict(face_input)
        predicted_class = np.argmax(prediction)
        emotion_label = emotions[predicted_class]

    return emotion_label

def process_video(video_path, frames_per_second):
    frames = extract_frames(video_path, frames_per_second)
    print(f"Extracted {len(frames)} frames for processing.")

    predictions = []
    for i, frame in enumerate(frames):
        emotion = predict_emotion_from_frame(frame)
        predictions.append(emotion)

    return predictions

def summarize_emotions(predictions):
    emotion_counts = Counter(predictions)
    total_frames = sum(emotion_counts.values())

    emotion_percentages = {emotion: (count / total_frames) * 100 for emotion, count in emotion_counts.items()}
    top_three = sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True)[:3]

    summary = ", ".join([f"{emotion}: {percent:.2f}%" for emotion, percent in top_three])
    print("\nTop 3 Emotions Detected:")
    print(summary)

def process_recorded_video():
    if st.session_state.full_video_path:
        emotion_predictions = process_video(st.session_state.full_video_path, frames_per_second=20)
        summarize_emotions(emotion_predictions)

if __name__ == "__main__":
    process_recorded_video()