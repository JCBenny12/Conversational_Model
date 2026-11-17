import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model

def get_first_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Read the first frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    if not ret:
        print("Error: Could not read frame.")
        return None
    
    return frame

def process_frame(frame):
    # Load the saved model
    model = load_model("../video_integration/emotion_recognition_model.h5")
    
    # Define emotion labels
    emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    
    # Convert frame to grayscale
    print("Converting frame to grayscale")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade for face detection
    print("Loading Haar Cascade for face detection")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the frame
    print("Detecting faces in the frame")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    emotion_label = "Neutral"  # Default emotion
    
    for (x, y, w, h) in faces:
        # Extract the face region
        face = gray[y:y+h, x:x+w]
        
        # Resize the face image to 48x48 pixels
        print("Resizing face image")
        face_resized = cv2.resize(face, (48, 48))
        
        # Normalize the pixel values
        print("Normalizing pixel values")
        face_normalized = face_resized.astype('float32') / 255.0
        
        # Reshape the image for model input
        print("Reshaping image for model input")
        face_input = np.reshape(face_normalized, (1, 48, 48, 1))
        
        # Make a prediction
        print("Making prediction")
        prediction = model.predict(face_input)
        predicted_class = np.argmax(prediction)
        print(f"Predicted class: {predicted_class}")
        emotion_label = emotions[predicted_class]
        print(f"Visual Emotion: {emotion_label}")

    return emotion_label


