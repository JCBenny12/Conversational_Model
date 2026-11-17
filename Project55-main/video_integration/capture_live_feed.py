import cv2
import numpy as np
import requests
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model("emotion_recognition_model.h5")

# Define emotion labels
emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# Set up the webcam feed
cap = cv2.VideoCapture(0)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    emotion_label = "Neutral"  # Default emotion

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face, (48, 48))
        face_normalized = face_resized.astype('float32') / 255.0
        face_input = np.reshape(face_normalized, (1, 48, 48, 1))

        # Make a prediction
        prediction = model.predict(face_input)
        predicted_class = np.argmax(prediction)
        emotion_label = emotions[predicted_class]

        # Send detected emotion to Flask API (GUI)
        try:
            requests.post("http://127.0.0.1:5000/emotion", json={"emotion": emotion_label})
        except requests.exceptions.RequestException:
            print("Could not send emotion data to GUI. Ensure the Flask server is running.")

        # Draw face box and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the frame
    cv2.imshow("Live Emotion Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
