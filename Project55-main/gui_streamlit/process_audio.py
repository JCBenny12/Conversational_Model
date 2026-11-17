import torch
import pyaudio
import numpy as np
import librosa
import whisper
from transformers import pipeline

# --------------------------------------------------------
# 1) Load the WavLM-based emotion classification model
# --------------------------------------------------------
try:
    emotion_pipeline = pipeline(
        task="audio-classification",
        model="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition",
        use_auth_token=True  # If needed for private models
    )
    print("Audio Emotion classification pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading pipeline: {e}")
    emotion_pipeline = None

# --------------------------------------------------------
# 2) Load Whisper model for transcription
# --------------------------------------------------------
whisper_model = whisper.load_model("base")

# --------------------------------------------------------
# 3) Define your emotion labels
#    NOTE: "superb/wavlm-base-superb-er" has 8 classes
#    (id2label: {0: 'neutral', 1: 'calm', 2: 'happy', 3: 'sad', 4: 'angry', 5: 'fearful', 6: 'disgust', 7: 'surprised'})
#    We can reuse the same label list you had:
# --------------------------------------------------------
emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]


def preprocess_audio(audio_array, orig_sr=16000, target_sr=16000):
    """
    Preprocesses the audio array for the WavLM emotion model.
    
    Args:
        audio_array (np.ndarray): The input audio array.
        orig_sr (int): Original sampling rate of the audio.
        target_sr (int): The target (model-required) sampling rate.
        
    Returns:
        torch.Tensor: The preprocessed audio tensor.
    """
    # Resample the audio to the required sampling rate (if needed)
    if orig_sr != target_sr:
        audio_array = librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)
    print("Preprocessing done.")
    return audio_array


def predict_emotion(audio_array, sr=16000):
    """
    Predicts emotion from captured audio using superb/wavlm-base-superb-er.
    """
    # Preprocess audio for the model
    # audio_array, sr = librosa.load(audio_file, sr=16000)

    # Predict emotion using the pipeline
    if emotion_pipeline is not None:
        emotion_result = emotion_pipeline(audio_array)
        predicted_emotion = emotion_result[0]['label']  # Most confident prediction
    else:
        print("Emotion model unavailable")
        predicted_emotion = "neutral"

    print("Predicted Emotion:", predicted_emotion)
    return predicted_emotion


def transcribe_audio(audio_array, sr=16000):
    """
    Transcribes speech using Whisper model.
    """
    # Resample audio if needed for Whisper (whisper base generally expects 16k too)
    if sr != 16000:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)

    # Whisper requires float32 audio
    audio_array = audio_array.astype(np.float32)

    # Transcribe with Whisper
    result = whisper_model.transcribe(audio_array)
    return result["text"]


def process_audio_file(audio_file):
    """
    Processes an audio file to predict emotion and transcribe speech.
    
    Returns:
        (str, str): (detected_emotion, transcription)
    """
    # Load audio file at 16k
    audio_array, sr = librosa.load(audio_file, sr=16000)

    # Predict emotion
    detected_emotion = predict_emotion(audio_array, sr=sr)

    # Transcribe
    transcription = transcribe_audio(audio_array, sr=sr)

    return detected_emotion, transcription
