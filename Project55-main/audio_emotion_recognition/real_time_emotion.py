import torch
import pyaudio
import numpy as np
import librosa
import whisper
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

# Load fine-tuned emotion model
model_path = "wav2vec2-emotion-ravdess"
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)

# Load Whisper model for transcription
whisper_model = whisper.load_model("base")

# Emotion labels
emotion_labels = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]

# PyAudio settings
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 0.001  # Adjust if needed
SILENCE_FRAMES = 20  # Number of silent frames before stopping

# Initialize PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("🎤 Listening for speech... Speak and then pause to get emotion prediction and transcription.")

def is_silent(audio_chunk):
    """Detect if the audio is silent based on Root Mean Square (RMS) energy."""
    rms_energy = np.sqrt(np.mean(np.square(audio_chunk)))
    return rms_energy < SILENCE_THRESHOLD

def predict_emotion(audio_array):
    """Predicts emotion from captured audio."""
    input_values = feature_extractor(audio_array, sampling_rate=16000, return_tensors="pt")["input_values"]

    with torch.no_grad():
        logits = emotion_model(input_values).logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    return emotion_labels[predicted_class]

def transcribe_audio(audio_array):
    """Transcribes speech using Whisper model."""
    # Convert NumPy array to float32 WAV format
    audio_array = librosa.resample(audio_array, orig_sr=16000, target_sr=16000)
    whisper_result = whisper_model.transcribe(audio_array)
    return whisper_result["text"]

try:
    buffer = []
    silent_frames = 0
    speaking = False

    while True:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        
        if not is_silent(audio_array):  
            buffer.append(audio_array)  
            silent_frames = 0
            speaking = True
        else:
            if speaking:
                silent_frames += 1

            if silent_frames > SILENCE_FRAMES:  # If silence is detected long enough
                if buffer:
                    print("\n🔍 Processing speech...")
                    full_audio = np.concatenate(buffer, axis=0)
                    
                    # Predict emotion
                    detected_emotion = predict_emotion(full_audio)

                    # Transcribe speech
                    transcription = transcribe_audio(full_audio)

                    # Display results
                    print(f"🎭 Detected Emotion: {detected_emotion}")
                    print(f"📝 Transcription: {transcription}\n")

                # Reset buffer
                buffer = []
                silent_frames = 0
                speaking = False

except KeyboardInterrupt:
    print("\n🛑 Stopping...")
    stream.stop_stream()
    stream.close()
    p.terminate()
