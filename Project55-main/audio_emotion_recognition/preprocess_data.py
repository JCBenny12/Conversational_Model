import os
import librosa
import pandas as pd
import numpy as np
from tqdm import tqdm

# Paths to datasets
DATASET_PATHS = {
    "RAVDESS": "C:/Users/Haley/Desktop/ELEC 490/1188976",
    "CREMA-D": "C:/Users/Haley/Desktop/ELEC 490/CREMA-D",
    "TESS": "C:/Users/Haley/Desktop/ELEC 490/TESS"
}

# Destination to save the processed CSV file
OUTPUT_PATH = r"C:/Users/Haley/Desktop/ELEC 490/emotion_features.csv"

# Emotion labels mapping (RAVDESS, CREMA-D, TESS)
emotion_dict = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}

# Function to extract features from audio files
def extract_features(file_path, sample_rate=16000):
    y, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)
    
    # Extract Pitch (F0)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=300, sr=sr)
    pitch_mean = np.nanmean(f0) if np.any(~np.isnan(f0)) else 0
    
    # Extract Energy (RMS)
    rms = np.mean(librosa.feature.rms(y=y))
    
    # Spectral Features
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    return [pitch_mean, spectral_centroid, spectral_bandwidth, rms, zcr] + list(mfccs_mean)

# List to hold extracted features and labels
data = []

# Process each dataset
for dataset_name, dataset_path in DATASET_PATHS.items():
    print(f"Processing {dataset_name}...")
    
    for root, _, files in os.walk(dataset_path):
        for file in tqdm(files):
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                
                # Extract emotion label from filename
                if dataset_name == "RAVDESS":
                    emotion_code = file.split("-")[2]
                    emotion_label = emotion_dict.get(emotion_code)
                elif dataset_name == "CREMA-D":
                    emotion_label = file.split("_")[2].lower()
                elif dataset_name == "TESS":
                    emotion_label = file.split("_")[2].split(".")[0].lower()
                else:
                    continue
                
                if emotion_label:
                    features = extract_features(file_path)
                    data.append([file_path, dataset_name, emotion_label] + features)

# Create a DataFrame
columns = ["file_path", "dataset", "emotion", "pitch", "spectral_centroid", "spectral_bandwidth", "rms", "zcr"] + [f"mfcc_{i+1}" for i in range(13)]
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to the specified location
df.to_csv(OUTPUT_PATH, index=False)
print(f"Feature extraction complete. Data saved to: {OUTPUT_PATH}")
