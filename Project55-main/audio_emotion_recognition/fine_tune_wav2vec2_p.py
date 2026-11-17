import torch
import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import Wav2Vec2FeatureExtractor, WavLMForSequenceClassification, Trainer, TrainingArguments, WavLMModel
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import torchaudio
from google.colab import drive

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"

# Define Google Drive paths
drive_path = "/content/drive/MyDrive/ELEC490/"
csv_path = drive_path + "emotion_features.csv"
save_model_path = drive_path + "wavlm_emotion_model"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
df = pd.read_csv(csv_path)

# Convert emotion labels to numerical values
emotion_mapping = {label: idx for idx, label in enumerate(df["emotion"].unique())}
df["emotion"] = df["emotion"].map(emotion_mapping)

# === BALANCE DATASET USING OVERSAMPLING ===
ros = RandomOverSampler(random_state=42)
df_balanced, _ = ros.fit_resample(df, df["emotion"])
print("Balanced dataset class distribution:")
print(df_balanced["emotion"].value_counts())  # Verify balanced data

# Split into training and test sets
train_df, test_df = train_test_split(df_balanced, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Load smaller WavLM model
model_name = "microsoft/wavlm-base"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
wavlm_model = WavLMModel.from_pretrained(model_name).to(device)

# Function to preprocess inputs
def preprocess_function(examples):
    file_path = examples["file_path"].replace("\\", "/")
    if not file_path.startswith("/content/drive/"):
        file_path = os.path.join("/content/drive/MyDrive/ELEC490/", os.path.basename(file_path))
    
    audio_waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        audio_waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio_waveform)
    audio_waveform = audio_waveform.squeeze(0)
    
    inputs = feature_extractor(audio_waveform.numpy(), return_tensors="pt", sampling_rate=16000).to(device)
    with torch.no_grad():
        wavlm_embeddings = wavlm_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    
    additional_features = np.array([
        examples["pitch"], examples["spectral_centroid"], examples["spectral_bandwidth"],
        examples["rms"], examples["zcr"]
    ], dtype=np.float32)
    
    combined_features = np.concatenate([wavlm_embeddings, additional_features])
    
    return {
        "input_values": torch.tensor(combined_features, dtype=torch.float32),
        "labels": torch.tensor(examples["emotion"], dtype=torch.long)
    }

# Apply preprocessing
train_dataset = train_dataset.map(preprocess_function, remove_columns=["dataset", "file_path"], batched=False)
test_dataset = test_dataset.map(preprocess_function, remove_columns=["dataset", "file_path"], batched=False)

# Load classification model
model = WavLMForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(emotion_mapping),
    problem_type="single_label_classification"
).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="./wavlm-emotion",  
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,  # Adjusted batch size
    per_device_eval_batch_size=4,
    num_train_epochs=5,  # Increased epochs for better training
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    save_total_limit=2,
    fp16=True,
    report_to="none",
)

# Fix `mask_time_length` issue
model.config.mask_time_length = 1

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train model
trainer.train()

# Save model to Google Drive
model.save_pretrained(save_model_path)
print(f"✅ Fine-tuning complete. Model saved to {save_model_path}")
