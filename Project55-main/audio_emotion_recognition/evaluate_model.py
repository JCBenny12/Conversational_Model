import torch
import os
import numpy as np
import pandas as pd
from datasets import Dataset
from transformers import WavLMForSequenceClassification, Trainer, TrainingArguments, Wav2Vec2FeatureExtractor
from sklearn.metrics import accuracy_score
import torchaudio
from google.colab import drive

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"

# Define Google Drive paths
drive_path = "/content/drive/MyDrive/ELEC490/"
csv_path = drive_path + "emotion_features.csv"
model_path = drive_path + "wavlm_emotion_model"

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load dataset
df = pd.read_csv(csv_path)

# Convert emotion labels to numerical values
emotion_mapping = {label: idx for idx, label in enumerate(df["emotion"].unique())}
df["emotion"] = df["emotion"].map(emotion_mapping)

# Use test set
test_df = df.sample(frac=0.2, random_state=42)
test_dataset = Dataset.from_pandas(test_df)

# Load the fine-tuned model
model = WavLMForSequenceClassification.from_pretrained(model_path, local_files_only=True).to(device)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base")

# Dynamically determine expected feature length
EXPECTED_FEATURE_LENGTH = model.config.hidden_size if hasattr(model.config, "hidden_size") else 46446
print(f"Expected feature length: {EXPECTED_FEATURE_LENGTH}")

def preprocess_function(examples):
    file_path = examples["file_path"].replace("\\", "/")
    if not file_path.startswith("/content/drive/"):
        file_path = os.path.join("/content/drive/MyDrive/ELEC490/", os.path.basename(file_path))
    
    audio_waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        audio_waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(audio_waveform)
    audio_waveform = audio_waveform.squeeze(0)
    
    inputs = feature_extractor(audio_waveform.numpy(), return_tensors="pt", sampling_rate=16000).to(device)
    input_values = inputs["input_values"].squeeze(0).cpu().numpy()
    
    # Ensure fixed length
    if input_values.shape[0] > EXPECTED_FEATURE_LENGTH:
        input_values = input_values[:EXPECTED_FEATURE_LENGTH]  # Truncate
    else:
        pad_length = EXPECTED_FEATURE_LENGTH - input_values.shape[0]
        input_values = np.pad(input_values, (0, pad_length), mode='constant')
    
    return {
        "input_values": input_values,
        "labels": examples["emotion"]
    }

# Apply preprocessing
test_dataset = test_dataset.map(preprocess_function, remove_columns=["dataset", "file_path"], batched=False)

# Use Trainer for evaluation
training_args = TrainingArguments(
    output_dir="./wavlm-emotion",  
    per_device_eval_batch_size=4,
    logging_dir="./logs",
    report_to="none",
)

trainer = Trainer(model=model, args=training_args)
eval_results = trainer.evaluate(test_dataset)

# Compute accuracy
def compute_accuracy(model, dataset):
    true_labels = []
    predictions = []
    model.to(device)

    for example in dataset:
        input_values = torch.tensor(example["input_values"]).unsqueeze(0).to(device)
        true_labels.append(example["labels"])

        with torch.no_grad():
            logits = model(input_values).logits
            predicted_label = torch.argmax(logits, dim=-1).item()
            predictions.append(predicted_label)

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Compute and print accuracy
accuracy = compute_accuracy(model, test_dataset)
eval_results["accuracy"] = accuracy
print("Evaluation Results:", eval_results)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
