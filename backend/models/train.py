import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Load data
df = pd.read_csv("sample_text.csv")

texts = df["text"].tolist()
labels = df["label"].astype(float).tolist()

# Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class UrgencyDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=128)
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

dataset = UrgencyDataset(texts, labels)

# Model
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=1
)
model.config.problem_type = "regression"

# Training setup
training_args = TrainingArguments(
    output_dir="./model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")

print("Training complete.")

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# 1. Split data BEFORE creating the dataset
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42
)

# 2. Create two datasets
train_dataset = UrgencyDataset(train_texts, train_labels)
test_dataset = UrgencyDataset(test_texts, test_labels)

# Training setup
training_args = TrainingArguments(
    output_dir="./model",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    eval_strategy="epoch", # Add this to track progress
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset # Give the trainer the test data
)

# 3. Train and then generate predictions
trainer.train()

# This generates the 'predictions' variable you were missing
output = trainer.predict(test_dataset)
predictions = output.predictions.flatten() 
y_test = test_labels # This is your 'real' data

# 4. Now the metrics will work!
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n--- Model Performance ---")
print(f"Average Error (MAE): {mae:.2f}")
print(f"Penalty Score (MSE): {mse:.2f}")
print(f"Model Confidence (R2): {r2:.2f}")

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")