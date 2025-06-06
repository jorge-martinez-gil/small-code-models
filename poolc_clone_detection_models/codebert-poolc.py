# --------------------------------------------------------------------------------
# Installation (uncomment if running in a fresh environment like Google Colab)
# --------------------------------------------------------------------------------
# !pip install transformers[torch] --quiet
# !pip install datasets --quiet
# !pip install accelerate -U --quiet

import os
import random
import numpy as np
import torch
import logging

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# --------------------------------------------------------------------------------
# Environment & Logging
# --------------------------------------------------------------------------------
os.environ["WANDB_DISABLED"] = "true"  # Disable W&B if you do not wish to log there
logging.disable(logging.WARNING)

# Set seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------------
# Load model & tokenizer
# --------------------------------------------------------------------------------
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# --------------------------------------------------------------------------------
# Preprocessing function
# --------------------------------------------------------------------------------
def preprocess_function(examples):
    """
    Tokenize pairs (code1, code2) and set up labels from 'similar'.
    Uses dynamic padding at collator level, so no need for max_length here
    (unless you strongly prefer truncation).
    """
    return tokenizer(examples["code1"], examples["code2"], truncation=True)

# --------------------------------------------------------------------------------
# Load and preprocess the dataset
# --------------------------------------------------------------------------------
dataset = load_dataset('PoolC/1-fold-clone-detection-600k-5fold')

# Instead of just selecting the first samples, let's randomly sample indices
# for training and validation for better generalization.
train_indices = random.sample(range(len(dataset['train'])), 10000)
val_indices = random.sample(range(len(dataset['val'])), 2000)

train_dataset = dataset["train"].select(train_indices).map(preprocess_function, batched=True)
val_dataset = dataset["val"].select(val_indices).map(preprocess_function, batched=True)

# Convert 'similar' into integer labels
train_dataset = train_dataset.rename_column("similar", "labels")
val_dataset = val_dataset.rename_column("similar", "labels")

# --------------------------------------------------------------------------------
# Data Collator (for dynamic padding)
# --------------------------------------------------------------------------------
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

# --------------------------------------------------------------------------------
# Custom Metrics
# --------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary', zero_division=0)
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --------------------------------------------------------------------------------
# Training Arguments
# --------------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",         # Disable additional logging (like WandB); set to "tensorboard" if desired
    fp16=True,                # Enable mixed precision for faster training on GPUs
    group_by_length=True,     # Group sequences of similar lengths for efficiency
    seed=42,
)

# --------------------------------------------------------------------------------
# Trainer
# --------------------------------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# --------------------------------------------------------------------------------
# Train & Evaluate
# --------------------------------------------------------------------------------
trainer.train()
eval_results = trainer.evaluate()

print("Evaluation Results:", eval_results)