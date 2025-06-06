# --------------------------------------------------------------------------------
# Installation (uncomment if running in a fresh environment like Google Colab)
# --------------------------------------------------------------------------------
!pip install transformers[torch] --quiet
!pip install datasets --quiet
!pip install accelerate -U --quiet
#!pip install huggingface_hub --quiet



import os
import random
import numpy as np
import torch
import logging

#from huggingface_hub import login
#login("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")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,  # Use the classification model
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# --------------------------------------------------------------------------------
# Environment & Logging
# --------------------------------------------------------------------------------
os.environ["WANDB_DISABLED"] = "true"  # Disable logging to Weights & Biases
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
# Load model & tokenizer for sequence classification
# --------------------------------------------------------------------------------
# Load PolyCoder
tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-160M")

# Set a padding token if one isn't defined.
if tokenizer.pad_token is None:
    # Use EOS token as padding token
    tokenizer.pad_token = tokenizer.eos_token

# Update model config with the tokenizer's pad token id.
model = AutoModelForSequenceClassification.from_pretrained("NinedayWang/PolyCoder-160M", num_labels=2)
model.config.pad_token_id = tokenizer.pad_token_id

# Optionally, adjust the token embeddings if you've added a new token.
model.resize_token_embeddings(len(tokenizer))

model.to(device)

# --------------------------------------------------------------------------------
# Preprocessing function
# --------------------------------------------------------------------------------
def preprocess_function(examples):
    """
    Tokenize pairs (code1, code2) using the UniXcoder tokenizer.
    This function processes the input pair and applies truncation.
    """
    return tokenizer(
        examples["code1"],
        examples["code2"],
        truncation=True,
        max_length=512  # Specify a maximum length to avoid overly long sequences
    )

# --------------------------------------------------------------------------------
# Load and preprocess the dataset
# --------------------------------------------------------------------------------
dataset = load_dataset('PoolC/1-fold-clone-detection-600k-5fold')

# For faster experimentation, randomly sample indices.
train_indices = random.sample(range(len(dataset['train'])), 10000)
val_indices = random.sample(range(len(dataset['val'])), 2000)

train_dataset = dataset["train"].select(train_indices).map(preprocess_function, batched=True)
val_dataset = dataset["val"].select(val_indices).map(preprocess_function, batched=True)

# Rename the 'similar' column to 'labels' for Trainer compatibility.
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='binary', zero_division=0
    )
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
    report_to="none",         # Change to "tensorboard" to enable TensorBoard logging.
    fp16=True,                # Enable mixed precision training on GPUs.
    group_by_length=True,     # Group sequences of similar lengths for efficiency.
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
