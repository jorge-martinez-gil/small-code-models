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
    PLBartTokenizer,
    PLBartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# --------------------------------------------------------------------------------
# Environment & Logging
# --------------------------------------------------------------------------------
os.environ["WANDB_DISABLED"] = "true"  # Disable Weights & Biases logging
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
# Load PLBart model & tokenizer
# --------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("uclanlp/plbart-base")

# --------------------------------------------------------------------------------
# Preprocessing function
# --------------------------------------------------------------------------------
def preprocess_function(examples):
    """
    For each example, create a prompt by concatenating code1 and code2 with a prefix.
    Convert the numerical label into a string. This prepares the input for our
    seq2seq (text-to-text) classification setup.
    """
    inputs = [
        "classify: " + code1 + " </s> " + code2 
        for code1, code2 in zip(examples["code1"], examples["code2"])
    ]
    # Tokenize inputs with a fixed max_length for consistency
    model_inputs = tokenizer(
        inputs, truncation=True, padding="max_length", max_length=512
    )
    
    # Convert labels (0 or 1) to strings
    labels = [str(label) for label in examples["similar"]]
    # Tokenize labels (target texts)
    with tokenizer.as_target_tokenizer():
        labels_tokenized = tokenizer(
            labels, truncation=True, padding="max_length", max_length=10
        )
    
    model_inputs["labels"] = labels_tokenized["input_ids"]
    return model_inputs

# --------------------------------------------------------------------------------
# Load and preprocess the dataset
# --------------------------------------------------------------------------------
dataset = load_dataset('PoolC/1-fold-clone-detection-600k-5fold')

# For faster experimentation, randomly sample indices
train_indices = random.sample(range(len(dataset['train'])), 10000)
val_indices = random.sample(range(len(dataset['val'])), 2000)

train_dataset = dataset["train"].select(train_indices).map(preprocess_function, batched=True)
val_dataset = dataset["val"].select(val_indices).map(preprocess_function, batched=True)

# --------------------------------------------------------------------------------
# Data Collator (ensures proper padding and batching for seq2seq tasks)
# --------------------------------------------------------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --------------------------------------------------------------------------------
# Custom Metrics
# --------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Replace -100 (the ignore index) with the pad token id for decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Debug: Uncomment these to see the decoded outputs.
    # print("Decoded Predictions:", decoded_preds)
    # print("Decoded Labels:", decoded_labels)
    
    # Convert decoded strings to integers. If conversion fails for an individual
    # example, assign 0 to that example.
    predictions_int = []
    labels_int = []
    for pred, label in zip(decoded_preds, decoded_labels):
        try:
            predictions_int.append(int(pred.strip()))
        except ValueError:
            logging.error(f"Could not convert prediction '{pred}' to int. Defaulting to 0.")
            predictions_int.append(0)
        try:
            labels_int.append(int(label.strip()))
        except ValueError:
            logging.error(f"Could not convert label '{label}' to int. Defaulting to 0.")
            labels_int.append(0)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_int, predictions_int, average='binary', zero_division=0
    )
    acc = accuracy_score(labels_int, predictions_int)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# --------------------------------------------------------------------------------
# Training Arguments
# --------------------------------------------------------------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",            # Use "eval_strategy"
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    predict_with_generate=True,       # Enable generation for evaluation
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",                 # Change to "tensorboard" if desired
    fp16=True,                        # Enable mixed precision training if supported
    seed=42,
)

# --------------------------------------------------------------------------------
# Seq2Seq Trainer
# --------------------------------------------------------------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# --------------------------------------------------------------------------------
# Train & Evaluate
# --------------------------------------------------------------------------------
trainer.train()
eval_results = trainer.evaluate()

print("Evaluation Results:", eval_results)
