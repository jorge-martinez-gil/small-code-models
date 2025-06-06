import os
import random
import numpy as np
import torch
import logging

from transformers import (
    T5Tokenizer,  # Use the slow tokenizer instead of T5TokenizerFast
    T5ForConditionalGeneration,
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
# Load CodeT5 model & tokenizer using the slow tokenizer
# --------------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5-base")

# --------------------------------------------------------------------------------
# Preprocessing function
# --------------------------------------------------------------------------------
def preprocess_function(examples):
    """
    Create a prompt by concatenating code1 and code2 with a prefix.
    The model is tasked with generating the label as text ("0" or "1").
    """
    # Construct the input prompt.
    inputs = [
        "classify: " + code1 + " </s> " + code2 
        for code1, code2 in zip(examples["code1"], examples["code2"])
    ]
    # Tokenize the inputs.
    model_inputs = tokenizer(
        inputs, truncation=True, padding="max_length", max_length=512
    )
    
    # Convert numerical label into a string.
    labels = [str(label) for label in examples["similar"]]
    # Tokenize the target texts.
    labels = tokenizer(
        labels, truncation=True, padding="max_length", max_length=10
    )
    
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# --------------------------------------------------------------------------------
# Load and preprocess the dataset
# --------------------------------------------------------------------------------
dataset = load_dataset('PoolC/1-fold-clone-detection-600k-5fold')

# Instead of using the full dataset, we sample a subset for faster experimentation.
train_indices = random.sample(range(len(dataset['train'])), 10000)
val_indices = random.sample(range(len(dataset['val'])), 2000)

train_dataset = dataset["train"].select(train_indices).map(preprocess_function, batched=True)
val_dataset = dataset["val"].select(val_indices).map(preprocess_function, batched=True)

# --------------------------------------------------------------------------------
# Data Collator (for dynamic padding in seq2seq tasks)
# --------------------------------------------------------------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# --------------------------------------------------------------------------------
# Custom Metrics
# --------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Replace -100 in labels (used for padding) with the tokenizer's pad token id.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Convert the decoded strings into integers.
    try:
        predictions_int = [int(pred.strip()) for pred in decoded_preds]
        labels_int = [int(label.strip()) for label in decoded_labels]
    except ValueError:
        # Fallback in case decoding fails.
        predictions_int = [0] * len(decoded_preds)
        labels_int = [0] * len(decoded_labels)
    
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
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    predict_with_generate=True,  # Enable generation during evaluation.
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none",         # Set to "tensorboard" to log with TensorBoard.
    fp16=True,                # Mixed precision training.
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