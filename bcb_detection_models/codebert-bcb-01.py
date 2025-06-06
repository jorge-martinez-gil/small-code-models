# --------------------------------------------------------------------------------
# Installation (uncomment if running in a fresh environment like Google Colab)
# --------------------------------------------------------------------------------
!pip install transformers[torch] --quiet
!pip install datasets --quiet
!pip install accelerate -U --quiet

import os
import random
import json
import gc
import torch
import logging
import numpy as np

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)

from sklearn.metrics import f1_score, precision_score, recall_score

# Clear CUDA cache and run garbage collection
torch.cuda.empty_cache()
gc.collect()

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

# --------------------------------------------------------------------------------
# Data Loading Functions
# --------------------------------------------------------------------------------
def load_code_snippets(jsonl_path):
    """
    Load code snippets from a JSONL file.
    """
    code_snippets = {}
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            code_snippets[data["idx"]] = data["func"]
    return code_snippets

def load_dataset(txt_path, code_snippets, sample_percentage=100.0):
    """
    Load dataset from a TXT file and sample a percentage of the data.
    """
    all_pairs = []
    all_labels = []
    with open(txt_path, 'r') as file:
        for line in file:
            id1, id2, label = line.strip().split('\t')
            code1 = code_snippets.get(id1, "")
            code2 = code_snippets.get(id2, "")
            if code1 and code2:
                all_pairs.append((code1, code2))
                all_labels.append(int(label))

    sample_size = int(len(all_labels) * sample_percentage / 100)
    indices = random.sample(range(len(all_labels)), sample_size)
    pairs = [all_pairs[i] for i in indices]
    labels = [all_labels[i] for i in indices]

    return pairs, labels

# --------------------------------------------------------------------------------
# Custom Dataset using dynamic padding (no fixed padding here)
# --------------------------------------------------------------------------------
from torch.utils.data import Dataset

class CloneDetectionDataset(Dataset):
    """
    Custom Dataset class for clone detection.
    Note: We remove fixed padding here so that our data collator can apply dynamic padding.
    """
    def __init__(self, tokenizer, pairs, labels):
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Use truncation and max_length, but do not pad here.
        encoding = self.tokenizer(
            self.pairs[idx][0],
            self.pairs[idx][1],
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        # Squeeze to remove extra dimensions; dynamic padding will be applied later.
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def prepare_dataset(filepath, tokenizer, code_snippets, sample_percentage=100.0):
    """
    Prepare dataset by loading and tokenizing the data.
    """
    pairs, labels = load_dataset(filepath, code_snippets, sample_percentage)
    return CloneDetectionDataset(tokenizer, pairs, labels)

# --------------------------------------------------------------------------------
# Custom Metrics Function
# --------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics: accuracy, F1 score, precision, and recall.
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = np.mean(predictions == labels)
    f1 = f1_score(labels, predictions, average='binary')
    precision = precision_score(labels, predictions, average='binary')
    recall = recall_score(labels, predictions, average='binary')
    
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

# --------------------------------------------------------------------------------
# Main Training Function
# --------------------------------------------------------------------------------
def main():
    # Load the GraphCodeBERT tokenizer and model
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

    # Load code snippets from your JSONL file on Google Drive
    code_snippets = load_code_snippets('/content/drive/MyDrive/datasets/bcb/data.jsonl')

    # Load and prepare datasets
    train_dataset = prepare_dataset('/content/drive/MyDrive/datasets/bcb/train.txt', tokenizer, code_snippets, sample_percentage=1.0)
    val_dataset = prepare_dataset('/content/drive/MyDrive/datasets/bcb/valid.txt', tokenizer, code_snippets, sample_percentage=1.0)
    test_dataset = prepare_dataset('/content/drive/MyDrive/datasets/bcb/test.txt', tokenizer, code_snippets, sample_percentage=1.0)

    # --------------------------------------------------------------------------------
    # Data Collator for Dynamic Padding
    # --------------------------------------------------------------------------------
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # --------------------------------------------------------------------------------
    # Training Arguments (progress bar enabled)
    # --------------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir='results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=True,                # Enable mixed precision training
        group_by_length=True,     # Group samples of similar lengths for efficiency
        seed=42,                  # Ensure reproducibility
        disable_tqdm=False,       # Ensure that TQDM progress bars are not disabled
        # Omitting report_to to use default progress reporting
    )

    # --------------------------------------------------------------------------------
    # Initialize Trainer
    # --------------------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,  # Use our dynamic padding collator
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # --------------------------------------------------------------------------------
    # Train & Evaluate
    # --------------------------------------------------------------------------------
    trainer.train()
    test_results = trainer.evaluate(test_dataset)

    # Print evaluation results
    print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall: {test_results['eval_recall']:.4f}")
    print(f"F1 Score: {test_results['eval_f1']:.4f}")

if __name__ == "__main__":
    main()
