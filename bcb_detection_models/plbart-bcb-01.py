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
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
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
    """
    def __init__(self, tokenizer, pairs, labels):
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.pairs[idx][0],
            self.pairs[idx][1],
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
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
# Custom Metrics Function (Fixed `logits` Handling)
# --------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    """
    Compute evaluation metrics: accuracy, F1 score, precision, and recall.
    """
    logits, labels = eval_pred

    # Ensure logits is a NumPy array
    if isinstance(logits, tuple):
        logits = logits[0]  # Extract first element if logits is a tuple

    logits = np.array(logits)  # Convert to NumPy array explicitly

    # Handle potential extra dimensions in logits
    if logits.ndim == 3:  
        logits = logits[:, 0, :]  

    # Compute predictions
    predictions = np.argmax(logits, axis=1)

    # Compute metrics
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
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --------------------------------------------------------------------
    # Use PLBart as a classification model:
    # We use "uclanlp/plbart-base" with AutoModelForSequenceClassification.
    # --------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")
    model = AutoModelForSequenceClassification.from_pretrained("uclanlp/plbart-base", num_labels=2)
    model.to(device)

    # Load code snippets
    code_snippets = load_code_snippets('/content/drive/MyDrive/datasets/bcb/data.jsonl')

    # Prepare datasets
    train_dataset = prepare_dataset('/content/drive/MyDrive/datasets/bcb/train.txt', tokenizer, code_snippets, sample_percentage=1.0)
    val_dataset = prepare_dataset('/content/drive/MyDrive/datasets/bcb/valid.txt', tokenizer, code_snippets, sample_percentage=1.0)
    test_dataset = prepare_dataset('/content/drive/MyDrive/datasets/bcb/test.txt', tokenizer, code_snippets, sample_percentage=1.0)

    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Training arguments
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
        fp16=True,
        group_by_length=True,
        seed=42,
        disable_tqdm=False,
    )

    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Train the model
    trainer.train()

    # Evaluate the model on test data
    test_results = trainer.evaluate(test_dataset)

    print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall: {test_results['eval_recall']:.4f}")
    print(f"F1 Score: {test_results['eval_f1']:.4f}")

if __name__ == "__main__":
    main()