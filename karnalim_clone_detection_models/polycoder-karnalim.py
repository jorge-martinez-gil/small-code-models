# --------------------------------------------------------------------------------
# Installation (uncomment if running in a fresh environment like Google Colab)
# --------------------------------------------------------------------------------
!pip install transformers[torch] --quiet
!pip install datasets --quiet
!pip install accelerate -U --quiet

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,  # Use the classification model
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
import torch
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import json
import logging
import random

# Disable warnings and logging messages
logging.disable(logging.WARNING)

# Set seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def load_code_snippets(json_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    random.shuffle(data)
    total = len(data)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    return train_data, val_data, test_data

def prepare_dataset(code_snippets, tokenizer):
    pairs, labels = [], []
    for snippet in code_snippets:
        code1 = snippet['code1']
        code2 = snippet['code2']
        label = snippet['score']
        pairs.append((code1, code2))
        labels.append(label)
    return CloneDetectionDataset(tokenizer, pairs, labels)

class CloneDetectionDataset(Dataset):
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
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='binary'),
        'precision': precision_score(labels, predictions, average='binary'),
        'recall': recall_score(labels, predictions, average='binary'),
    }

def main():
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
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data, val_data, test_data = load_code_snippets('/content/drive/MyDrive/datasets/karnalim/data.json', 0.70, 0.15, 0.15)
    train_dataset = prepare_dataset(train_data, tokenizer)
    val_dataset = prepare_dataset(val_data, tokenizer)
    test_dataset = prepare_dataset(test_data, tokenizer)

    training_args = TrainingArguments(
        output_dir='results',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    test_results = trainer.evaluate(test_dataset)

    print(f"Accuracy: {test_results['eval_accuracy']:.4f}")
    print(f"Precision: {test_results['eval_precision']:.4f}")
    print(f"Recall: {test_results['eval_recall']:.4f}")
    print(f"F1 Score: {test_results['eval_f1']:.4f}")

if __name__ == "__main__":
    main()