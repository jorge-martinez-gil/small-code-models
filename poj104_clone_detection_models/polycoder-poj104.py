import random
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging

os.environ["WANDB_DISABLED"] = "true"
logging.disable(logging.WARNING)

# Set seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

class CloneDetectionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, num_pairs=None, max_length=512):
        self.examples = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_pairs = num_pairs or len(self.examples)

        # Build an index mapping for each label.
        self.label_to_indices = {}
        for idx, item in enumerate(self.examples):
            self.label_to_indices.setdefault(item["label"], []).append(idx)

        # Create pairs of code snippets along with an integer label.
        # For positive pairs, the label is 1; for negative pairs, it is 0.
        self.pairs = []
        for _ in range(self.num_pairs):
            base_idx = random.choice(range(len(self.examples)))
            base_label = self.examples[base_idx]["label"]

            is_positive = random.random() < 0.5 and len(self.label_to_indices[base_label]) > 1
            if is_positive:
                pair_idx = random.choice([i for i in self.label_to_indices[base_label] if i != base_idx])
                label = 1
            else:
                negative_labels = [lbl for lbl in self.label_to_indices if lbl != base_label]
                negative_label = random.choice(negative_labels)
                pair_idx = random.choice(self.label_to_indices[negative_label])
                label = 0

            # Instead of concatenating the two codes into one string, store them separately.
            code1 = self.examples[base_idx]["code"]
            code2 = self.examples[pair_idx]["code"]
            self.pairs.append((code1, code2, label))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        code1, code2, label = self.pairs[idx]
        # Use the tokenizer's text pair mode
        encoding = self.tokenizer(
            code1,
            code2,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        # Remove the extra batch dimension from each tensor.
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        return encoding

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

def main():
    # Load the dataset from the Hugging Face Hub.
    raw_datasets = load_dataset("semeru/Code-Code-CloneDetection-POJ104")
    tokenizer = AutoTokenizer.from_pretrained("NinedayWang/PolyCoder-160M")
    
    # Ensure a padding token is set.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare training and testing datasets.
    train_dataset = CloneDetectionDataset(raw_datasets["train"], tokenizer, num_pairs=2000, max_length=512)
    test_dataset = CloneDetectionDataset(raw_datasets["test"], tokenizer, num_pairs=500, max_length=512)

    # Load the model for sequence classification with two labels.
    model = AutoModelForSequenceClassification.from_pretrained(
        "NinedayWang/PolyCoder-160M",
        num_labels=2,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

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
        report_to="none",
        fp16=True,
        group_by_length=True,
        seed=42,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_results = trainer.evaluate()
    print("Evaluation Results:", eval_results)

if __name__ == "__main__":
    main()