!pip install transformers[torch] --quiet
!pip install datasets --quiet
!pip install accelerate -U --quiet

import random
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import precision_recall_fscore_support
import logging


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

# --- Accelerated Clone Detection Dataset (Pre-tokenized using batch tokenization) ---
class CloneDetectionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, num_pairs=None, max_length=256):
        """
        Pre-generates tokenized pairs (in batch) to reduce runtime overhead.
        hf_dataset: Hugging Face dataset split (each example with keys "code", "label", "index")
        tokenizer: GraphCodeBERT tokenizer
        num_pairs: Number of pairs to generate (if None, defaults to len(hf_dataset))
        max_length: Maximum tokenization length
        """
        self.examples = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.num_pairs = num_pairs if num_pairs is not None else len(self.examples)
        
        # Build a mapping from label to indices for positive sampling.
        self.label_to_indices = {}
        for idx, item in enumerate(self.examples):
            label = item["label"]
            self.label_to_indices.setdefault(label, []).append(idx)
        self.all_indices = list(range(len(self.examples)))
        
        # Precompute negative candidates for each label to avoid repeated filtering.
        self.label_to_negative_indices = {}
        for label in self.label_to_indices:
            negatives = [i for i in self.all_indices if self.examples[i]["label"] != label]
            self.label_to_negative_indices[label] = negatives
        
        # Pre-generate code pairs and their labels.
        code1_list, code2_list, labels_list = [], [], []
        for _ in range(self.num_pairs):
            base_idx = random.choice(self.all_indices)
            base_sample = self.examples[base_idx]
            label = base_sample["label"]
            
            if random.random() < 0.5:
                # Attempt a positive pair.
                candidates = self.label_to_indices[label][:]
                if base_idx in candidates:
                    candidates.remove(base_idx)
                if candidates:
                    pair_label = 1  # clone pair
                    pair_idx = random.choice(candidates)
                else:
                    # Fallback to negative if no positive candidate exists.
                    negatives = self.label_to_negative_indices[label]
                    pair_label = 0
                    pair_idx = random.choice(negatives) if negatives else base_idx
            else:
                # Negative pair: different label.
                negatives = self.label_to_negative_indices[label]
                if negatives:
                    pair_label = 0
                    pair_idx = random.choice(negatives)
                else:
                    # Fallback to positive if negatives are unavailable.
                    candidates = self.label_to_indices[label][:]
                    if base_idx in candidates:
                        candidates.remove(base_idx)
                    pair_label = 1
                    pair_idx = random.choice(candidates) if candidates else base_idx

            pair_sample = self.examples[pair_idx]
            code1_list.append(base_sample["code"])
            code2_list.append(pair_sample["code"])
            labels_list.append(pair_label)
        
        # Batch tokenization for all pairs.
        tokenized = self.tokenizer(
            code1_list,
            code2_list,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        tokenized["labels"] = torch.tensor(labels_list, dtype=torch.long)
        self.data = tokenized

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.data.items()}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", pos_label=1
    )
    return {"precision": precision, "recall": recall, "f1": f1}

def main():
    # 1. Load the dataset from Hugging Face.
    raw_datasets = load_dataset("semeru/Code-Code-CloneDetection-POJ104")
    
    # 2. Initialize the GraphCodeBERT tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    
    # 3. Create clone detection datasets (pre-generated pairs).
    train_dataset = CloneDetectionDataset(raw_datasets["train"], tokenizer, num_pairs=10000, max_length=256)
    test_dataset = CloneDetectionDataset(raw_datasets["test"], tokenizer, num_pairs=2000, max_length=256)
    
    # 4. Load the GraphCodeBERT model for sequence classification (2 labels).
    model = AutoModelForSequenceClassification.from_pretrained("microsoft/graphcodebert-base", num_labels=2)
    
    # 5. Move the model to CUDA if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 6. Set up training arguments with acceleration options.
    training_args = TrainingArguments(
        output_dir="./temp_clone_detection",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,  # Helps speed up data transfer to CUDA
        fp16=True,
        logging_strategy="steps",
        logging_steps=10,
		seed=42
    )
    
    # 7. Initialize the Trainer.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 8. Train the model.
    trainer.train()
    
    # 9. Evaluate the model.
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    
if __name__ == "__main__":
    main()