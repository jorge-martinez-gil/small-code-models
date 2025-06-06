# --------------------------------------------------------------------------------
# If running in a fresh Google Colab environment, uncomment the installs below:
# --------------------------------------------------------------------------------
!pip install transformers[torch] --quiet
!pip install datasets --quiet
!pip install accelerate -U --quiet

import os
# Set environment variables to help manage CUDA memory
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import random
import json
import torch
import gc
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    RobertaTokenizer,
    T5ForConditionalGeneration,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging

# Disable unnecessary logging
os.environ["WANDB_DISABLED"] = "true"
logging.disable(logging.WARNING)

# --------------------------------------------------------------------------------
# Custom Dataset Class for Code Clone Detection
# --------------------------------------------------------------------------------
class CodeCloneDataset(Dataset):
    """
    Custom Dataset class for Code Clone Detection from JSONL files.
    Each line in the file is expected to be a JSON object with keys:
      - "src_code": the source code snippet (a string)
      - "tgt_code": the target code snippet (a string)
      - "label": an integer label (e.g., 0 or 1; if -1, it will be converted to 0)
    """
    def __init__(self, file_path, tokenizer, max_length=256, sample_ratio=1.0):
        self.samples = self.load_data(file_path, sample_ratio)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loaded {len(self.samples)} samples from {file_path}")

    def load_data(self, file_path, sample_ratio):
        samples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"ERROR: File not found: {file_path}")
            return samples

        random.shuffle(lines)
        num_samples = int(len(lines) * sample_ratio)
        lines = lines[:num_samples]

        for line in lines:
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping line due to JSON error: {e}\nLine: {line.strip()}")
                continue

            src_code = data.get("src_code", "")
            tgt_code = data.get("tgt_code", "")
            label = data.get("label", None)

            if label is None:
                print(f"Skipping line with missing label: {line.strip()}")
                continue

            try:
                label = int(label)
            except ValueError:
                print(f"Skipping line with non-integer label: {line.strip()}")
                continue

            if label == -1:
                label = 0
            elif label not in [0, 1]:
                print(f"Warning: label {label} is out of expected range for sample: {line.strip()}")

            samples.append((src_code, tgt_code, label))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src_code, tgt_code, label = self.samples[idx]
        text = src_code + " " + tgt_code
        inputs = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --------------------------------------------------------------------------------
# Compute Metrics Function
# --------------------------------------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# --------------------------------------------------------------------------------
# T5 Wrapper for Clone Detection
# --------------------------------------------------------------------------------
class T5ForCloneDetection(torch.nn.Module):
    """
    Wrap the T5 model (T5ForConditionalGeneration) for clone detection.
    The model is prompted with a dummy decoder input and outputs a single token.
    The logits for tokens "0" and "1" (the two classes) are extracted.
    Also, we delegate saving and configuration access to the underlying model.
    """
    def __init__(self, model, tokenizer, class_weights=None):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.class_token_ids = torch.tensor(self.tokenizer.convert_tokens_to_ids(["0", "1"]))
        self.class_weights = class_weights

    def forward(self, input_ids, attention_mask, labels=None):
        batch_size = input_ids.shape[0]
        decoder_start_token_id = self.model.config.decoder_start_token_id
        decoder_input_ids = torch.full(
            (batch_size, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=input_ids.device
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        first_token_logits = outputs.logits[:, 0, :]
        class_token_ids = self.class_token_ids.to(first_token_logits.device)
        classification_logits = first_token_logits[:, class_token_ids]
        loss = None
        if labels is not None:
            if self.class_weights is not None:
                loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(input_ids.device))
            else:
                loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(classification_logits, labels)
        output = {"logits": classification_logits}
        if loss is not None:
            output["loss"] = loss
        return output

    def save_pretrained(self, save_directory, **kwargs):
        # Delegate saving to the underlying model
        return self.model.save_pretrained(save_directory, **kwargs)

    @property
    def config(self):
        # Expose the underlying model's config attribute
        return self.model.config

# --------------------------------------------------------------------------------
# Custom Trainer to Avoid SafeTensors Saving Issues
# --------------------------------------------------------------------------------
from transformers import Trainer

class MyTrainer(Trainer):
    def save_model(self, output_dir=None, _internal_call=False):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        self.model.save_pretrained(output_dir, safe_serialization=False)
        self.tokenizer.save_pretrained(output_dir)

# --------------------------------------------------------------------------------
# Main Training & Evaluation Function
# --------------------------------------------------------------------------------
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load the SalesForce CodeT5 model and tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    base_model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-base')
    
    # Reduce memory consumption:
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    
    base_model = base_model.float().to(device)
    model = T5ForCloneDetection(base_model, tokenizer)
    model.to(device)

    # Define dataset file paths (adjust as needed)
    train_file = '/content/drive/MyDrive/datasets/gcj/train.jsonl'
    valid_file = '/content/drive/MyDrive/datasets/gcj/valid.jsonl'
    test_file  = '/content/drive/MyDrive/datasets/gcj/test.jsonl'

    # Use a low sample ratio to reduce memory usage
    train_dataset = CodeCloneDataset(train_file, tokenizer, sample_ratio=1.0)
    valid_dataset = CodeCloneDataset(valid_file, tokenizer, sample_ratio=1.0)
    test_dataset = CodeCloneDataset(test_file, tokenizer, sample_ratio=1.0)

    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        print(f"Sample {i} label: {sample['labels'].item()}")  # Should be 0 or 1

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    # Training arguments: lower batch size and enable mixed precision
    training_args = TrainingArguments(
        output_dir='results',
        num_train_epochs=3,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        warmup_steps=100,
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

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    torch.cuda.empty_cache()
    gc.collect()

    print("Evaluating on test dataset...")
    test_results = trainer.evaluate(test_dataset)

    print("Final Test Metrics:")
    print(f"Accuracy: {test_results.get('eval_accuracy', 0):.4f}")
    print(f"Precision: {test_results.get('eval_precision', 0):.4f}")
    print(f"Recall: {test_results.get('eval_recall', 0):.4f}")
    print(f"F1 Score: {test_results.get('eval_f1', 0):.4f}")

if __name__ == "__main__":
    main()