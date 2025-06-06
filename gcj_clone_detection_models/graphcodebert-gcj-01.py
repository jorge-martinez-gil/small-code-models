# --------------------------------------------------------------------------------
# If running in a fresh Google Colab environment, uncomment the installs below:
# --------------------------------------------------------------------------------
!pip install transformers[torch] --quiet
!pip install datasets --quiet
!pip install accelerate -U --quiet

import os
import random
import json
import torch
import gc
import numpy as np
from torch.utils.data import Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import logging


# Clear CUDA cache and run garbage collection
torch.cuda.empty_cache()
gc.collect()

# --------------------------------------------------------------------------------
# Environment & Logging
# --------------------------------------------------------------------------------
os.environ["WANDB_DISABLED"] = "true"  # Disable W&B if you do not wish to log there
logging.disable(logging.WARNING)

class CodeCloneDataset(Dataset):
    """
    Custom Dataset class for Code Clone Detection from JSONL files.
    Each line in the file is expected to be a JSON object with keys:
      - "src_code": the source code snippet (a string)
      - "tgt_code": the target code snippet (a string)
      - "label": an integer label (e.g., 0 or 1; if -1, it will be converted to 0)
    """
    def __init__(self, file_path, tokenizer, max_length=512, sample_ratio=1.0):
        self.samples = self.load_data(file_path, sample_ratio)
        self.tokenizer = tokenizer
        self.max_length = max_length
        print(f"Loaded {len(self.samples)} samples from {file_path}")

    def load_data(self, file_path, sample_ratio):
        """
        Load data from a JSONL file and sample a ratio of the dataset.
        """
        samples = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"ERROR: File not found: {file_path}")
            return samples

        # Shuffle and sample lines
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

            # Convert -1 to 0; ensure labels are in {0, 1}
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
        # Tokenize the two code snippets as a pair.
        inputs = self.tokenizer(
            src_code, tgt_code, 
            truncation=True, 
            padding='max_length', 
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

def compute_metrics(eval_pred):
    """
    Compute accuracy, precision, recall, and F1 score.
    This function is called at each evaluation step.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    
    # Return metrics in a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load pre-trained CodeBERT tokenizer and model.
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
    model = RobertaForSequenceClassification.from_pretrained("microsoft/graphcodebert-base", num_labels=2)
    model.to(device)

    # Paths to the JSONL dataset files
    train_file = '/content/drive/MyDrive/datasets/gcj/train.jsonl'
    valid_file = '/content/drive/MyDrive/datasets/gcj/valid.jsonl'
    test_file = '/content/drive/MyDrive/datasets/gcj/test.jsonl'

    # Create datasets (sample_ratio set to 1.0 to use the full dataset; adjust as needed)
    train_dataset = CodeCloneDataset(train_file, tokenizer, sample_ratio=1.0)
    valid_dataset = CodeCloneDataset(valid_file, tokenizer, sample_ratio=1.0) 
    test_dataset = CodeCloneDataset(test_file, tokenizer, sample_ratio=1.0) 

    # (Optional) Print a few labels from the train dataset for a sanity check
    for i in range(min(5, len(train_dataset))):
        sample = train_dataset[i]
        print(f"Sample {i} label: {sample['labels'].item()}")  # Should print 0 or 1

    # Set up training arguments.
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",  # Evaluates at the end of each epoch
        save_strategy="epoch",
        logging_steps=10,
        load_best_model_at_end=True,
        seed=42,           # For reproducibility.
        no_cuda=False,     # Set to True to force CPU usage (if needed).
    )

    # Create Trainer instance with the custom compute_metrics function.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # This will log intermediate metrics.
    )

    # Train the model. Intermediate evaluation metrics will be printed at the end of each epoch.
    print("Starting training...")
    trainer.train()

    # Evaluate the model on the test dataset using predict to get raw predictions.
    print("Evaluating on test dataset...")
    predictions_output = trainer.predict(test_dataset)
    predictions = predictions_output.predictions
    labels = predictions_output.label_ids

    # Extract the predicted labels.
    preds = np.argmax(predictions, axis=1)

    # Calculate accuracy, precision, recall, and F1 score.
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')

    # Print the final results.
    print(f"Final Test Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()