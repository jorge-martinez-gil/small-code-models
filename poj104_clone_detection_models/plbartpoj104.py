!pip install transformers[torch] --quiet
!pip install datasets --quiet
!pip install accelerate -U --quiet

import random
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from sklearn.metrics import precision_recall_fscore_support
import logging

os.environ["WANDB_DISABLED"] = "true"  # Disable W&B logging
logging.disable(logging.WARNING)

# Set seed for reproducibility
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Adapted Clone Detection Dataset for PLBART ---
class CloneDetectionDataset(Dataset):
    def __init__(self, hf_dataset, tokenizer, num_pairs=None, max_input_length=256, max_target_length=10):
        """
        Pre-generates prompts and target texts for clone detection.
        hf_dataset: Hugging Face dataset split with keys "code", "label", "index"
        tokenizer: PLBART tokenizer
        num_pairs: Number of pairs to generate (if None, defaults to len(hf_dataset))
        max_input_length: Maximum token length for the prompt
        max_target_length: Maximum token length for the target text ("0" or "1")
        """
        self.examples = hf_dataset
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.num_pairs = num_pairs if num_pairs is not None else len(self.examples)
        
        # Build mapping from label to indices for positive sampling.
        self.label_to_indices = {}
        for idx, item in enumerate(self.examples):
            label = item["label"]
            self.label_to_indices.setdefault(label, []).append(idx)
        self.all_indices = list(range(len(self.examples)))
        
        # Precompute negative candidates for each label.
        self.label_to_negative_indices = {}
        for label in self.label_to_indices:
            negatives = [i for i in self.all_indices if self.examples[i]["label"] != label]
            self.label_to_negative_indices[label] = negatives
        
        # Pre-generate code pairs and corresponding prompt/target texts.
        self.prompts = []
        self.targets = []
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
                    pair_label = 1  # Clone pair.
                    pair_idx = random.choice(candidates)
                else:
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
                    candidates = self.label_to_indices[label][:]
                    if base_idx in candidates:
                        candidates.remove(base_idx)
                    pair_label = 1
                    pair_idx = random.choice(candidates) if candidates else base_idx

            pair_sample = self.examples[pair_idx]
            code1 = base_sample["code"]
            code2 = pair_sample["code"]
            # Create a prompt by concatenating the two code snippets.
            input_text = f"code pair: {code1} </s> {code2}"
            self.prompts.append(input_text)
            # Target label: "1" for clones, "0" otherwise.
            self.targets.append("1" if pair_label == 1 else "0")
        
    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        input_text = self.prompts[idx]
        target_text = self.targets[idx]
        tokenized_input = self.tokenizer(
            input_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_input_length,
            return_tensors="pt"
        )
        tokenized_target = self.tokenizer(
            target_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_target_length,
            return_tensors="pt"
        )
        # Remove extra batch dimensions.
        item = {key: val.squeeze(0) for key, val in tokenized_input.items()}
        item["labels"] = tokenized_target["input_ids"].squeeze(0)
        return item

def main():
    # 1. Load the dataset.
    raw_datasets = load_dataset("semeru/Code-Code-CloneDetection-POJ104")
    
    # 2. Initialize the PLBART tokenizer.
    tokenizer = AutoTokenizer.from_pretrained("uclanlp/plbart-base")
    
    # 3. Create clone detection datasets with pre-generated pairs.
    train_dataset = CloneDetectionDataset(raw_datasets["train"], tokenizer, num_pairs=10000, max_input_length=256, max_target_length=10)
    test_dataset = CloneDetectionDataset(raw_datasets["test"], tokenizer, num_pairs=2000, max_input_length=256, max_target_length=10)
    
    # 4. Load the PLBART model for conditional generation.
    model = AutoModelForSeq2SeqLM.from_pretrained("uclanlp/plbart-base")
    
    # 5. Move the model to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 6. Define a compute_metrics function that decodes predictions.
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        pred_labels = [pred.strip() for pred in decoded_preds]
        true_labels = [label.strip() for label in decoded_labels]
        # Convert decoded labels to binary values.
        pred_binary = [1 if label == "1" else 0 for label in pred_labels]
        true_binary = [1 if label == "1" else 0 for label in true_labels]
        precision, recall, f1, _ = precision_recall_fscore_support(true_binary, pred_binary, average="binary", pos_label=1)
        accuracy = np.mean(np.array(pred_binary) == np.array(true_binary))
        return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}

    
    # 7. Set up Seq2Seq training arguments.
    training_args = Seq2SeqTrainingArguments(
        output_dir="./temp_clone_detection_plbart",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        predict_with_generate=True,  # Enables sequence generation during evaluation.
        save_strategy="no",
        dataloader_num_workers=4,
        dataloader_pin_memory=True,  # Speeds up data transfer to GPU.
        fp16=True,
        logging_strategy="steps",
        logging_steps=50,
		seed=42
    )
    
    # 8. Initialize the Seq2SeqTrainer.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    
    # 9. Train the model.
    trainer.train()
    
    # 10. Evaluate the model.
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)
    
if __name__ == "__main__":
    main()
