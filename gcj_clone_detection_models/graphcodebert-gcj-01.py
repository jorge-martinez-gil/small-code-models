"""Run GraphCodeBERT clone-detection experiments on the GCJ benchmark."""

from __future__ import annotations

import argparse

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding

from small_code_models.data import build_datasets
from small_code_models.metrics import print_metrics_table
from small_code_models.trainer import CloneDetectionTrainer, get_training_args

MODEL_ID = "microsoft/graphcodebert-base"
MODEL_NAME = "GraphCodeBERT"
DATASET_NAME = "gcj"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a benchmark run.

    Args:
        None.

    Returns:
        Parsed command-line namespace.

    Raises:
        SystemExit: Raised by argparse for invalid CLI usage.
    """
    parser = argparse.ArgumentParser(
        description=f"Run {MODEL_NAME} on {DATASET_NAME} clone detection."
    )
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Directory with data.jsonl, train.txt, valid.txt, and test.txt",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory for model outputs and checkpoints",
    )
    parser.add_argument(
        "--sample_pct",
        type=float,
        default=100.0,
        help="Percentage of each split to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    return parser.parse_args()


def load_model_and_tokenizer() -> tuple[AutoTokenizer, AutoModelForSequenceClassification]:
    """Create tokenizer and sequence-classification model for the configured model id.

    Args:
        None.

    Returns:
        A ``(tokenizer, model)`` tuple.

    Raises:
        OSError: If model assets cannot be resolved from Hugging Face.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(MODEL_ID, num_labels=2)
    if config.pad_token_id is None and tokenizer.pad_token_id is not None:
        config.pad_token_id = tokenizer.pad_token_id

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,
    )
    return tokenizer, model


def main() -> None:
    """Run training and report test metrics.

    Args:
        None.

    Returns:
        None.

    Raises:
        RuntimeError: If training or evaluation fails.
    """
    args = parse_args()
    tokenizer, model = load_model_and_tokenizer()

    train_ds, val_ds, test_ds = build_datasets(
        args.data_dir,
        tokenizer,
        sample_pct=args.sample_pct,
    )

    trainer = CloneDetectionTrainer(
        model=model,
        args=get_training_args(
            args.output_dir,
            num_train_epochs=args.epochs,
            fp16=False,
        ),
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        ),
    )
    test_results = trainer.run(train_ds, val_ds, test_ds)

    metrics = {
        "test": {
            "accuracy": test_results.get("eval_accuracy", 0.0),
            "precision": test_results.get("eval_precision", 0.0),
            "recall": test_results.get("eval_recall", 0.0),
            "f1": test_results.get("eval_f1", 0.0),
        }
    }
    print_metrics_table(metrics)


if __name__ == "__main__":
    main()
