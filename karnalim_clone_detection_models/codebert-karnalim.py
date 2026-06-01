"""Run CodeBERT clone-detection experiments on the KARNALIM benchmark."""

from __future__ import annotations

import argparse

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding

from small_code_models.data import build_datasets
from small_code_models.metrics import print_metrics_table
from small_code_models.trainer import CloneDetectionTrainer, get_training_args

MODEL_ID = "microsoft/codebert-base"
MODEL_NAME = "CodeBERT"
DATASET_NAME = "karnalim"


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
        default=1.0,
        help="Percentage of each split to use",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for training, sampling, and bootstrap intervals",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum token length for each code pair",
    )
    parser.add_argument(
        "--strict_data",
        action="store_true",
        help="Fail fast on malformed pair rows, missing snippets, or invalid labels",
    )
    parser.add_argument(
        "--no_artifacts",
        action="store_true",
        help="Disable metrics.json, predictions.jsonl, and run_manifest.json outputs",
    )
    parser.add_argument(
        "--bootstrap_resamples",
        type=int,
        default=1000,
        help="Bootstrap resamples used for confidence intervals",
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
        max_length=args.max_length,
        strict=args.strict_data,
    )

    trainer = CloneDetectionTrainer(
        model=model,
        args=get_training_args(
            args.output_dir,
            num_train_epochs=args.epochs,
            fp16=False,
            seed=args.seed,
            data_seed=args.seed,
        ),
        data_collator=DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        ),
    )
    test_results = trainer.run(
        train_ds,
        val_ds,
        test_ds,
        run_metadata={
            "model_id": MODEL_ID,
            "model_name": MODEL_NAME,
            "dataset_name": DATASET_NAME,
            "sample_pct": args.sample_pct,
            "epochs": args.epochs,
            "seed": args.seed,
            "max_length": args.max_length,
            "strict_data": args.strict_data,
        },
        write_artifacts=not args.no_artifacts,
        bootstrap_resamples=args.bootstrap_resamples,
    )

    metrics = {
        "test": {
            key[5:]: value
            for key, value in test_results.items()
            if key.startswith("eval_")
        }
    }
    print_metrics_table(metrics)


if __name__ == "__main__":
    main()
