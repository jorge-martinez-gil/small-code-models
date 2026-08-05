"""Run any registered model on any normalized clone-detection benchmark."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Keep third-party libraries quiet by default. On Colab/CI this driver is
# launched without a TTY (e.g. via `!bash` or subprocess), where progress bars
# from model downloads, tokenization and per-step training emit a new line on
# every update. Across a large matrix that output floods the notebook front-end
# until the browser tab freezes. These are only defaults -- exporting the same
# variables before launching still takes precedence. They must be set before
# transformers/huggingface_hub are imported (which happens inside main()).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVANCED_TQDM", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_code_models.registry import (
    get_benchmark_spec,
    get_model_spec,
    list_benchmark_specs,
    list_model_specs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a registered code model on a normalized clone benchmark."
    )
    parser.add_argument("--model", help="Model registry key")
    parser.add_argument("--benchmark", help="Benchmark registry key")
    parser.add_argument("--data_dir", help="Directory with data.jsonl and split files")
    parser.add_argument("--output_dir", help="Directory for outputs and artifacts")
    parser.add_argument("--model_path", help="Optional local checkpoint path")
    parser.add_argument("--sample_pct", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--strict_data", action="store_true")
    parser.add_argument("--no_artifacts", action="store_true")
    parser.add_argument("--bootstrap_resamples", type=int, default=1000)
    parser.add_argument("--list_models", action="store_true")
    parser.add_argument("--list_benchmarks", action="store_true")
    return parser.parse_args()


def _print_models() -> None:
    for spec in list_model_specs():
        marker = "runnable" if spec.runnable else "local checkpoint required"
        model_id = spec.model_id or "n/a"
        print(f"{spec.key:18s} {marker:26s} {model_id}")


def _print_benchmarks() -> None:
    for spec in list_benchmark_specs():
        marker = "runnable" if spec.runnable else "protocol"
        print(f"{spec.key:20s} {marker:10s} {spec.expected_layout}")


def main() -> None:
    args = parse_args()
    if args.list_models:
        _print_models()
        return
    if args.list_benchmarks:
        _print_benchmarks()
        return

    missing = [
        name
        for name in ("model", "benchmark", "data_dir", "output_dir")
        if getattr(args, name) is None
    ]
    if missing:
        missing_flags = ", ".join("--" + name for name in missing)
        raise SystemExit(f"Missing required arguments: {missing_flags}")

    model_spec = get_model_spec(args.model)
    benchmark_spec = get_benchmark_spec(args.benchmark)
    if not model_spec.runnable and args.model_path is None:
        raise SystemExit(model_spec.notes)
    if benchmark_spec.expected_layout not in {"pair_jsonl", "problem_directories or pair_jsonl"}:
        raise SystemExit(
            f"{benchmark_spec.display_name} is registered as {benchmark_spec.expected_layout}; "
            "prepare it as pair_jsonl before training."
        )

    import transformers
    from transformers import DataCollatorWithPadding

    # Belt-and-suspenders alongside the env vars above: drop routine INFO log
    # lines and turn off transformers' own progress bars so an unattended run
    # cannot flood the console/notebook output.
    transformers.utils.logging.set_verbosity_warning()
    transformers.utils.logging.disable_progress_bar()

    from small_code_models.data import build_datasets
    from small_code_models.metrics import print_metrics_table
    from small_code_models.modeling import load_model_and_tokenizer
    from small_code_models.trainer import CloneDetectionTrainer, get_training_args

    tokenizer, model = load_model_and_tokenizer(model_spec, model_path=args.model_path)
    train_ds, val_ds, test_ds = build_datasets(
        args.data_dir,
        tokenizer,
        sample_pct=args.sample_pct,
        max_length=args.max_length,
        strict=args.strict_data,
    )

    training_overrides = {
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "fp16": args.fp16,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    if args.learning_rate is not None:
        training_overrides["learning_rate"] = args.learning_rate

    trainer = CloneDetectionTrainer(
        model=model,
        args=get_training_args(args.output_dir, **training_overrides),
        train_dataset=train_ds,
        eval_dataset=val_ds,
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
            "model_key": model_spec.key,
            "model_id": args.model_path or model_spec.model_id,
            "model_name": model_spec.display_name,
            "benchmark_key": benchmark_spec.key,
            "dataset_name": benchmark_spec.display_name,
            "sample_pct": args.sample_pct,
            "epochs": args.epochs,
            "seed": args.seed,
            "max_length": args.max_length,
            "strict_data": args.strict_data,
        },
        write_artifacts=not args.no_artifacts,
        bootstrap_resamples=args.bootstrap_resamples,
    )

    print_metrics_table(
        {
            "test": {
                key[5:]: value
                for key, value in test_results.items()
                if key.startswith("eval_")
            }
        }
    )


if __name__ == "__main__":
    main()
