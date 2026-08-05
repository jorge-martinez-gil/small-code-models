"""Measure deployment efficiency of each small code model.

The paper argues that small code models are attractive because they are cheap to
deploy, yet it reports no efficiency numbers. This script fills that gap with a
fast (minutes, no training) profiling pass that records, per model:

* parameter count (total and trainable);
* model footprint in memory (MiB) and parameter dtype;
* inference latency over many forward passes: mean, p50, p95 (ms / batch);
* throughput (clone pairs per second) at a fixed batch size;
* peak GPU memory during inference (MiB), when CUDA is available;
* forward-pass FLOPs per pair (measured with fvcore/thop when installed,
  otherwise an analytic 2 * params * seq_len estimate, clearly labelled).

Outputs ``efficiency.json`` and ``efficiency.csv``. Designed to add < 0.2 GPU
hours total to the experimental budget.

Example:
    python scripts/benchmark_efficiency.py \\
        --models codebert graphcodebert codet5 unixcoder plbart polycoder \\
        --output_dir efficiency_out --batch_size 8 --seq_length 512 --iters 50
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from small_code_models.registry import get_model_spec, list_model_specs  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["codebert", "graphcodebert", "codet5", "unixcoder", "plbart", "polycoder"],
        help="Registry keys to profile (default: the six paper models).",
    )
    parser.add_argument("--output_dir", default="efficiency_out")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--iters", type=int, default=50, help="Timed forward passes.")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warmup passes.")
    parser.add_argument("--fp16", action="store_true", help="Profile in float16 (GPU only).")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is present.")
    return parser.parse_args()


def _count_flops(model, input_ids, attention_mask) -> tuple[float | None, str]:
    """Return (flops_per_batch, method). Best-effort, never raises."""
    inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    try:
        from fvcore.nn import FlopCountAnalysis  # type: ignore

        analysis = FlopCountAnalysis(model, (input_ids, attention_mask))
        analysis.unsupported_ops_warnings(False)
        analysis.uncalled_modules_warnings(False)
        return float(analysis.total()), "fvcore"
    except Exception:
        pass
    try:
        from thop import profile  # type: ignore

        flops, _ = profile(model, inputs=(input_ids, attention_mask), verbose=False)
        return float(flops), "thop"
    except Exception:
        pass
    return None, "none"


def _make_synthetic_inputs(tokenizer, model, *, batch_size, seq_length, device):
    """Build fixed-shape inputs that satisfy sequence-classifier token contracts."""
    import torch

    if batch_size < 1 or seq_length < 1:
        raise ValueError("batch_size and seq_length must both be positive")

    vocab = int(getattr(model.config, "vocab_size", tokenizer.vocab_size or 30000))
    vocab = max(vocab, 2)
    generator = torch.Generator().manual_seed(0)
    input_ids = torch.randint(
        0,
        vocab,
        (batch_size, seq_length),
        generator=generator,
    )

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(model.config, "eos_token_id", None)
    if eos_token_id is not None and 0 <= int(eos_token_id) < vocab:
        eos_token_id = int(eos_token_id)
        replacement_id = 0 if eos_token_id != 0 else 1
        input_ids.masked_fill_(input_ids.eq(eos_token_id), replacement_id)
        input_ids[:, -1] = eos_token_id

    input_ids = input_ids.to(device)
    return input_ids, torch.ones_like(input_ids)


def profile_model(
    model_key: str,
    *,
    batch_size: int,
    seq_length: int,
    iters: int,
    warmup: int,
    fp16: bool,
    device: "object",
) -> dict:
    import torch

    from small_code_models.modeling import load_model_and_tokenizer

    spec = get_model_spec(model_key)
    tokenizer, model = load_model_and_tokenizer(spec)
    dtype = torch.float16 if (fp16 and device.type == "cuda") else torch.float32
    model = model.to(device=device, dtype=dtype).eval()

    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    param_bytes = int(sum(p.numel() * p.element_size() for p in model.parameters()))

    input_ids, attention_mask = _make_synthetic_inputs(
        tokenizer,
        model,
        batch_size=batch_size,
        seq_length=seq_length,
        device=device,
    )

    flops_per_batch, flops_method = _count_flops(model, input_ids, attention_mask)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    latencies_ms: list[float] = []
    with torch.no_grad():
        for _ in range(warmup):
            model(input_ids=input_ids, attention_mask=attention_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        for _ in range(iters):
            start = time.perf_counter()
            model(input_ids=input_ids, attention_mask=attention_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies_ms.append((time.perf_counter() - start) * 1000.0)

    latencies_ms.sort()
    mean_ms = float(statistics.mean(latencies_ms))
    p50_ms = float(statistics.median(latencies_ms))
    p95_ms = float(latencies_ms[min(len(latencies_ms) - 1, int(0.95 * len(latencies_ms)))])
    throughput = float(batch_size / (mean_ms / 1000.0)) if mean_ms > 0 else float("nan")

    peak_mem_mib = None
    if device.type == "cuda":
        peak_mem_mib = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    if flops_per_batch is None:
        flops_per_pair = 2.0 * total_params * seq_length
        flops_method = "analytic_estimate(2*params*seq_len)"
    else:
        flops_per_pair = flops_per_batch / batch_size

    result = {
        "model_key": model_key,
        "model_name": spec.display_name,
        "model_id": spec.model_id,
        "architecture": spec.architecture,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "batch_size": batch_size,
        "seq_length": seq_length,
        "total_params": total_params,
        "total_params_millions": round(total_params / 1e6, 2),
        "trainable_params": trainable_params,
        "param_memory_mib": round(param_bytes / (1024 * 1024), 2),
        "latency_mean_ms": round(mean_ms, 3),
        "latency_p50_ms": round(p50_ms, 3),
        "latency_p95_ms": round(p95_ms, 3),
        "throughput_pairs_per_s": round(throughput, 2),
        "peak_gpu_mem_mib": None if peak_mem_mib is None else round(peak_mem_mib, 2),
        "gflops_per_pair": None if flops_per_pair is None else round(flops_per_pair / 1e9, 3),
        "flops_method": flops_method,
    }

    del model
    try:
        import torch as _torch

        if device.type == "cuda":
            _torch.cuda.empty_cache()
    except Exception:
        pass
    return result


def main() -> None:
    args = parse_args()
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("PyTorch is required for efficiency profiling.") from exc

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    )
    print(f"Profiling on device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    valid_keys = {spec.key for spec in list_model_specs()}
    results: list[dict] = []
    for model_key in args.models:
        if model_key not in valid_keys:
            print(f"[skip] unknown model key: {model_key}")
            continue
        print(f"[profile] {model_key} ...", flush=True)
        try:
            results.append(
                profile_model(
                    model_key,
                    batch_size=args.batch_size,
                    seq_length=args.seq_length,
                    iters=args.iters,
                    warmup=args.warmup,
                    fp16=args.fp16,
                    device=device,
                )
            )
        except Exception as exc:  # keep going on a single model failure
            print(f"[fail] {model_key}: {exc}")
            results.append({"model_key": model_key, "error": str(exc)})

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "efficiency.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")

    ok_rows = [r for r in results if "error" not in r]
    if ok_rows:
        fieldnames = list(ok_rows[0].keys())
        with (output_dir / "efficiency.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ok_rows)

    print("\n=== Efficiency summary ===")
    print(
        f"{'model':<16}{'params(M)':>10}{'lat_p50(ms)':>13}"
        f"{'pairs/s':>10}{'GPU MiB':>10}{'GFLOPs/pair':>13}"
    )
    for r in results:
        if "error" in r:
            print(f"{r['model_key']:<16}  ERROR: {r['error']}")
            continue
        print(
            f"{r['model_key']:<16}{r['total_params_millions']:>10}"
            f"{r['latency_p50_ms']:>13}{r['throughput_pairs_per_s']:>10}"
            f"{(r['peak_gpu_mem_mib'] or 0):>10}{(r['gflops_per_pair'] or 0):>13}"
        )
    print(f"\nWrote {output_dir / 'efficiency.json'} and efficiency.csv")


if __name__ == "__main__":
    main()
