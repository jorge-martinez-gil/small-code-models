"""Cross-seed aggregation and statistical-significance analysis.

This module turns the per-run artifacts written by ``run_clone_experiment.py``
(``metrics.json``, ``predictions.jsonl``, ``run_manifest.json``) into the
publication-grade evidence that journal reviewers expect from an empirical
benchmark study:

* multi-seed aggregation (mean, sample standard deviation, min/max, n);
* pooled non-parametric bootstrap confidence intervals across seeds;
* paired significance tests (McNemar's exact test and a paired bootstrap of the
  metric difference) between models evaluated on the *same* test examples;
* a cross-dataset Friedman omnibus test with the Nemenyi critical-difference
  post-hoc, the standard protocol for comparing classifiers over multiple
  datasets (Demsar, JMLR 2006).

The implementation depends only on the Python standard library and NumPy so it
can run anywhere, including continuous-integration environments without
PyTorch, scikit-learn, or SciPy installed. SciPy is used only when available to
obtain an exact chi-square tail probability; otherwise a self-contained
incomplete-gamma implementation is used.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# Metrics summarised by default. These are read directly from each run's
# ``metrics.json`` (the heavy sklearn-based computation already happened at
# training time), so this module never needs scikit-learn.
DEFAULT_METRICS: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "mcc",
    "roc_auc",
    "pr_auc",
    "expected_calibration_error",
)


# ---------------------------------------------------------------------------
# Loading run artifacts
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunRecord:
    """A single (model, benchmark, seed) evaluation run."""

    run_dir: str
    model_key: str
    benchmark_key: str
    seed: int
    metrics: dict[str, float]
    confidence_intervals: dict[str, dict[str, Any]]
    model_name: str = ""
    dataset_name: str = ""

    @property
    def group_key(self) -> tuple[str, str]:
        return (self.model_key, self.benchmark_key)


def _strip_eval_prefix(metrics: dict[str, Any]) -> dict[str, float]:
    """Normalise ``eval_f1`` -> ``f1`` and keep only finite scalar values."""
    cleaned: dict[str, float] = {}
    for raw_key, value in metrics.items():
        key = raw_key[5:] if raw_key.startswith("eval_") else raw_key
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(numeric) or math.isinf(numeric):
            continue
        cleaned[key] = numeric
    return cleaned


def load_run(run_dir: str | Path) -> RunRecord | None:
    """Load a single run directory, or ``None`` if it is not a finished run."""
    directory = Path(run_dir)
    metrics_path = directory / "metrics.json"
    if not metrics_path.exists():
        return None

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics_payload = json.load(handle)
    metrics = _strip_eval_prefix(metrics_payload.get("metrics", {}))
    intervals = metrics_payload.get("bootstrap_confidence_intervals", {})

    metadata: dict[str, Any] = {}
    manifest_path = directory / "run_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle).get("run_metadata", {})

    model_key = str(metadata.get("model_key") or _infer_from_name(directory.name, 0))
    benchmark_key = str(metadata.get("benchmark_key") or _infer_from_name(directory.name, 1))
    seed = int(metadata.get("seed", 0))

    return RunRecord(
        run_dir=str(directory),
        model_key=model_key,
        benchmark_key=benchmark_key,
        seed=seed,
        metrics=metrics,
        confidence_intervals=intervals,
        model_name=str(metadata.get("model_name", model_key)),
        dataset_name=str(metadata.get("dataset_name", benchmark_key)),
    )


def _infer_from_name(dir_name: str, position: int) -> str:
    """Best-effort fallback parser for ``<model>_<benchmark>[_seed<seed>]`` dirs."""
    stem = dir_name
    for marker in ("_seed", "-seed"):
        if marker in stem:
            stem = stem.split(marker)[0]
            break
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[0] if position == 0 else "_".join(parts[1:])
    return stem


def discover_runs(root: str | Path) -> list[RunRecord]:
    """Find every finished run beneath ``root`` (recursively)."""
    root_path = Path(root)
    records: list[RunRecord] = []
    for metrics_path in sorted(root_path.rglob("metrics.json")):
        record = load_run(metrics_path.parent)
        if record is not None:
            records.append(record)
    return records


def load_predictions(
    run_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, list[str]]:
    """Return ``(labels, predictions, scores, keys)`` from ``predictions.jsonl``.

    ``keys`` are stable example identifiers used to align two models'
    predictions on the same test items before a paired test. The first
    available of ``example_id``, ``pair_id``, or the row ``index`` is used.
    """
    path = Path(run_dir) / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"No predictions.jsonl in {run_dir}")

    labels: list[int] = []
    predictions: list[int] = []
    scores: list[float] = []
    keys: list[str] = []
    has_scores = True
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels.append(int(row["label"]))
            predictions.append(int(row["prediction"]))
            if "positive_score" in row and row["positive_score"] is not None:
                scores.append(float(row["positive_score"]))
            else:
                has_scores = False
            key = row.get("example_id") or row.get("pair_id")
            keys.append(str(key) if key is not None else str(row.get("index", len(keys))))

    labels_array = np.asarray(labels, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    scores_array = np.asarray(scores, dtype=float) if has_scores and scores else None
    return labels_array, predictions_array, scores_array, keys


# ---------------------------------------------------------------------------
# Multi-seed aggregation
# ---------------------------------------------------------------------------
@dataclass
class MetricSummary:
    """Cross-seed summary statistics for one metric."""

    mean: float
    std: float
    minimum: float
    maximum: float
    n: int
    values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean,
            "std": self.std,
            "min": self.minimum,
            "max": self.maximum,
            "n": self.n,
            "values": self.values,
        }


def summarise_metric(values: Iterable[float]) -> MetricSummary:
    array = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if array.size == 0:
        return MetricSummary(float("nan"), float("nan"), float("nan"), float("nan"), 0, [])
    # Sample standard deviation (ddof=1); 0 for a single seed.
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return MetricSummary(
        mean=float(np.mean(array)),
        std=std,
        minimum=float(np.min(array)),
        maximum=float(np.max(array)),
        n=int(array.size),
        values=[float(v) for v in array],
    )


def aggregate_seeds(
    records: list[RunRecord],
    metrics: tuple[str, ...] = DEFAULT_METRICS,
) -> dict[tuple[str, str], dict[str, MetricSummary]]:
    """Group runs by (model, benchmark) and summarise each metric over seeds."""
    grouped: dict[tuple[str, str], list[RunRecord]] = {}
    for record in records:
        grouped.setdefault(record.group_key, []).append(record)

    summaries: dict[tuple[str, str], dict[str, MetricSummary]] = {}
    for group_key, group_records in grouped.items():
        per_metric: dict[str, MetricSummary] = {}
        for metric in metrics:
            per_metric[metric] = summarise_metric(
                record.metrics.get(metric, float("nan")) for record in group_records
            )
        per_metric["_seeds"] = MetricSummary(
            mean=float("nan"),
            std=float("nan"),
            minimum=float("nan"),
            maximum=float("nan"),
            n=len(group_records),
            values=[float(r.seed) for r in group_records],
        )
        summaries[group_key] = per_metric
    return summaries


# ---------------------------------------------------------------------------
# Paired significance tests (NumPy-only)
# ---------------------------------------------------------------------------
def _align(
    keys_a: list[str],
    keys_b: list[str],
    *arrays_b: np.ndarray,
) -> list[np.ndarray]:
    """Reorder ``arrays_b`` so row i corresponds to ``keys_a[i]``.

    Raises ``ValueError`` if the two key sets are not identical.
    """
    if keys_a == keys_b:
        return list(arrays_b)
    index_b = {key: i for i, key in enumerate(keys_b)}
    if set(keys_a) != set(index_b.keys()) or len(index_b) != len(keys_b):
        raise ValueError(
            "Prediction files cover different / duplicated examples and cannot "
            "be paired. Ensure both runs used the same dataset split and seed."
        )
    order = np.asarray([index_b[key] for key in keys_a], dtype=int)
    return [array[order] for array in arrays_b]


def mcnemar_test(
    labels: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
) -> dict[str, float | int]:
    """McNemar's paired test (exact binomial; normal approx for large counts).

    ``a`` is the baseline, ``b`` the candidate. The reported ``p_value`` is the
    two-sided probability under H0 that the two models are equally accurate.
    """
    correct_a = predictions_a == labels
    correct_b = predictions_b == labels
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(correct_b & ~correct_a))
    discordant = a_only + b_only

    if discordant == 0:
        p_value = 1.0
    elif discordant <= 1000:
        tail = min(a_only, b_only)
        cumulative = sum(math.comb(discordant, k) for k in range(tail + 1))
        p_value = min(1.0, 2.0 * cumulative * (0.5**discordant))
    else:
        z = (abs(b_only - a_only) - 1.0) / math.sqrt(discordant)
        p_value = math.erfc(z / math.sqrt(2.0))

    return {
        "baseline_correct_candidate_wrong": a_only,
        "candidate_correct_baseline_wrong": b_only,
        "discordant_pairs": discordant,
        "p_value": float(p_value),
    }


def holm_bonferroni(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, preserving input order.

    Controls the family-wise error rate across a family of hypotheses (here, the
    pairwise model comparisons reported on one dataset). For sorted ascending
    p(1) <= ... <= p(m), the adjusted value is the running maximum of
    ``min(1, (m - k + 1) * p(k))``. ``None``/NaN entries are passed through.
    """
    indexed = [(i, p) for i, p in enumerate(p_values) if p is not None and p == p]
    m = len(indexed)
    adjusted: list[float | None] = [None] * len(p_values)
    if m == 0:
        return adjusted
    indexed.sort(key=lambda kv: kv[1])
    running = 0.0
    for rank, (orig_index, p) in enumerate(indexed):
        value = min(1.0, (m - rank) * float(p))
        running = max(running, value)  # enforce monotonic non-decreasing
        adjusted[orig_index] = running
    return adjusted


def _f1_from_counts(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return float(2 * tp / denom) if denom > 0 else 0.0


def _metric_from_predictions(
    labels: np.ndarray, predictions: np.ndarray, metric: str
) -> float:
    if metric == "accuracy":
        return float(np.mean(labels == predictions))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    if metric == "f1":
        return _f1_from_counts(tp, fp, fn)
    if metric == "precision":
        return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    if metric == "recall":
        return float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    if metric == "mcc":
        num = tp * tn - fp * fn
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return float(num / den) if den > 0 else 0.0
    raise ValueError(f"Unsupported paired-bootstrap metric: {metric}")


def paired_bootstrap_difference(
    labels: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    *,
    metric: str = "f1",
    n_resamples: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, float | int | None]:
    """Bootstrap the candidate(b)-minus-baseline(a) difference for ``metric``.

    Returns the observed difference, its mean and percentile confidence
    interval, and a two-sided bootstrap p-value (proportion of resamples whose
    difference has the opposite sign of, or equals, zero).
    """
    observed = _metric_from_predictions(labels, predictions_b, metric) - _metric_from_predictions(
        labels, predictions_a, metric
    )
    rng = np.random.default_rng(seed)
    n = labels.shape[0]
    diffs = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        l = labels[idx]
        diffs[i] = _metric_from_predictions(l, predictions_b[idx], metric) - _metric_from_predictions(
            l, predictions_a[idx], metric
        )
    alpha = 1.0 - confidence_level
    lower = float(np.percentile(diffs, 100.0 * alpha / 2.0))
    upper = float(np.percentile(diffs, 100.0 * (1.0 - alpha / 2.0)))
    # Two-sided bootstrap p-value via the proportion of resamples on the
    # opposite side of zero from the observed effect.
    if observed >= 0:
        p_one_sided = float(np.mean(diffs <= 0.0))
    else:
        p_one_sided = float(np.mean(diffs >= 0.0))
    p_value = min(1.0, 2.0 * p_one_sided)
    return {
        "metric": metric,
        "observed_difference": float(observed),
        "ci_lower": lower,
        "ci_upper": upper,
        "p_value": p_value,
        "n_resamples": int(n_resamples),
        "confidence_level": confidence_level,
    }


def compare_models_on_benchmark(
    records: list[RunRecord],
    benchmark_key: str,
    *,
    metric: str = "f1",
    reference: str | None = None,
    n_resamples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    """Pairwise significance of every model vs. the best (or ``reference``) model.

    For each model the seed whose ``metric`` is the median is used as the
    representative prediction file, keeping the test deterministic and avoiding
    cherry-picking the best seed. Predictions are aligned by example id.
    """
    by_model: dict[str, list[RunRecord]] = {}
    for record in records:
        if record.benchmark_key == benchmark_key:
            by_model.setdefault(record.model_key, []).append(record)
    if not by_model:
        return {"benchmark": benchmark_key, "metric": metric, "comparisons": []}

    representative: dict[str, RunRecord] = {}
    for model_key, model_records in by_model.items():
        ordered = sorted(model_records, key=lambda r: r.metrics.get(metric, float("nan")))
        representative[model_key] = ordered[len(ordered) // 2]  # median seed

    if reference is None:
        reference = max(
            representative, key=lambda m: representative[m].metrics.get(metric, float("nan"))
        )

    ref_labels, ref_pred, _, ref_keys = load_predictions(representative[reference].run_dir)
    comparisons: list[dict[str, Any]] = []
    for model_key, record in sorted(representative.items()):
        if model_key == reference:
            continue
        labels, pred, _, keys = load_predictions(record.run_dir)
        try:
            (aligned_pred,) = _align(ref_keys, keys, pred)
            aligned_labels = ref_labels
        except ValueError as exc:
            comparisons.append({"model": model_key, "error": str(exc)})
            continue
        mcnemar = mcnemar_test(aligned_labels, pred_a := ref_pred, aligned_pred)
        boot = paired_bootstrap_difference(
            aligned_labels,
            ref_pred,
            aligned_pred,
            metric=metric,
            n_resamples=n_resamples,
            seed=seed,
        )
        comparisons.append(
            {
                "reference": reference,
                "model": model_key,
                f"reference_{metric}": representative[reference].metrics.get(metric),
                f"model_{metric}": record.metrics.get(metric),
                "mcnemar": mcnemar,
                "paired_bootstrap": boot,
            }
        )

    # Holm-Bonferroni correction across the family of comparisons on this
    # dataset (one family per dataset). Raw p-values are kept; adjusted values
    # are added so the table can report family-wise-error-controlled results.
    valid = [c for c in comparisons if "error" not in c]
    mcnemar_adj = holm_bonferroni([c["mcnemar"]["p_value"] for c in valid])
    boot_adj = holm_bonferroni([c["paired_bootstrap"]["p_value"] for c in valid])
    for comparison, m_adj, b_adj in zip(valid, mcnemar_adj, boot_adj):
        comparison["mcnemar"]["p_value_holm"] = m_adj
        comparison["paired_bootstrap"]["p_value_holm"] = b_adj

    return {
        "benchmark": benchmark_key,
        "metric": metric,
        "reference": reference,
        "multiple_comparison_correction": "holm-bonferroni",
        "n_examples": int(ref_labels.shape[0]),
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------------------
# Cross-dataset omnibus: Friedman + Nemenyi critical difference
# ---------------------------------------------------------------------------
def _chi2_sf(x: float, k: int) -> float:
    """Upper tail of the chi-square distribution with ``k`` dof.

    Uses SciPy when importable; otherwise a regularised upper-incomplete-gamma
    series/continued-fraction implementation (Numerical Recipes style).
    """
    try:  # optional fast path
        from scipy.stats import chi2  # type: ignore

        return float(chi2.sf(x, k))
    except Exception:
        pass
    if x <= 0:
        return 1.0
    a = k / 2.0
    xx = x / 2.0
    # Regularised lower incomplete gamma P(a, xx); Q = 1 - P.
    if xx < a + 1.0:
        term = 1.0 / a
        total = term
        n = a
        for _ in range(500):
            n += 1.0
            term *= xx / n
            total += term
            if abs(term) < abs(total) * 1e-12:
                break
        p = total * math.exp(-xx + a * math.log(xx) - math.lgamma(a))
        return float(1.0 - p)
    # Continued fraction for Q(a, xx).
    tiny = 1e-300
    b = xx + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    q = math.exp(-xx + a * math.log(xx) - math.lgamma(a)) * h
    return float(q)


# Critical values of the Studentized range statistic divided by sqrt(2),
# for the Nemenyi test at alpha=0.05 and 0.10 (Demsar 2006, Table 5).
# Indexed by number of models k (>= 2).
_NEMENYI_Q05 = {
    2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031,
    9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391,
}
_NEMENYI_Q10 = {
    2: 1.645, 3: 2.052, 4: 2.291, 5: 2.460, 6: 2.589, 7: 2.693, 8: 2.780,
    9: 2.855, 10: 2.920, 11: 2.978, 12: 3.030, 13: 3.077, 14: 3.120, 15: 3.159,
}


def friedman_nemenyi(
    score_matrix: np.ndarray,
    model_keys: list[str],
    *,
    alpha: float = 0.05,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Friedman omnibus test with Nemenyi critical difference.

    Args:
        score_matrix: shape ``(n_models, n_datasets)`` of the metric to compare.
        model_keys: model labels, ``len == n_models``.
        alpha: significance level (0.05 or 0.10 supported for the CD table).
        higher_is_better: rank direction (True for F1/accuracy).

    Returns:
        Average ranks per model, the Friedman statistic and p-value, the
        Iman-Davenport F correction, and the Nemenyi critical difference. Two
        models differ significantly iff their average-rank gap exceeds the CD.
    """
    scores = np.asarray(score_matrix, dtype=float)
    if scores.ndim != 2:
        raise ValueError("score_matrix must be 2D (models x datasets).")
    k, n = scores.shape
    if k < 2 or n < 1:
        raise ValueError("Need at least 2 models and 1 dataset.")

    # Rank within each dataset (column); rank 1 = best.
    ranks = np.empty_like(scores)
    for j in range(n):
        column = scores[:, j]
        order = -column if higher_is_better else column
        # Average ranks for ties.
        sorter = np.argsort(order, kind="mergesort")
        ordered_vals = order[sorter]
        column_ranks = np.empty(k, dtype=float)
        i = 0
        while i < k:
            j2 = i
            while j2 + 1 < k and ordered_vals[j2 + 1] == ordered_vals[i]:
                j2 += 1
            avg_rank = (i + j2) / 2.0 + 1.0
            for s in range(i, j2 + 1):
                column_ranks[sorter[s]] = avg_rank
            i = j2 + 1
        ranks[:, j] = column_ranks

    average_ranks = ranks.mean(axis=1)
    chi2_stat = (12.0 * n / (k * (k + 1.0))) * (
        float(np.sum(average_ranks**2)) - k * (k + 1.0) ** 2 / 4.0
    )
    p_value = _chi2_sf(chi2_stat, k - 1)
    # Iman-Davenport F correction (less conservative than chi-square).
    if (n * (k - 1) - chi2_stat) != 0:
        f_stat = (n - 1) * chi2_stat / (n * (k - 1) - chi2_stat)
    else:
        f_stat = float("inf")

    q_table = _NEMENYI_Q05 if abs(alpha - 0.05) < 1e-9 else _NEMENYI_Q10
    q_alpha = q_table.get(k)
    critical_difference = (
        q_alpha * math.sqrt(k * (k + 1.0) / (6.0 * n)) if q_alpha is not None else None
    )

    ranking = sorted(
        ({"model": m, "avg_rank": float(r)} for m, r in zip(model_keys, average_ranks)),
        key=lambda item: item["avg_rank"],
    )
    return {
        "n_models": k,
        "n_datasets": n,
        "average_ranks": ranking,
        "friedman_chi2": float(chi2_stat),
        "friedman_dof": k - 1,
        "friedman_p_value": float(p_value),
        "iman_davenport_f": float(f_stat),
        "alpha": alpha,
        "nemenyi_critical_difference": critical_difference,
        "significant_omnibus": bool(p_value < alpha),
    }


def build_score_matrix(
    summaries: dict[tuple[str, str], dict[str, MetricSummary]],
    metric: str = "f1",
) -> tuple[np.ndarray, list[str], list[str]]:
    """Assemble a (models x datasets) matrix of mean ``metric`` for Friedman."""
    models = sorted({model for model, _ in summaries})
    datasets = sorted({benchmark for _, benchmark in summaries})
    matrix = np.full((len(models), len(datasets)), np.nan, dtype=float)
    for (model, benchmark), per_metric in summaries.items():
        i = models.index(model)
        j = datasets.index(benchmark)
        matrix[i, j] = per_metric.get(metric, MetricSummary(np.nan, np.nan, np.nan, np.nan, 0)).mean
    return matrix, models, datasets


# Holm-Bonferroni and CD-diagram support added for multi-comparison rigor.

