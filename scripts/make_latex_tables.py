"""Generate drop-in LaTeX tables from analysis + efficiency outputs.

Consumes ``analysis.json`` (from scripts/analyze_results.py) and, optionally,
``efficiency.json`` (from scripts/benchmark_efficiency.py) and writes ready-to-
paste ``.tex`` files using the same ``|l|c|...|`` + ``\\hline`` tabular style as
manuscript_v2.tex:

    tab_multiseed_<metric>.tex   mean +/- std per (model, dataset)
    tab_significance.tex         vs-best McNemar / paired-bootstrap per dataset
    tab_ranks.tex                Friedman avg ranks + Nemenyi critical difference
    tab_efficiency.tex           params / latency / throughput / memory / FLOPs

The script never trains and has no heavy dependencies.

Example:
    python scripts/make_latex_tables.py results_multiseed/analysis/analysis.json \\
        --efficiency efficiency_out/efficiency.json --output_dir paper_tables
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from small_code_models.registry import MODEL_REGISTRY

    _DISPLAY = {k: v.display_name for k, v in MODEL_REGISTRY.items()}
except Exception:  # pragma: no cover
    _DISPLAY = {}

_DATASET_DISPLAY = {
    "bcb": "BCB",
    "karnalim": "Karnalim",
    "poj104": "POJ104",
    "poolc": "PoolC",
    "gcj": "GCJ",
}


def display_model(key: str) -> str:
    return _DISPLAY.get(key, key)


def display_dataset(key: str) -> str:
    return _DATASET_DISPLAY.get(key, key)


def _tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "--"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return _tex_escape(str(value))
    if f != f:  # NaN
        return "--"
    return f"{f:.{digits}f}"


def _pval(value) -> str:
    if value is None or value != value:
        return "--"
    f = float(value)
    if f < 0.001:
        return r"$<$0.001"
    return f"{f:.3f}"


def table_multiseed(analysis: dict, metric: str) -> str:
    agg = analysis["aggregate"]
    models = sorted({v["model"] for v in agg.values()})
    datasets = sorted({v["benchmark"] for v in agg.values()})
    lut = {(v["model"], v["benchmark"]): v for v in agg.values()}

    col = "|l|" + "c|" * len(datasets)
    head = " & ".join([r"\textbf{Model}"] + [rf"\textbf{{{display_dataset(d)}}}" for d in datasets])
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        rf"\begin{{tabular}}{{{col}}}",
        r"\hline",
        head + r" \\",
        r"\hline",
    ]
    for model in models:
        cells = [_tex_escape(display_model(model))]
        for dataset in datasets:
            rec = lut.get((model, dataset))
            if rec is None:
                cells.append("--")
                continue
            stats = rec["metrics"].get(metric, {})
            mean = stats.get("mean")
            std = stats.get("std")
            cells.append(f"{_fmt(mean)} $\\pm$ {_fmt(std)}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{Mean {metric.upper()} $\pm$ standard deviation across seeds "
        r"(higher is better). Each cell aggregates the multi-seed runs for that "
        r"model and dataset.}",
        rf"\label{{tab:multiseed_{metric}}}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table_significance(analysis: dict) -> str:
    sig = analysis.get("significance_per_dataset", {})
    metric = analysis.get("primary_metric", "f1")
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{|l|l|c|c|c|c|}",
        r"\hline",
        r"\textbf{Dataset} & \textbf{Model vs. best} & "
        rf"\textbf{{$\Delta$ {metric.upper()}}} & "
        r"\textbf{McNemar $p$} & \textbf{McNemar $p_{\mathrm{Holm}}$} & "
        r"\textbf{Bootstrap $p$} \\",
        r"\hline",
    ]
    for dataset in sorted(sig):
        payload = sig[dataset]
        reference = payload.get("reference", "")
        comparisons = payload.get("comparisons", [])
        if not comparisons:
            continue
        first = True
        for c in comparisons:
            if "error" in c:
                continue
            boot = c["paired_bootstrap"]
            mc = c["mcnemar"]
            dataset_cell = (
                rf"{display_dataset(dataset)} (best: {_tex_escape(display_model(reference))})"
                if first
                else ""
            )
            first = False
            delta = boot["observed_difference"]
            mc_holm = mc.get("p_value_holm")
            sig_mark = r"$^{*}$" if (mc_holm is not None and mc_holm < 0.05) else ""
            lines.append(
                f"{dataset_cell} & {_tex_escape(display_model(c['model']))} & "
                f"{delta:+.3f}{sig_mark} & {_pval(mc['p_value'])} & "
                f"{_pval(mc_holm)} & {_pval(boot['p_value'])} \\\\"
            )
        lines.append(r"\hline")
    lines += [
        r"\end{tabular}",
        r"\caption{Pairwise statistical comparison of each model against the "
        r"best-performing model on each dataset, using the median seed. "
        r"$\Delta$ is the candidate-minus-best metric difference. McNemar "
        r"$p_{\mathrm{Holm}}$ is the Holm-Bonferroni family-wise-corrected "
        r"$p$-value across the comparisons on that dataset; "
        r"$^{*}$ denotes $p_{\mathrm{Holm}}<0.05$.}",
        r"\label{tab:significance}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table_ranks(analysis: dict) -> str:
    fr = analysis.get("cross_dataset_friedman_nemenyi", {})
    if fr.get("skipped"):
        return f"% Friedman table skipped: {fr.get('reason')}\n"
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{|l|c|}",
        r"\hline",
        r"\textbf{Model} & \textbf{Average rank} \\",
        r"\hline",
    ]
    for item in fr.get("average_ranks", []):
        lines.append(f"{_tex_escape(display_model(item['model']))} & {item['avg_rank']:.2f} \\\\")
    cd = fr.get("nemenyi_critical_difference")
    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{Average F1 ranks across datasets (1 = best). "
        rf"Friedman $\chi^2={fr.get('friedman_chi2', float('nan')):.3f}$ "
        rf"($p={fr.get('friedman_p_value', float('nan')):.3f}$); "
        rf"Nemenyi critical difference $={'%.3f' % cd if cd is not None else 'n/a'}$ "
        rf"at $\alpha={fr.get('alpha', 0.05)}$. Two models differ significantly "
        r"when their average-rank gap exceeds the critical difference.}",
        r"\label{tab:ranks}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def table_efficiency(efficiency: list) -> str:
    rows = [r for r in efficiency if "error" not in r]
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{|l|c|c|c|c|c|}",
        r"\hline",
        r"\textbf{Model} & \textbf{Params (M)} & \textbf{Latency p50 (ms)} & "
        r"\textbf{Throughput (pairs/s)} & \textbf{Peak GPU (MiB)} & "
        r"\textbf{GFLOPs/pair} \\",
        r"\hline",
    ]
    for r in rows:
        lines.append(
            f"{_tex_escape(display_model(r.get('model_key', '')))} & "
            f"{_fmt(r.get('total_params_millions'), 1)} & "
            f"{_fmt(r.get('latency_p50_ms'), 2)} & "
            f"{_fmt(r.get('throughput_pairs_per_s'), 1)} & "
            f"{_fmt(r.get('peak_gpu_mem_mib'), 0)} & "
            f"{_fmt(r.get('gflops_per_pair'), 2)} \\\\"
        )
    device = rows[0].get("device", "") if rows else ""
    seq = rows[0].get("seq_length", "") if rows else ""
    bs = rows[0].get("batch_size", "") if rows else ""
    lines += [
        r"\hline",
        r"\end{tabular}",
        rf"\caption{{Inference efficiency on {_tex_escape(str(device))} "
        rf"(batch size {bs}, sequence length {seq}). Latency and throughput are "
        r"measured over repeated forward passes after warm-up.}",
        r"\label{tab:efficiency}",
        r"\end{table}",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", help="Path to analysis.json")
    parser.add_argument("--efficiency", default=None, help="Path to efficiency.json (optional)")
    parser.add_argument("--output_dir", default="paper_tables")
    parser.add_argument("--metric", default=None, help="Override primary metric for the main table")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.analysis_json, "r", encoding="utf-8") as handle:
        analysis = json.load(handle)
    metric = args.metric or analysis.get("primary_metric", "f1")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written = []
    p = output_dir / f"tab_multiseed_{metric}.tex"
    p.write_text(table_multiseed(analysis, metric) + "\n", encoding="utf-8")
    written.append(p)

    p = output_dir / "tab_significance.tex"
    p.write_text(table_significance(analysis) + "\n", encoding="utf-8")
    written.append(p)

    p = output_dir / "tab_ranks.tex"
    p.write_text(table_ranks(analysis) + "\n", encoding="utf-8")
    written.append(p)

    if args.efficiency and Path(args.efficiency).exists():
        with open(args.efficiency, "r", encoding="utf-8") as handle:
            efficiency = json.load(handle)
        p = output_dir / "tab_efficiency.tex"
        p.write_text(table_efficiency(efficiency) + "\n", encoding="utf-8")
        written.append(p)

    for path in written:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
