"""Render a Demsar (2006) critical-difference (CD) diagram.

Consumes the ``cross_dataset_friedman_nemenyi`` block written by
``scripts/analyze_results.py`` (inside ``analysis.json``) and draws the standard
CD diagram: a rank axis with each model placed at its average rank (best on the
left) and bold horizontal bars connecting groups of models whose average-rank
difference is below the Nemenyi critical difference -- i.e. models that are
*not* significantly different.

Only matplotlib + NumPy are required; no training, GPU, or SciPy.

Examples:
    # from a full analysis.json
    python scripts/plot_cd_diagram.py results_multiseed/analysis/analysis.json \\
        --output cd_diagram.png

    # directly from the Friedman/Nemenyi payload, or a {model: avg_rank} json
    python scripts/plot_cd_diagram.py friedman.json --output cd.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DISPLAY = {
    "codebert": "CodeBERT", "graphcodebert": "GraphCodeBERT", "codet5": "CodeT5",
    "codet5_base": "CodeT5", "unixcoder": "UniXCoder", "plbart": "PLBART",
    "polycoder": "PolyCoder",
}


def _extract(payload: dict) -> tuple[list[str], list[float], float | None, dict]:
    """Pull (models, avg_ranks, CD, meta) from any supported JSON shape."""
    block = payload
    if "cross_dataset_friedman_nemenyi" in payload:
        block = payload["cross_dataset_friedman_nemenyi"]
    if block.get("skipped"):
        raise SystemExit(f"Friedman analysis was skipped: {block.get('reason')}")
    if "average_ranks" in block:
        ranks = block["average_ranks"]
        models = [r["model"] for r in ranks]
        avg = [float(r["avg_rank"]) for r in ranks]
        cd = block.get("nemenyi_critical_difference")
        meta = {
            "chi2": block.get("friedman_chi2"),
            "p": block.get("friedman_p_value"),
            "alpha": block.get("alpha", 0.05),
            "n_datasets": block.get("n_datasets"),
        }
        return models, avg, cd, meta
    # fallback: a plain {model: avg_rank} dict
    items = sorted(payload.items(), key=lambda kv: kv[1])
    return [k for k, _ in items], [float(v) for _, v in items], None, {}


def _cliques(avg_ranks: np.ndarray, cd: float) -> list[tuple[int, int]]:
    """Maximal groups (by sorted-rank index) with span <= CD; drop subsets."""
    order = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[order]
    groups: list[tuple[int, int]] = []
    n = len(sorted_ranks)
    for i in range(n):
        j = i
        while j + 1 < n and (sorted_ranks[j + 1] - sorted_ranks[i]) <= cd + 1e-9:
            j += 1
        if j > i:
            groups.append((i, j))
    # remove groups fully contained in another
    maximal = [g for g in groups if not any(g != h and h[0] <= g[0] and g[1] <= h[1] for h in groups)]
    return maximal, order


def plot_cd(models, avg_ranks, cd, meta, output: Path, title: str | None) -> None:
    avg = np.asarray(avg_ranks, dtype=float)
    order = np.argsort(avg)
    models = [models[i] for i in order]
    avg = avg[order]
    labels = [DISPLAY.get(m, m) for m in models]

    k = len(models)
    low, high = 1, k  # rank axis bounds (best rank = 1 on the left)

    fig, ax = plt.subplots(figsize=(9.0, 0.6 * k + 2.0))
    ax.set_xlim(high + 0.5, low - 0.5)  # reversed: best (1) on the left
    ax.set_ylim(0, k + 2)
    ax.axis("off")

    axis_y = k + 1
    ax.plot([low, high], [axis_y, axis_y], "k-", lw=1.2)
    for tick in range(low, high + 1):
        ax.plot([tick, tick], [axis_y, axis_y + 0.12], "k-", lw=1.2)
        ax.text(tick, axis_y + 0.32, str(tick), ha="center", va="bottom", fontsize=10)

    # place models: half on each side for readability
    half = (k + 1) // 2
    for idx, (label, rank) in enumerate(zip(labels, avg)):
        left_side = idx < half
        row_y = axis_y - 1 - (idx if left_side else (k - 1 - idx))
        elbow_x = low - 0.3 if left_side else high + 0.3
        text_x = low - 0.4 if left_side else high + 0.4
        ha = "right" if left_side else "left"
        ax.plot([rank, rank], [axis_y, row_y], "k-", lw=0.9)
        ax.plot([rank, elbow_x], [row_y, row_y], "k-", lw=0.9)
        ax.text(text_x, row_y, f"{label}  ({rank:.2f})", ha=ha, va="center", fontsize=10)

    # CD bar
    if cd is not None:
        bar_y = axis_y + 0.85
        ax.plot([low, low + cd], [bar_y, bar_y], "k-", lw=2.2)
        ax.plot([low, low], [bar_y - 0.1, bar_y + 0.1], "k-", lw=2.2)
        ax.plot([low + cd, low + cd], [bar_y - 0.1, bar_y + 0.1], "k-", lw=2.2)
        ax.text(low + cd / 2, bar_y + 0.18, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=10)

        # cliques of non-significantly-different models
        groups, _ = _cliques(avg, cd)
        clique_y = axis_y - 0.35
        for gi, (a, b) in enumerate(groups):
            yy = clique_y - 0.22 * gi
            ax.plot([avg[a] - 0.05, avg[b] + 0.05], [yy, yy], "-", lw=4.0, color="0.25",
                    solid_capstyle="round")

    if title is None:
        bits = []
        if meta.get("chi2") is not None:
            bits.append(f"Friedman $\\chi^2$={meta['chi2']:.2f}")
        if meta.get("p") is not None:
            bits.append(f"p={meta['p']:.3f}")
        if meta.get("n_datasets"):
            bits.append(f"N={meta['n_datasets']} datasets")
        title = "Critical-difference diagram (F1 ranks)"
        if bits:
            title += " -- " + ", ".join(bits)
    ax.set_title(title, fontsize=12, pad=14)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    # also emit a PDF sibling for LaTeX inclusion
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Wrote {output.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", help="analysis.json or a Friedman/ranks JSON")
    parser.add_argument("--output", default="cd_diagram.png")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    payload = json.loads(Path(args.analysis_json).read_text(encoding="utf-8"))
    models, avg_ranks, cd, meta = _extract(payload)
    if cd is None:
        print("[warn] no critical difference in payload; drawing ranks without CD bar")
    plot_cd(models, avg_ranks, cd, meta, Path(args.output), args.title)


if __name__ == "__main__":
    main()
