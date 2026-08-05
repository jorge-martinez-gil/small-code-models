"""CI smoke test: the no-GPU stats+reporting pipeline runs end to end.

Runs scripts/selftest_pipeline.py against a temp directory. This exercises
analyze_results.py and make_latex_tables.py on a synthetic multi-seed matrix so
the full post-training chain (aggregation, McNemar, paired bootstrap, Friedman +
Nemenyi, LaTeX generation) is guaranteed to work before any real GPU runs.
Needs only NumPy + matplotlib; no torch/sklearn/CUDA.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_selftest_runs(tmp_path) -> None:
    keep = tmp_path / "selftest_out"
    result = subprocess.run(
        [sys.executable, "scripts/selftest_pipeline.py", "--keep", str(keep)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"self-test failed:\n{result.stdout}\n{result.stderr}"
    assert "PASS: end-to-end pipeline is wired correctly." in result.stdout
    assert (keep / "analysis" / "analysis.json").exists()
    assert (keep / "paper_tables" / "tab_ranks.tex").exists()
