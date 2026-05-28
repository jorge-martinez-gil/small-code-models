"""Reproducibility utilities for benchmark runs."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


def set_reproducible_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch when available."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def package_version(package_name: str) -> str | None:
    """Return the installed package version, or ``None`` if unavailable."""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_revision(cwd: str | Path | None = None) -> dict[str, Any]:
    """Collect the current Git revision and dirty state when Git is available."""
    root = Path(cwd or Path.cwd())

    def _run_git(args: list[str]) -> str | None:
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--porcelain"])
    return {
        "commit": commit,
        "branch": branch,
        "dirty": bool(status),
    }


def collect_environment(cwd: str | Path | None = None) -> dict[str, Any]:
    """Collect environment metadata useful for reproducing a run."""
    packages = {
        name: package_version(name)
        for name in (
            "accelerate",
            "datasets",
            "numpy",
            "scikit-learn",
            "torch",
            "transformers",
        )
    }

    cuda: dict[str, Any] = {
        "available": False,
        "device_count": 0,
        "devices": [],
    }
    try:
        import torch

        cuda["available"] = bool(torch.cuda.is_available())
        cuda["device_count"] = int(torch.cuda.device_count())
        cuda["torch_cuda_version"] = torch.version.cuda
        cuda["devices"] = [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
    except ImportError:
        cuda["torch_cuda_version"] = None

    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "packages": packages,
        "cuda": cuda,
        "git": git_revision(cwd),
    }
