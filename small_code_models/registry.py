"""Central registry of supported models and clone-detection benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelSpec:
    """Metadata needed to load or document a code model."""

    key: str
    display_name: str
    model_id: str | None
    architecture: str
    parameters_m: float | None
    runnable: bool = True
    notes: str = ""
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSpec:
    """Metadata describing a benchmark target and expected local layout."""

    key: str
    display_name: str
    languages: tuple[str, ...]
    benchmark_type: str
    expected_layout: str
    runnable: bool = True
    notes: str = ""
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "codebert": ModelSpec(
        key="codebert",
        display_name="CodeBERT",
        model_id="microsoft/codebert-base",
        architecture="encoder-only",
        parameters_m=125,
        source_url="https://huggingface.co/microsoft/codebert-base",
    ),
    "graphcodebert": ModelSpec(
        key="graphcodebert",
        display_name="GraphCodeBERT",
        model_id="microsoft/graphcodebert-base",
        architecture="encoder-only",
        parameters_m=125,
        source_url="https://huggingface.co/microsoft/graphcodebert-base",
    ),
    "plbart": ModelSpec(
        key="plbart",
        display_name="PLBART",
        model_id="uclanlp/plbart-base",
        architecture="encoder-decoder",
        parameters_m=140,
        source_url="https://huggingface.co/uclanlp/plbart-base",
    ),
    "polycoder": ModelSpec(
        key="polycoder",
        display_name="PolyCoder",
        model_id="NinedayWang/PolyCoder-160M",
        architecture="decoder-only",
        parameters_m=160,
        source_url="https://huggingface.co/NinedayWang/PolyCoder-160M",
    ),
    "unixcoder": ModelSpec(
        key="unixcoder",
        display_name="UniXCoder",
        model_id="microsoft/unixcoder-base",
        architecture="unified encoder-decoder",
        parameters_m=125,
        source_url="https://huggingface.co/microsoft/unixcoder-base",
    ),
    "codet5": ModelSpec(
        key="codet5",
        display_name="CodeT5 Base",
        model_id="Salesforce/codet5-base",
        architecture="encoder-decoder",
        parameters_m=220,
        source_url="https://huggingface.co/Salesforce/codet5-base",
    ),
    "codet5_small": ModelSpec(
        key="codet5_small",
        display_name="CodeT5 Small",
        model_id="Salesforce/codet5-small",
        architecture="encoder-decoder",
        parameters_m=60,
        source_url="https://huggingface.co/Salesforce/codet5-small",
    ),
    "codet5p_220m": ModelSpec(
        key="codet5p_220m",
        display_name="CodeT5+ 220M",
        model_id="Salesforce/codet5p-220m",
        architecture="encoder-decoder",
        parameters_m=220,
        source_url="https://huggingface.co/Salesforce/codet5p-220m",
    ),
    "codegpt_py": ModelSpec(
        key="codegpt_py",
        display_name="CodeGPT Small Python",
        model_id="microsoft/CodeGPT-small-py",
        architecture="decoder-only",
        parameters_m=124,
        source_url="https://huggingface.co/microsoft/CodeGPT-small-py",
    ),
    "codegpt_java": ModelSpec(
        key="codegpt_java",
        display_name="CodeGPT Small Java",
        model_id="microsoft/CodeGPT-small-java",
        architecture="decoder-only",
        parameters_m=124,
        source_url="https://huggingface.co/microsoft/CodeGPT-small-java",
    ),
    "codeberta_small": ModelSpec(
        key="codeberta_small",
        display_name="CodeBERTa Small",
        model_id="huggingface/CodeBERTa-small-v1",
        architecture="encoder-only",
        parameters_m=84,
        source_url="https://huggingface.co/huggingface/CodeBERTa-small-v1",
    ),
    "cotext_1_cc": ModelSpec(
        key="cotext_1_cc",
        display_name="CoTexT 1-CC",
        model_id="razent/cotext-1-cc",
        architecture="encoder-decoder",
        parameters_m=220,
        source_url="https://huggingface.co/razent/cotext-1-cc",
    ),
    "cotext_2_cc": ModelSpec(
        key="cotext_2_cc",
        display_name="CoTexT 2-CC",
        model_id="razent/cotext-2-cc",
        architecture="encoder-decoder",
        parameters_m=220,
        source_url="https://huggingface.co/razent/cotext-2-cc",
    ),
    "syncobert": ModelSpec(
        key="syncobert",
        display_name="SynCoBERT",
        model_id=None,
        architecture="encoder-only",
        parameters_m=125,
        runnable=False,
        notes="Paper-relevant baseline; provide a local checkpoint with --model_path.",
        source_url="https://arxiv.org/abs/2108.04556",
    ),
    "code_mvp": ModelSpec(
        key="code_mvp",
        display_name="Code-MVP",
        model_id=None,
        architecture="encoder-only",
        parameters_m=125,
        runnable=False,
        notes="Paper-relevant baseline; provide a local checkpoint with --model_path.",
        source_url="https://arxiv.org/abs/2205.02029",
    ),
}


BENCHMARK_REGISTRY: dict[str, BenchmarkSpec] = {
    "bcb": BenchmarkSpec(
        key="bcb",
        display_name="BigCloneBench",
        languages=("Java",),
        benchmark_type="monolingual clone detection",
        expected_layout="pair_jsonl",
        source_url="https://github.com/clonebench/BigCloneBench",
    ),
    "poj104": BenchmarkSpec(
        key="poj104",
        display_name="POJ-104",
        languages=("C", "C++"),
        benchmark_type="program similarity / retrieval",
        expected_layout="pair_jsonl",
        source_url="https://github.com/microsoft/CodeXGLUE",
    ),
    "gcj": BenchmarkSpec(
        key="gcj",
        display_name="Google Code Jam",
        languages=("mixed",),
        benchmark_type="problem-solution clone detection",
        expected_layout="pair_jsonl",
    ),
    "karnalim": BenchmarkSpec(
        key="karnalim",
        display_name="Karnalim",
        languages=("Java",),
        benchmark_type="educational clone detection",
        expected_layout="pair_jsonl",
    ),
    "poolc": BenchmarkSpec(
        key="poolc",
        display_name="PoolC",
        languages=("C",),
        benchmark_type="educational clone detection",
        expected_layout="pair_jsonl",
    ),
    "codexglue_bcb": BenchmarkSpec(
        key="codexglue_bcb",
        display_name="CodeXGLUE BigCloneBench",
        languages=("Java",),
        benchmark_type="official CodeXGLUE clone detection",
        expected_layout="pair_jsonl",
        source_url="https://github.com/microsoft/CodeXGLUE",
    ),
    "codexglue_poj104": BenchmarkSpec(
        key="codexglue_poj104",
        display_name="CodeXGLUE POJ-104",
        languages=("C", "C++"),
        benchmark_type="official CodeXGLUE clone retrieval",
        expected_layout="pair_jsonl",
        source_url="https://github.com/microsoft/CodeXGLUE",
    ),
    "codenet": BenchmarkSpec(
        key="codenet",
        display_name="Project CodeNet",
        languages=("55 languages",),
        benchmark_type="large-scale code similarity",
        expected_layout="problem_directories or pair_jsonl",
        source_url="https://github.com/IBM/Project_CodeNet",
        notes="Use scripts/prepare_pair_dataset.py for problem-directory subsets.",
    ),
    "semanticclonebench": BenchmarkSpec(
        key="semanticclonebench",
        display_name="SemanticCloneBench",
        languages=("Java",),
        benchmark_type="semantic clone detection",
        expected_layout="pair_jsonl",
        source_url=(
            "https://clones.usask.ca/pubfiles/articles/"
            "Omari_SemanticClonesBenchIWSC2020.pdf"
        ),
    ),
    "gptclonebench": BenchmarkSpec(
        key="gptclonebench",
        display_name="GPTCloneBench",
        languages=("Java", "C", "C#", "Python"),
        benchmark_type="semantic and cross-language clone detection",
        expected_layout="pair_jsonl",
        source_url="https://arxiv.org/abs/2308.13963",
        notes="Generated benchmark; report separately from human-curated sets.",
    ),
    "clcdsa": BenchmarkSpec(
        key="clcdsa",
        display_name="CLCDSA",
        languages=("Java", "C#", "C++", "Python"),
        benchmark_type="cross-language clone detection",
        expected_layout="problem_directories or pair_jsonl",
        source_url="https://clones.usask.ca/pubfiles/articles/Nafi_CLCDSAASE2019.pdf",
        notes="Use scripts/prepare_pair_dataset.py for problem-directory subsets.",
    ),
    "robustness": BenchmarkSpec(
        key="robustness",
        display_name="Transformation Robustness Suite",
        languages=("dataset-derived",),
        benchmark_type="mutation and obfuscation stress test",
        expected_layout="derived pair_jsonl",
        runnable=False,
        notes="Protocol entry for transformed splits derived from base datasets.",
    ),
}


MODEL_ALIASES = {
    "t5": "codet5",
    "codet5_base": "codet5",
    "salesforce_t5": "codet5",
    "codebert-small": "codeberta_small",
    "codeberta": "codeberta_small",
    "codet5+": "codet5p_220m",
    "codet5p": "codet5p_220m",
    "codegpt": "codegpt_py",
}

BENCHMARK_ALIASES = {
    "bigclonebench": "bcb",
    "codejam": "gcj",
    "poj": "poj104",
    "poj-104": "poj104",
    "semantic_clone_bench": "semanticclonebench",
    "gpt_clone_bench": "gptclonebench",
}


def get_model_spec(name: str) -> ModelSpec:
    key = MODEL_ALIASES.get(name.lower(), name.lower())
    try:
        return MODEL_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model {name!r}. Available models: {available}") from exc


def get_benchmark_spec(name: str) -> BenchmarkSpec:
    key = BENCHMARK_ALIASES.get(name.lower(), name.lower())
    try:
        return BENCHMARK_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(BENCHMARK_REGISTRY))
        raise KeyError(f"Unknown benchmark {name!r}. Available benchmarks: {available}") from exc


def list_model_specs(runnable_only: bool = False) -> list[ModelSpec]:
    specs = sorted(MODEL_REGISTRY.values(), key=lambda spec: spec.key)
    if runnable_only:
        return [spec for spec in specs if spec.runnable]
    return specs


def list_benchmark_specs(runnable_only: bool = False) -> list[BenchmarkSpec]:
    specs = sorted(BENCHMARK_REGISTRY.values(), key=lambda spec: spec.key)
    if runnable_only:
        return [spec for spec in specs if spec.runnable]
    return specs
