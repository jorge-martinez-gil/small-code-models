"""Unit tests for experiment registries."""

import pytest

from small_code_models.registry import (
    get_benchmark_spec,
    get_model_spec,
    list_benchmark_specs,
    list_model_specs,
)


def test_new_model_specs_are_registered() -> None:
    assert get_model_spec("codet5_small").model_id == "Salesforce/codet5-small"
    assert get_model_spec("codegpt").key == "codegpt_py"
    assert get_model_spec("syncobert").runnable is False


def test_new_benchmark_specs_are_registered() -> None:
    assert get_benchmark_spec("codenet").expected_layout == "problem_directories or pair_jsonl"
    assert get_benchmark_spec("gpt_clone_bench").key == "gptclonebench"
    assert get_benchmark_spec("robustness").runnable is False


def test_registry_listing_filters_runnable_specs() -> None:
    runnable_models = list_model_specs(runnable_only=True)
    runnable_benchmarks = list_benchmark_specs(runnable_only=True)

    assert all(spec.runnable for spec in runnable_models)
    assert all(spec.runnable for spec in runnable_benchmarks)


def test_unknown_registry_keys_raise_clear_errors() -> None:
    with pytest.raises(KeyError, match="Available models"):
        get_model_spec("not-a-model")
    with pytest.raises(KeyError, match="Available benchmarks"):
        get_benchmark_spec("not-a-benchmark")
