"""Unit tests for data loading helpers."""

from pathlib import Path

from small_code_models.data import load_code_snippets


def test_load_code_snippets_from_jsonl(tmp_path: Path) -> None:
    jsonl = tmp_path / "data.jsonl"
    jsonl.write_text('{"idx":"1","func":"print(1)"}\n{"idx":"2","func":"print(2)"}\n')

    result = load_code_snippets(jsonl)

    assert result == {"1": "print(1)", "2": "print(2)"}
