"""Unit tests for the runner module that don't need a Showdown server."""

from __future__ import annotations

import json
from pathlib import Path

from vgc_ai.eval.runner import make_run_dir, write_jsonl


def test_make_run_dir_creates_under_root(tmp_path: Path) -> None:
    d1 = make_run_dir(tmp_path)
    assert d1.exists() and d1.is_dir()
    assert d1.parent == tmp_path


def test_write_jsonl_roundtrip(tmp_path: Path) -> None:
    log = tmp_path / "episodes.jsonl"
    records: list[dict[str, object]] = [
        {"battle_tag": "a", "won": True, "turn": 12},
        {"battle_tag": "b", "won": False, "turn": 8},
    ]
    write_jsonl(log, records)
    loaded = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert loaded == records


def test_write_jsonl_appends(tmp_path: Path) -> None:
    log = tmp_path / "episodes.jsonl"
    write_jsonl(log, [{"a": 1}])
    write_jsonl(log, [{"b": 2}])
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
