"""Tests for the proposer loop's per-wake logic.

The proposer's zero-Claude paths (pause-while-PR-open, cap-hit, dry-run)
must each return 0 without firing Claude. Each is exercised explicitly
with a mocked runner that fails the test if invoked.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from vgc_ai import proposer


def _write_battle_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "timestamp",
        "track",
        "strategy_a",
        "strategy_b",
        "wins_a",
        "wins_b",
        "ties",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r.get(k, "") for k in header])


def test_summarize_track_empty_csv(tmp_path: Path) -> None:
    summary = proposer.summarize_track(
        tmp_path / "battle.csv",
        candidates=["a", "b"],
        default="heuristic_det",
    )
    assert summary["default"] == "heuristic_det"
    assert summary["rows_seen"] == 0
    assert summary["candidates"] == []


def test_summarize_track_pools_per_candidate(tmp_path: Path) -> None:
    rows = [
        {"strategy_a": "candX", "strategy_b": "heuristic_det", "wins_a": "30", "wins_b": "70"},
        {"strategy_a": "candX", "strategy_b": "heuristic_det", "wins_a": "35", "wins_b": "65"},
        {"strategy_a": "candY", "strategy_b": "heuristic_det", "wins_a": "60", "wins_b": "40"},
    ]
    _write_battle_csv(tmp_path / "battle.csv", rows)
    s = proposer.summarize_track(
        tmp_path / "battle.csv",
        candidates=["candX", "candY", "heuristic_det"],
        default="heuristic_det",
    )
    # Self-pair (default vs default) excluded; candX rows pooled (65/200);
    # candY single row (60/100).
    names = {c["name"]: c for c in s["candidates"]}
    assert set(names) == {"candX", "candY"}
    assert names["candX"]["pooled_n"] == 200
    assert names["candX"]["pooled_wins"] == 65
    assert names["candY"]["pooled_n"] == 100
    assert names["candY"]["pooled_wins"] == 60


def test_pick_most_contested_track_prefers_smaller_default_cushion() -> None:
    # Battle: default is way ahead (default cushion = 0.4)
    # Championship: default barely ahead (cushion = 0.05)
    summaries = {
        "battle": {
            "default": "heur",
            "candidates": [{"name": "x", "vs_default_margin": 0.4}],
            "rows_seen": 100,
        },
        "championship": {
            "default": "minimax",
            "candidates": [{"name": "y", "vs_default_margin": 0.05}],
            "rows_seen": 100,
        },
    }
    assert proposer.pick_most_contested_track(summaries) == "championship"


def test_pick_most_contested_track_returns_untested_track_first() -> None:
    # A track with no candidates measured = most "contested" by definition;
    # we don't know anything about it yet.
    summaries = {
        "battle": {
            "default": "heur",
            "candidates": [{"name": "x", "vs_default_margin": 0.05}],
            "rows_seen": 100,
        },
        "championship": {"default": "minimax", "candidates": [], "rows_seen": 0},
    }
    assert proposer.pick_most_contested_track(summaries) == "championship"


def test_pick_most_contested_track_handles_all_empty() -> None:
    summaries = {
        "battle": {"default": "heur", "candidates": [], "rows_seen": 0},
        "championship": {"default": "minimax", "candidates": [], "rows_seen": 0},
    }
    # First empty track wins (any deterministic choice is fine).
    track = proposer.pick_most_contested_track(summaries)
    assert track in ("battle", "championship")


def test_previous_attempts_empty_when_file_missing(tmp_path: Path) -> None:
    assert proposer.previous_attempts(tmp_path / "nope.csv") == []


def test_previous_attempts_reads_rows(tmp_path: Path) -> None:
    path = tmp_path / "attempts.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "track", "compound_name", "exit_code"])
        w.writerow(["2026-05-20T08:00:00+00:00", "battle", "foo", "0"])
        w.writerow(["2026-05-20T09:00:00+00:00", "championship", "bar", "1"])
    rows = proposer.previous_attempts(path)
    assert len(rows) == 2
    assert rows[0]["compound_name"] == "foo"
    assert rows[1]["exit_code"] == "1"


def test_build_prompt_contains_track_and_summaries() -> None:
    summaries = {
        "battle": {"default": "heuristic_det", "candidates": [], "rows_seen": 0},
        "championship": {"default": "minimax", "candidates": [], "rows_seen": 0},
    }
    prompt = proposer.build_prompt("battle", summaries, attempts=[])
    assert "Current track to improve: **battle**" in prompt
    assert "heuristic_det" in prompt
    assert "NEW COMPOUND" in prompt
    assert "track=battle" in prompt
    # Existing files must not be reverentially out-of-scope here; the
    # prompt explicitly tells Claude what IT is allowed to touch.
    assert "src/vgc_ai/policies/" in prompt
    assert "ops/" in prompt  # mentioned only as a forbidden directory


def test_build_prompt_includes_attempt_log() -> None:
    summaries = {"battle": {"default": "heur", "candidates": [], "rows_seen": 0}}
    attempts = [
        {
            "timestamp": "2026-05-20T08:00:00+00:00",
            "track": "battle",
            "compound_name": "foo_bar",
            "exit_code": "0",
        }
    ]
    prompt = proposer.build_prompt("battle", summaries, attempts)
    assert "foo_bar" in prompt
    assert "do NOT re-propose" in prompt


def test_invoke_claude_logs_to_budget_and_attempts(tmp_path: Path) -> None:
    budget = tmp_path / "budget.csv"
    attempts = tmp_path / "attempts.csv"

    def fake_claude(cmd: list[str], prompt: str) -> tuple[int, str, str]:
        assert cmd[0] == "claude"
        return (0, json.dumps({"usage": {"input_tokens": 12345, "output_tokens": 1500}}), "")

    rc = proposer.invoke_claude(
        track="battle",
        prompt="test prompt",
        budget_path=budget,
        attempts_path=attempts,
        claude_runner=fake_claude,
    )
    assert rc == 0
    budget_rows = list(csv.DictReader(budget.open(encoding="utf-8")))
    assert len(budget_rows) == 1
    assert budget_rows[0]["track"] == "battle"
    assert budget_rows[0]["candidate"] == "proposer"
    assert budget_rows[0]["input_tokens"] == "12345"
    # Attempt log should also have a row.
    attempt_rows = list(csv.DictReader(attempts.open(encoding="utf-8")))
    assert len(attempt_rows) == 1
    assert attempt_rows[0]["track"] == "battle"
    assert attempt_rows[0]["exit_code"] == "0"


def test_main_pauses_when_pr_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(proposer, "list_open_reviewer_prs", lambda: [99])
    monkeypatch.setattr(
        proposer,
        "invoke_claude",
        lambda *a, **k: pytest.fail("must not invoke claude while a PR is open"),
    )
    rc = proposer.main(
        [
            "--csv-dir",
            str(tmp_path),
            "--budget",
            str(tmp_path / "b.csv"),
            "--attempts",
            str(tmp_path / "a.csv"),
        ]
    )
    assert rc == 0
    assert "PR(s) open" in capsys.readouterr().err


def test_main_pauses_when_cap_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(proposer, "list_open_reviewer_prs", lambda: [])
    monkeypatch.setattr(proposer, "daily_claude_calls", lambda *_a, **_k: 12)
    monkeypatch.setattr(
        proposer,
        "invoke_claude",
        lambda *a, **k: pytest.fail("must not invoke claude when cap hit"),
    )
    rc = proposer.main(
        [
            "--csv-dir",
            str(tmp_path),
            "--budget",
            str(tmp_path / "b.csv"),
            "--attempts",
            str(tmp_path / "a.csv"),
            "--cap",
            "12",
        ]
    )
    assert rc == 0
    assert "cap hit" in capsys.readouterr().err


def test_main_dry_run_prints_prompt_and_does_not_invoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(proposer, "list_open_reviewer_prs", lambda: [])
    monkeypatch.setattr(proposer, "daily_claude_calls", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        proposer,
        "invoke_claude",
        lambda *a, **k: pytest.fail("must not invoke claude in dry-run"),
    )

    csv_dir = tmp_path / "strategies"
    csv_dir.mkdir()

    rc = proposer.main(
        [
            "--csv-dir",
            str(csv_dir),
            "--budget",
            str(tmp_path / "b.csv"),
            "--attempts",
            str(tmp_path / "a.csv"),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    # The dry-run path prints the prompt to stdout.
    assert "NEW COMPOUND" in out


# ---- has_new_signal -----------------------------------------------------


def _write_attempts(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "track", "compound_name", "exit_code"])
        for r in rows:
            w.writerow([r[k] for k in ["timestamp", "track", "compound_name", "exit_code"]])


def test_has_new_signal_fires_when_no_prior_attempts(tmp_path: Path) -> None:
    ok, reason = proposer.has_new_signal(tmp_path / "nope.csv")
    assert ok is True
    assert "no prior attempts" in reason


def test_has_new_signal_fires_on_previous_failure(tmp_path: Path) -> None:
    """Even if the last attempt was 1s ago, if it FAILED we retry —
    Claude has new failure context to work with."""
    path = tmp_path / "attempts.csv"
    one_second_ago = (datetime.now(UTC) - timedelta(seconds=1)).isoformat(timespec="seconds")
    _write_attempts(
        path,
        [
            {
                "timestamp": one_second_ago,
                "track": "battle",
                "compound_name": "foo",
                "exit_code": "1",
            }
        ],
    )
    ok, reason = proposer.has_new_signal(path)
    assert ok is True
    assert "exited 1" in reason


def test_has_new_signal_skips_when_too_recent_success(tmp_path: Path) -> None:
    """A successful attempt 60s ago must NOT trigger another fire (less
    than the 30-min default interval). Bench loop hasn't measured the
    new compound yet."""

    path = tmp_path / "attempts.csv"
    one_minute_ago = (datetime.now(UTC) - timedelta(seconds=60)).isoformat(timespec="seconds")
    _write_attempts(
        path,
        [
            {
                "timestamp": one_minute_ago,
                "track": "battle",
                "compound_name": "foo",
                "exit_code": "0",
            }
        ],
    )
    ok, reason = proposer.has_new_signal(path)
    assert ok is False
    assert "only" in reason


def test_has_new_signal_fires_when_min_interval_elapsed(tmp_path: Path) -> None:

    path = tmp_path / "attempts.csv"
    long_ago = (datetime.now(UTC) - timedelta(hours=2)).isoformat(timespec="seconds")
    _write_attempts(
        path,
        [
            {
                "timestamp": long_ago,
                "track": "battle",
                "compound_name": "foo",
                "exit_code": "0",
            }
        ],
    )
    ok, reason = proposer.has_new_signal(path, min_interval_sec=1800)
    assert ok is True
    assert "threshold" in reason


def test_has_new_signal_fires_on_unparseable_timestamp(tmp_path: Path) -> None:
    path = tmp_path / "attempts.csv"
    _write_attempts(
        path,
        [
            {
                "timestamp": "definitely-not-iso",
                "track": "battle",
                "compound_name": "foo",
                "exit_code": "0",
            }
        ],
    )
    ok, reason = proposer.has_new_signal(path)
    assert ok is True
    assert "unparseable" in reason


def test_main_skips_when_no_signal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: a recent successful attempt blocks the next fire."""

    monkeypatch.setattr(proposer, "list_open_reviewer_prs", lambda: [])
    monkeypatch.setattr(proposer, "daily_claude_calls", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        proposer,
        "invoke_claude",
        lambda *a, **k: pytest.fail("must not invoke claude when signal gate skips"),
    )

    attempts = tmp_path / "attempts.csv"
    recent = (datetime.now(UTC) - timedelta(seconds=120)).isoformat(timespec="seconds")
    _write_attempts(
        attempts,
        [{"timestamp": recent, "track": "battle", "compound_name": "foo", "exit_code": "0"}],
    )

    rc = proposer.main(
        [
            "--csv-dir",
            str(tmp_path),
            "--budget",
            str(tmp_path / "b.csv"),
            "--attempts",
            str(attempts),
            "--min-interval",
            "1800",
        ]
    )
    assert rc == 0
    assert "skipping" in capsys.readouterr().err


# ---- build_prompt last-attempt status -----------------------------------


def test_build_prompt_announces_last_failure() -> None:
    attempts = [
        {
            "timestamp": "2026-05-20T08:00:00+00:00",
            "track": "battle",
            "compound_name": "broken_thing",
            "exit_code": "1",
        }
    ]
    prompt = proposer.build_prompt(
        "battle",
        {"battle": {"default": "heur", "candidates": [], "rows_seen": 0}},
        attempts,
    )
    assert "Last attempt FAILED" in prompt
    assert "broken_thing" in prompt


def test_build_prompt_announces_last_success_and_warns_against_duplicates() -> None:
    attempts = [
        {
            "timestamp": "2026-05-20T08:00:00+00:00",
            "track": "battle",
            "compound_name": "great_thing",
            "exit_code": "0",
        }
    ]
    prompt = proposer.build_prompt(
        "battle",
        {"battle": {"default": "heur", "candidates": [], "rows_seen": 0}},
        attempts,
    )
    assert "Last attempt SUCCEEDED" in prompt
    assert "great_thing" in prompt
    assert "different angle" in prompt
