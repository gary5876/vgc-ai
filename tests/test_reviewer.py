"""Tests for the reviewer loop's per-wake logic.

Token-budget rule is load-bearing — the dry-run path, the open-PR check,
and the daily-cap check must all return 0 *without* invoking Claude. Each
of those is exercised explicitly here. The Claude-firing path is tested
with a mocked runner so no tokens are spent.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from vgc_ai import reviewer


def _write_battle_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a minimal battle.csv with the columns the reviewer reads."""
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


def test_wilson_ci_reproduces_certified_bounds() -> None:
    # PR #9: heuristic_det 1081/2000, ci=[0.5186, 0.5622].
    lo, hi = reviewer.wilson_ci_95(1081, 2000)
    assert round(lo, 4) == 0.5186
    assert round(hi, 4) == 0.5622


def test_wilson_ci_zero_n() -> None:
    assert reviewer.wilson_ci_95(0, 0) == (0.0, 0.0)


def test_list_open_reviewer_prs_filters_by_bench_gate_marker() -> None:
    fake_response = json.dumps(
        [
            {"number": 25, "body": "Plain PR body without the marker."},
            {"number": 26, "body": "BENCH GATE\ntrack=battle\ncandidate=foo"},
            {"number": 27, "body": None},
        ]
    )

    def fake_gh(cmd: list[str]) -> str:
        assert cmd[0] == "gh"
        return fake_response

    assert reviewer.list_open_reviewer_prs(gh_runner=fake_gh) == [26]


def test_list_open_reviewer_prs_returns_sentinel_when_gh_missing() -> None:
    def fake_gh(cmd: list[str]) -> str:
        raise FileNotFoundError("gh")

    # Conservative: pretend something is open so the reviewer pauses.
    assert reviewer.list_open_reviewer_prs(gh_runner=fake_gh) == [0]


def test_list_open_reviewer_prs_returns_sentinel_on_bad_json() -> None:
    def fake_gh(cmd: list[str]) -> str:
        return "not json"

    assert reviewer.list_open_reviewer_prs(gh_runner=fake_gh) == [0]


def test_list_open_reviewer_prs_includes_new_compound_marker() -> None:
    """Both loop PR classes must pause the proposer — failing to detect
    NEW COMPOUND PRs was the root cause of today's PR-33/35 conflict
    chaos (proposer fired hourly while its own prior PRs were still in
    flight; multiple PRs touched registry.py concurrently)."""
    fake_response = json.dumps(
        [
            {"number": 100, "body": "no marker"},
            {"number": 101, "body": "BENCH GATE\ntrack=battle"},
            {"number": 102, "body": "## Summary\n\nNEW COMPOUND\ntrack=championship"},
        ]
    )

    def fake_gh(cmd: list[str]) -> str:
        return fake_response

    assert sorted(reviewer.list_open_reviewer_prs(gh_runner=fake_gh)) == [101, 102]


def test_list_open_reviewer_prs_strict_against_prose_mentions() -> None:
    """A PR description that *mentions* the marker in prose must not
    pause the loop (matches the auto-handler's strict marker detection
    from PR #32)."""
    fake_response = json.dumps(
        [
            {
                "number": 200,
                "body": (
                    "## Summary\n\nThis PR widens the handler to accept "
                    "`BENCH GATE` and `NEW COMPOUND` markers.\n"
                ),
            }
        ]
    )

    def fake_gh(cmd: list[str]) -> str:
        return fake_response

    assert reviewer.list_open_reviewer_prs(gh_runner=fake_gh) == []


def test_daily_claude_calls_counts_today_only(tmp_path: Path) -> None:
    budget = tmp_path / "claude_budget.csv"
    with budget.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "timestamp",
                "track",
                "candidate",
                "input_tokens",
                "output_tokens",
                "elapsed_sec",
                "exit_code",
            ]
        )
        w.writerow(["2026-05-20T08:00:00+00:00", "battle", "x", "100", "50", "1.0", "0"])
        w.writerow(["2026-05-20T10:00:00+00:00", "battle", "y", "200", "60", "2.0", "0"])
        w.writerow(["2026-05-19T22:00:00+00:00", "battle", "z", "300", "70", "3.0", "0"])

    assert reviewer.daily_claude_calls(budget, today="2026-05-20") == 2
    assert reviewer.daily_claude_calls(budget, today="2026-05-19") == 1
    assert reviewer.daily_claude_calls(budget, today="2026-01-01") == 0


def test_daily_claude_calls_zero_when_file_missing(tmp_path: Path) -> None:
    assert reviewer.daily_claude_calls(tmp_path / "nope.csv") == 0


def test_load_recent_rows_returns_tail(tmp_path: Path) -> None:
    path = tmp_path / "battle.csv"
    rows = [{"strategy_a": "a", "strategy_b": "b", "wins_a": "1", "wins_b": "0", "ties": "0"}] * 60
    _write_battle_csv(path, rows)
    last = reviewer.load_recent_rows(path, n=5)
    assert len(last) == 5


def test_evaluate_gate_pools_wins_correctly(tmp_path: Path) -> None:
    # 10 rows of cand-vs-default at n=10 each, 7-3 each → pooled 70/100 → 0.7
    path = tmp_path / "battle.csv"
    rows = [
        {"strategy_a": "candX", "strategy_b": "heuristic_det", "wins_a": "7", "wins_b": "3"}
    ] * 10
    _write_battle_csv(path, rows)
    decisions = reviewer.evaluate_gate(
        path, candidates=["candX"], default="heuristic_det", track="battle"
    )
    assert len(decisions) == 1
    d = decisions[0]
    assert d["candidate"] == "candX"
    assert d["pooled_n"] == 100
    assert d["pooled_wins"] == 70
    assert d["win_rate"] == 0.7
    # 70/100 → Wilson lower ~0.6042; well above 0.5.
    assert d["ci95_low"] > 0.5
    assert d["gate_fires"] is True


def test_evaluate_gate_does_not_fire_below_threshold(tmp_path: Path) -> None:
    # Marginal candidate: 52 wins / 100 → ci_low ~0.42, gate must NOT fire.
    path = tmp_path / "battle.csv"
    rows = [{"strategy_a": "candX", "strategy_b": "heuristic_det", "wins_a": "52", "wins_b": "48"}]
    _write_battle_csv(path, rows)
    decisions = reviewer.evaluate_gate(
        path, candidates=["candX"], default="heuristic_det", track="battle"
    )
    assert decisions[0]["gate_fires"] is False


def test_evaluate_gate_skips_self_pair(tmp_path: Path) -> None:
    path = tmp_path / "battle.csv"
    _write_battle_csv(
        path,
        [
            {
                "strategy_a": "heuristic_det",
                "strategy_b": "heuristic_det",
                "wins_a": "5",
                "wins_b": "5",
            }
        ],
    )
    decisions = reviewer.evaluate_gate(
        path,
        candidates=["heuristic_det"],
        default="heuristic_det",
        track="battle",
    )
    assert decisions == []


def test_evaluate_gate_no_rows(tmp_path: Path) -> None:
    assert (
        reviewer.evaluate_gate(
            tmp_path / "missing.csv", candidates=["x"], default="y", track="battle"
        )
        == []
    )


def test_build_prompt_contains_key_decision_fields() -> None:
    decision = {
        "track": "battle",
        "candidate": "candX",
        "default": "heuristic_det",
        "pooled_n": 100,
        "pooled_wins": 70,
        "win_rate": 0.7,
        "ci95_low": 0.6042,
        "ci95_high": 0.7836,
    }
    prompt = reviewer.build_prompt(decision)
    assert "candX" in prompt
    assert "heuristic_det" in prompt
    assert "BENCH GATE" in prompt
    assert "BATTLE_DEFAULT" in prompt
    # Auto-handler-parseable block contents:
    assert "track=battle" in prompt
    assert "candidate=candX" in prompt
    assert "ci95_low=0.6042" in prompt


def test_invoke_claude_logs_token_usage(tmp_path: Path) -> None:
    budget = tmp_path / "claude_budget.csv"

    def fake_claude(cmd: list[str], prompt: str) -> tuple[int, str, str]:
        assert cmd[0] == "claude"
        return (0, json.dumps({"usage": {"input_tokens": 4321, "output_tokens": 765}}), "")

    decision = {
        "track": "battle",
        "candidate": "candX",
        "default": "heuristic_det",
        "pooled_n": 100,
        "pooled_wins": 70,
        "win_rate": 0.7,
        "ci95_low": 0.6,
        "ci95_high": 0.78,
    }
    rc = reviewer.invoke_claude(decision, budget, claude_runner=fake_claude)
    assert rc == 0
    rows = list(csv.DictReader(budget.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["candidate"] == "candX"
    assert rows[0]["input_tokens"] == "4321"
    assert rows[0]["output_tokens"] == "765"
    assert rows[0]["exit_code"] == "0"


def test_invoke_claude_handles_missing_usage_field(tmp_path: Path) -> None:
    budget = tmp_path / "claude_budget.csv"

    def fake_claude(cmd: list[str], prompt: str) -> tuple[int, str, str]:
        return (0, "not even json", "")

    decision = {
        "track": "battle",
        "candidate": "candX",
        "default": "heuristic_det",
        "pooled_n": 10,
        "pooled_wins": 7,
        "win_rate": 0.7,
        "ci95_low": 0.6,
        "ci95_high": 0.8,
    }
    rc = reviewer.invoke_claude(decision, budget, claude_runner=fake_claude)
    assert rc == 0
    rows = list(csv.DictReader(budget.open(encoding="utf-8")))
    # The call still gets logged so it counts against the daily cap.
    assert len(rows) == 1
    assert rows[0]["input_tokens"] == "0"


def test_main_pauses_when_pr_is_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(reviewer, "list_open_reviewer_prs", lambda: [42])
    # If Claude is invoked, this fails the test.
    monkeypatch.setattr(
        reviewer,
        "invoke_claude",
        lambda *a, **k: pytest.fail("must not invoke claude while a PR is open"),
    )
    rc = reviewer.main(["--csv-dir", str(tmp_path), "--budget", str(tmp_path / "b.csv")])
    assert rc == 0
    assert "PR(s) open" in capsys.readouterr().err


def test_main_pauses_when_cap_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(reviewer, "list_open_reviewer_prs", lambda: [])
    monkeypatch.setattr(reviewer, "daily_claude_calls", lambda *_a, **_k: 12)
    monkeypatch.setattr(
        reviewer,
        "invoke_claude",
        lambda *a, **k: pytest.fail("must not invoke claude when cap hit"),
    )
    rc = reviewer.main(
        ["--csv-dir", str(tmp_path), "--budget", str(tmp_path / "b.csv"), "--cap", "12"]
    )
    assert rc == 0
    assert "cap hit" in capsys.readouterr().err


def test_main_dry_run_does_not_invoke_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(reviewer, "list_open_reviewer_prs", lambda: [])
    monkeypatch.setattr(reviewer, "daily_claude_calls", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        reviewer,
        "invoke_claude",
        lambda *a, **k: pytest.fail("must not invoke claude in dry-run"),
    )

    csv_dir = tmp_path / "strategies"
    rows = [
        {"strategy_a": "greedy", "strategy_b": "heuristic_det", "wins_a": "70", "wins_b": "30"}
    ] * 10
    _write_battle_csv(csv_dir / "battle.csv", rows)

    rc = reviewer.main(
        [
            "--csv-dir",
            str(csv_dir),
            "--budget",
            str(tmp_path / "b.csv"),
            "--dry-run",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    decision = json.loads(out)
    assert decision["candidate"] == "greedy"
    assert decision["gate_fires"] is True


def test_main_no_gate_fires_skips_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(reviewer, "list_open_reviewer_prs", lambda: [])
    monkeypatch.setattr(reviewer, "daily_claude_calls", lambda *_a, **_k: 0)
    monkeypatch.setattr(
        reviewer,
        "invoke_claude",
        lambda *a, **k: pytest.fail("must not invoke claude with no signal"),
    )

    csv_dir = tmp_path / "strategies"
    # 52-48 — positive point estimate, but gate doesn't fire.
    _write_battle_csv(
        csv_dir / "battle.csv",
        [{"strategy_a": "greedy", "strategy_b": "heuristic_det", "wins_a": "52", "wins_b": "48"}],
    )

    rc = reviewer.main(["--csv-dir", str(csv_dir), "--budget", str(tmp_path / "b.csv")])
    assert rc == 0
    assert "no gate fires" in capsys.readouterr().err
