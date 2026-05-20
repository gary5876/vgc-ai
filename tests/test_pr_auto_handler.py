"""Tests for the PR auto-handler.

The handler must never silently merge an out-of-scope or signal-degraded
PR. Each gate (scope, CI, bench re-check) is tested in isolation, and the
``evaluate_pr`` orchestrator is tested across all four ``action`` outcomes
with mocked runners so no real subprocess / gh calls happen during tests.
"""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from vgc_ai import pr_auto_handler as handler

# ---- parse_bench_gate ----------------------------------------------------


def test_parse_bench_gate_extracts_block() -> None:
    body = """## Summary

Bench evidence:

BENCH GATE
track=battle
candidate=foo
default=bar
pooled_n=100
pooled_wins=70
ci95_low=0.6
ci95_high=0.78

Some more text after.
"""
    parsed = handler.parse_bench_gate(body)
    assert parsed == {
        "track": "battle",
        "candidate": "foo",
        "default": "bar",
        "pooled_n": "100",
        "pooled_wins": "70",
        "ci95_low": "0.6",
        "ci95_high": "0.78",
    }


def test_parse_bench_gate_tolerates_fenced_block() -> None:
    body = """Header

```
BENCH GATE
track=championship
candidate=cand
default=def
pooled_n=50
pooled_wins=35
ci95_low=0.55
ci95_high=0.81
```

Trailing text.
"""
    parsed = handler.parse_bench_gate(body)
    assert parsed is not None
    assert parsed["track"] == "championship"
    assert parsed["ci95_low"] == "0.55"


def test_parse_bench_gate_returns_none_when_marker_absent() -> None:
    assert handler.parse_bench_gate("nothing in here") is None


def test_parse_bench_gate_returns_none_when_block_empty() -> None:
    assert handler.parse_bench_gate("Some text\n\nBENCH GATE\n\n\nUnrelated text") is None


# ---- scope_check ---------------------------------------------------------


def test_scope_check_passes_for_allowed_paths_only() -> None:
    ok, reason = handler.scope_check(["src/vgc_ai/strategies/registry.py"])
    assert ok is True
    assert reason == ""


def test_scope_check_passes_for_both_allowed_paths() -> None:
    ok, _ = handler.scope_check(
        [
            "src/vgc_ai/strategies/registry.py",
            "src/vgc_ai/policies/battle.py",
        ]
    )
    assert ok is True


def test_scope_check_fails_on_out_of_scope_path() -> None:
    ok, reason = handler.scope_check(
        ["src/vgc_ai/strategies/registry.py", "src/vgc_ai/competitor.py"]
    )
    assert ok is False
    assert "competitor.py" in reason


def test_scope_check_fails_on_test_modifications() -> None:
    # Even tests are out-of-scope — the reviewer's prompt forbids new tests
    # and the constant-swap doesn't need them.
    ok, _reason = handler.scope_check(["tests/test_strategies.py"])
    assert ok is False


# ---- parse_pr_age_seconds ------------------------------------------------


def test_parse_pr_age_seconds_returns_positive_for_past_timestamps() -> None:
    earlier = (datetime.now(UTC) - timedelta(seconds=120)).isoformat()
    age = handler.parse_pr_age_seconds(earlier)
    # Allow a small tolerance for the time elapsed between the two now() calls.
    assert 119 <= age <= 125


def test_parse_pr_age_seconds_handles_zulu_suffix() -> None:
    earlier_z = (datetime.now(UTC) - timedelta(seconds=60)).strftime("%Y-%m-%dT%H:%M:%SZ")
    age = handler.parse_pr_age_seconds(earlier_z)
    assert 59 <= age <= 65


# ---- bench_gate_still_fires ----------------------------------------------


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


def test_bench_gate_still_fires_passes_with_strong_signal(tmp_path: Path) -> None:
    _write_battle_csv(
        tmp_path / "battle.csv",
        [{"strategy_a": "candX", "strategy_b": "heuristic_det", "wins_a": "70", "wins_b": "30"}]
        * 10,
    )
    ok, reason = handler.bench_gate_still_fires(
        {"track": "battle", "candidate": "candX", "default": "heuristic_det"},
        tmp_path,
    )
    assert ok is True
    assert reason == ""


def test_bench_gate_still_fires_fails_when_signal_degrades(tmp_path: Path) -> None:
    _write_battle_csv(
        tmp_path / "battle.csv",
        [{"strategy_a": "candX", "strategy_b": "heuristic_det", "wins_a": "52", "wins_b": "48"}],
    )
    ok, reason = handler.bench_gate_still_fires(
        {"track": "battle", "candidate": "candX", "default": "heuristic_det"},
        tmp_path,
    )
    assert ok is False
    assert "gate no longer fires" in reason


def test_bench_gate_still_fires_fails_with_no_recent_data(tmp_path: Path) -> None:
    ok, reason = handler.bench_gate_still_fires(
        {"track": "battle", "candidate": "candX", "default": "heuristic_det"},
        tmp_path,  # csv_dir but no file
    )
    assert ok is False
    assert "no recent rows" in reason


def test_bench_gate_still_fires_rejects_unknown_track(tmp_path: Path) -> None:
    ok, reason = handler.bench_gate_still_fires(
        {"track": "unknown", "candidate": "x", "default": "y"},
        tmp_path,
    )
    assert ok is False
    assert "unknown track" in reason


# ---- list_matching_prs ---------------------------------------------------


def test_list_matching_prs_filters_on_bench_gate_marker() -> None:
    response = json.dumps(
        [
            {"number": 1, "body": "no marker", "files": [], "createdAt": "2026-05-20T00:00:00Z"},
            {
                "number": 2,
                "body": "BENCH GATE\ntrack=battle",
                "files": [{"path": "src/vgc_ai/strategies/registry.py"}],
                "createdAt": "2026-05-20T00:01:00Z",
            },
        ]
    )

    def fake_gh(cmd: list[str]) -> str:
        return response

    prs = handler.list_matching_prs(gh_runner=fake_gh)
    assert [p["number"] for p in prs] == [2]


def test_list_matching_prs_returns_empty_on_gh_failure() -> None:
    def fake_gh(cmd: list[str]) -> str:
        raise FileNotFoundError("gh")

    assert handler.list_matching_prs(gh_runner=fake_gh) == []


# ---- run_ci_gate ---------------------------------------------------------


def test_run_ci_gate_passes_when_all_green(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
        calls.append(cmd)
        return (0, "", "")

    ok, reason = handler.run_ci_gate(tmp_path, cmd_runner=fake_runner)
    assert ok is True
    assert reason == ""
    # All four commands run.
    assert len(calls) == 4
    assert calls[0] == ["uv", "run", "ruff", "format", "--check", "src", "tests"]


def test_run_ci_gate_stops_on_first_failure(tmp_path: Path) -> None:
    calls: list[list[str]] = []

    def fake_runner(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
        calls.append(cmd)
        # ruff format passes, ruff check fails — stop after 2 calls.
        if "check" in cmd and "ruff" in cmd:
            return (1, "", "would reformat foo.py")
        return (0, "", "")

    ok, reason = handler.run_ci_gate(tmp_path, cmd_runner=fake_runner)
    assert ok is False
    assert "ruff check" in reason
    assert "would reformat" in reason
    assert len(calls) == 2  # stopped after the failing one


# ---- evaluate_pr ---------------------------------------------------------


def _pr(
    *,
    number: int = 99,
    body: str | None = None,
    files: list[dict[str, str]] | None = None,
    age_seconds: float = 120,
) -> dict[str, Any]:
    if body is None:
        body = (
            "BENCH GATE\n"
            "track=battle\n"
            "candidate=candX\n"
            "default=heuristic_det\n"
            "pooled_n=100\n"
            "pooled_wins=70\n"
            "ci95_low=0.60\n"
            "ci95_high=0.78\n"
        )
    if files is None:
        files = [{"path": "src/vgc_ai/strategies/registry.py"}]
    created_at = (datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat()
    return {
        "number": number,
        "body": body,
        "files": files,
        "createdAt": created_at,
    }


def test_evaluate_pr_closes_when_body_missing_marker(tmp_path: Path) -> None:
    pr = _pr(body="plain body, no marker")
    action, reason = handler.evaluate_pr(pr, tmp_path, tmp_path)
    assert action == "close"
    assert "missing BENCH GATE" in reason


def test_evaluate_pr_waits_when_too_young(tmp_path: Path) -> None:
    pr = _pr(age_seconds=5)
    action, reason = handler.evaluate_pr(pr, tmp_path, tmp_path)
    assert action == "wait"
    assert "only 5s old" in reason or "only 4s old" in reason or "only 6s old" in reason


def test_evaluate_pr_closes_on_out_of_scope_diff(tmp_path: Path) -> None:
    pr = _pr(files=[{"path": "src/vgc_ai/competitor.py"}])
    action, reason = handler.evaluate_pr(pr, tmp_path, tmp_path)
    assert action == "close"
    assert "competitor.py" in reason


def test_evaluate_pr_closes_when_bench_gate_degrades(tmp_path: Path) -> None:
    # Fake CI green; bench csv shows a degraded signal.
    _write_battle_csv(
        tmp_path / "battle.csv",
        [{"strategy_a": "candX", "strategy_b": "heuristic_det", "wins_a": "50", "wins_b": "50"}],
    )

    def green_ci(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
        return (0, "", "")

    pr = _pr()
    action, reason = handler.evaluate_pr(pr, tmp_path, tmp_path, cmd_runner=green_ci)
    assert action == "close"
    assert "bench gate re-check failed" in reason


def test_evaluate_pr_waits_when_ci_fails_inside_sla(tmp_path: Path) -> None:
    def failing_ci(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
        return (1, "", "ruff failed")

    pr = _pr(age_seconds=60)
    action, reason = handler.evaluate_pr(pr, tmp_path, tmp_path, cmd_runner=failing_ci)
    assert action == "wait"
    assert "retry next cycle" in reason


def test_evaluate_pr_closes_when_ci_fails_past_sla(tmp_path: Path) -> None:
    def failing_ci(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
        return (1, "", "pytest exit 1")

    pr = _pr(age_seconds=400)
    action, reason = handler.evaluate_pr(
        pr, tmp_path, tmp_path, max_age_sec=300, cmd_runner=failing_ci
    )
    assert action == "close"
    assert "SLA=300" in reason


def test_evaluate_pr_merges_when_all_gates_pass(tmp_path: Path) -> None:
    _write_battle_csv(
        tmp_path / "battle.csv",
        [{"strategy_a": "candX", "strategy_b": "heuristic_det", "wins_a": "70", "wins_b": "30"}]
        * 10,
    )

    def green_ci(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
        return (0, "", "")

    pr = _pr()
    action, reason = handler.evaluate_pr(pr, tmp_path, tmp_path, cmd_runner=green_ci)
    assert action == "merge"
    assert reason == "all gates passed"


# ---- main ----------------------------------------------------------------


def test_main_no_prs_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(handler, "list_matching_prs", lambda: [])
    rc = handler.main(["--csv-dir", str(tmp_path), "--workdir", str(tmp_path)])
    assert rc == 0
    assert "no matching PRs" in capsys.readouterr().err


def test_main_dry_run_does_not_call_merge_or_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pr = _pr()
    monkeypatch.setattr(handler, "list_matching_prs", lambda: [pr])
    monkeypatch.setattr(
        handler,
        "merge_pr",
        lambda *a, **k: pytest.fail("must not call merge in dry-run"),
    )
    monkeypatch.setattr(
        handler,
        "close_pr",
        lambda *a, **k: pytest.fail("must not call close in dry-run"),
    )
    # Stub evaluate_pr to short-circuit subprocess calls.
    monkeypatch.setattr(handler, "evaluate_pr", lambda *a, **k: ("merge", "ok"))

    rc = handler.main(["--dry-run", "--csv-dir", str(tmp_path), "--workdir", str(tmp_path)])
    assert rc == 0


# ---- NEW COMPOUND PR class ----------------------------------------------


def test_parse_new_compound_extracts_block() -> None:
    body = """## Summary

NEW COMPOUND
track=battle
compound_name=heuristic_det_damage_term
rationale=Adds a base_power-weighted damage term to evaluate()

Some trailing text.
"""
    parsed = handler.parse_new_compound(body)
    assert parsed == {
        "track": "battle",
        "compound_name": "heuristic_det_damage_term",
        "rationale": "Adds a base_power-weighted damage term to evaluate()",
    }


def test_parse_new_compound_returns_none_when_marker_absent() -> None:
    assert handler.parse_new_compound("only BENCH GATE here\ntrack=x") is None


def test_scope_check_prefix_passes_for_allowed_prefixes() -> None:
    ok, reason = handler.scope_check_prefix(
        [
            "src/vgc_ai/policies/new_policy.py",
            "src/vgc_ai/eval/new_term.py",
            "tests/test_new_policy.py",
            "src/vgc_ai/strategies/registry.py",
        ]
    )
    assert ok is True
    assert reason == ""


def test_scope_check_prefix_rejects_ops_and_bench() -> None:
    ok, reason = handler.scope_check_prefix(
        ["src/vgc_ai/policies/new_policy.py", "ops/some_script.sh"]
    )
    assert ok is False
    assert "ops/" in reason


def test_scope_check_prefix_rejects_toplevel() -> None:
    ok, reason = handler.scope_check_prefix(["README.md", "src/vgc_ai/policies/x.py"])
    assert ok is False
    assert "README.md" in reason


def test_list_matching_prs_includes_both_markers() -> None:
    response = json.dumps(
        [
            {"number": 1, "body": "BENCH GATE\ntrack=battle", "files": [], "createdAt": "x"},
            {"number": 2, "body": "NEW COMPOUND\ntrack=battle", "files": [], "createdAt": "x"},
            {"number": 3, "body": "neither marker", "files": [], "createdAt": "x"},
        ]
    )

    def fake_gh(cmd: list[str]) -> str:
        return response

    prs = handler.list_matching_prs(gh_runner=fake_gh)
    assert sorted(p["number"] for p in prs) == [1, 2]


def _new_compound_pr(
    *,
    number: int = 99,
    body: str | None = None,
    files: list[dict[str, str]] | None = None,
    age_seconds: float = 120,
) -> dict[str, Any]:
    if body is None:
        body = (
            "NEW COMPOUND\n"
            "track=battle\n"
            "compound_name=heuristic_det_damage_term\n"
            "rationale=adds a damage term\n"
        )
    if files is None:
        files = [
            {"path": "src/vgc_ai/policies/heuristic_det_damage.py"},
            {"path": "src/vgc_ai/strategies/registry.py"},
            {"path": "tests/test_heuristic_det_damage.py"},
        ]
    created_at = (datetime.now(UTC) - timedelta(seconds=age_seconds)).isoformat()
    return {"number": number, "body": body, "files": files, "createdAt": created_at}


def test_evaluate_pr_new_compound_merges_when_ci_green(tmp_path: Path) -> None:
    """NEW COMPOUND PR with allowed-prefix files and green CI must merge.

    No bench re-verification — the compound is brand new and has no
    bench history yet. Only CI gates it.
    """

    def green_ci(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
        return (0, "", "")

    pr = _new_compound_pr()
    action, reason = handler.evaluate_pr(pr, tmp_path, tmp_path, cmd_runner=green_ci)
    assert action == "merge"
    assert reason == "all gates passed"


def test_evaluate_pr_new_compound_closes_on_out_of_scope_file(tmp_path: Path) -> None:
    pr = _new_compound_pr(
        files=[
            {"path": "src/vgc_ai/policies/x.py"},
            {"path": "ops/some_script.sh"},
        ]
    )

    def green_ci(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
        return (0, "", "")

    action, reason = handler.evaluate_pr(pr, tmp_path, tmp_path, cmd_runner=green_ci)
    assert action == "close"
    assert "ops/" in reason


def test_evaluate_pr_new_compound_closes_on_ci_fail_past_sla(tmp_path: Path) -> None:
    def red_ci(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
        return (1, "", "pytest exit 1")

    pr = _new_compound_pr(age_seconds=400)
    action, reason = handler.evaluate_pr(pr, tmp_path, tmp_path, max_age_sec=300, cmd_runner=red_ci)
    assert action == "close"
    assert "SLA=300" in reason


def test_evaluate_pr_new_compound_does_not_run_bench_recheck(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """NEW COMPOUND PRs must NOT call bench_gate_still_fires (the new
    compound has no bench history to verify against)."""

    def fail_if_called(*a: Any, **k: Any) -> tuple[bool, str]:
        pytest.fail("bench_gate_still_fires must not be called for NEW COMPOUND PRs")

    monkeypatch.setattr(handler, "bench_gate_still_fires", fail_if_called)

    def green_ci(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
        return (0, "", "")

    pr = _new_compound_pr()
    action, _ = handler.evaluate_pr(pr, tmp_path, tmp_path, cmd_runner=green_ci)
    assert action == "merge"


def test_evaluate_pr_closes_when_body_has_neither_marker(tmp_path: Path) -> None:
    pr = {
        "number": 1,
        "body": "just a regular PR with no marker",
        "files": [{"path": "src/vgc_ai/policies/x.py"}],
        "createdAt": (datetime.now(UTC) - timedelta(seconds=120)).isoformat(),
    }
    action, reason = handler.evaluate_pr(pr, tmp_path, tmp_path)
    assert action == "close"
    assert "missing BENCH GATE or NEW COMPOUND" in reason
