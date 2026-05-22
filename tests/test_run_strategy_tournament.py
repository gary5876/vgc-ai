"""Tests for ``bench.run_strategy_tournament``.

End-to-end coverage: each subcommand runs against a (possibly shrunk)
registry, the CSV gets the right header, and one row per pair / strategy
lands as expected. The Wilson CI helper is tested directly because its
output is what the reviewer loop's gate depends on.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from bench import run_strategy_tournament as mod

from vgc_ai.strategies import (
    BALANCE_STRATEGIES,
    BATTLE_STRATEGIES,
    CHAMPIONSHIP_STRATEGIES,
)


def test_wilson_ci_zero_n() -> None:
    assert mod.wilson_ci_95(0, 0) == (0.0, 0.0)


def test_wilson_ci_typical_case() -> None:
    # n=2000, wins=1081 is the certified heuristic_det-vs-greedy bench
    # (PR #9): win_rate=0.5405, ci95_low=0.5186, ci95_high=0.5622. The
    # implementation here should reproduce those bounds to 4 decimal places.
    lo, hi = mod.wilson_ci_95(1081, 2000)
    assert round(lo, 4) == 0.5186
    assert round(hi, 4) == 0.5622


def test_wilson_ci_full_pass() -> None:
    lo, hi = mod.wilson_ci_95(10, 10)
    assert lo > 0.7  # lower bound rises with n
    assert hi == 1.0


def test_ordered_pairs_excludes_self() -> None:
    pairs = list(mod._ordered_pairs(["a", "b", "c"]))
    assert ("a", "a") not in pairs
    assert ("a", "b") in pairs
    assert ("b", "a") in pairs  # ordered, not combinations
    assert len(pairs) == 6  # 3*2


def test_battle_tournament_writes_one_row_per_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Shrink the registry to two policies for a fast end-to-end test.
    tiny = {k: BATTLE_STRATEGIES[k] for k in ("random", "greedy")}
    monkeypatch.setattr(mod, "BATTLE_STRATEGIES", tiny)

    output_path = tmp_path / "battle.csv"
    rc = mod.run_battle_tournament(n_battles=2, output_path=output_path)
    assert rc == 0

    with output_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # (random, greedy) and (greedy, random)
    pair_set = {(r["strategy_a"], r["strategy_b"]) for r in rows}
    assert pair_set == {("random", "greedy"), ("greedy", "random")}
    for r in rows:
        assert r["track"] == "battle"
        assert int(r["n_battles"]) == 2
        assert int(r["wins_a"]) + int(r["wins_b"]) + int(r["ties"]) == 2


def test_battle_tournament_appends_without_duplicating_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tiny = {k: BATTLE_STRATEGIES[k] for k in ("random", "greedy")}
    monkeypatch.setattr(mod, "BATTLE_STRATEGIES", tiny)
    output_path = tmp_path / "battle.csv"
    mod.run_battle_tournament(n_battles=1, output_path=output_path)
    mod.run_battle_tournament(n_battles=1, output_path=output_path)
    lines = output_path.read_text(encoding="utf-8").splitlines()
    # header + 2 pairs * 2 rounds = 5 lines, header appears once
    assert len(lines) == 5
    assert lines[0].startswith("timestamp,track,")
    assert "timestamp,track," not in "\n".join(lines[1:])


def test_battle_tournament_records_default_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Force a known default that's in the shrunk registry.
    tiny = {k: BATTLE_STRATEGIES[k] for k in ("random", "greedy")}
    monkeypatch.setattr(mod, "BATTLE_STRATEGIES", tiny)
    monkeypatch.setattr(mod, "BATTLE_DEFAULT", "greedy")
    output_path = tmp_path / "battle.csv"
    mod.run_battle_tournament(n_battles=1, output_path=output_path)
    with output_path.open() as f:
        rows = list(csv.DictReader(f))
    flagged_a = [r for r in rows if r["is_default_a"] == "1"]
    flagged_b = [r for r in rows if r["is_default_b"] == "1"]
    assert {r["strategy_a"] for r in flagged_a} == {"greedy"}
    assert {r["strategy_b"] for r in flagged_b} == {"greedy"}


def test_championship_tournament_writes_one_row_per_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two strategies → 2 ordered pairs. Match runs ~2 battles per pair at
    # n=2, so a 4-battle smoke run total — slow per battle but bounded.
    tiny = {k: CHAMPIONSHIP_STRATEGIES[k] for k in ("minimax+matchup_aware", "random+random")}
    monkeypatch.setattr(mod, "CHAMPIONSHIP_STRATEGIES", tiny)

    output_path = tmp_path / "championship.csv"
    rc = mod.run_championship_tournament(n_battles=2, output_path=output_path)
    assert rc == 0

    with output_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    pair_set = {(r["strategy_a"], r["strategy_b"]) for r in rows}
    assert pair_set == {
        ("minimax+matchup_aware", "random+random"),
        ("random+random", "minimax+matchup_aware"),
    }
    for r in rows:
        assert r["track"] == "championship"
        # Wilson bounds are well-formed floats in [0, 1].
        ci_lo = float(r["ci95_low"])
        ci_hi = float(r["ci95_high"])
        assert 0.0 <= ci_lo <= ci_hi <= 1.0


def test_championship_real_writes_default_vs_challenger_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two strategies, default plus one challenger. Phase 1 (warmup) runs
    # 2 epochs of Championship; Phase 2 emits the (default, challenger)
    # and (challenger, default) rows. Small roster/moves keep the smoke
    # bounded; this test exists to verify the wiring (set_meta is called,
    # CSV format is right, rows are emitted) — not to measure win rates.
    tiny = {
        k: CHAMPIONSHIP_STRATEGIES[k]
        for k in ("minimax+meta_threat_aware_selection", "random+random")
    }
    monkeypatch.setattr(mod, "CHAMPIONSHIP_STRATEGIES", tiny)
    monkeypatch.setattr(mod, "CHAMPIONSHIP_DEFAULT", "minimax+meta_threat_aware_selection")

    output_path = tmp_path / "championship.csv"
    rc = mod.run_championship_real_tournament(
        n_battles=2,
        output_path=output_path,
        epochs=2,
        seed=42,
        roster_size=8,
        n_moves=16,
    )
    assert rc == 0

    with output_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2  # default vs challenger in both orderings
    pair_set = {(r["strategy_a"], r["strategy_b"]) for r in rows}
    assert pair_set == {
        ("minimax+meta_threat_aware_selection", "random+random"),
        ("random+random", "minimax+meta_threat_aware_selection"),
    }
    for r in rows:
        assert r["track"] == "championship"
        # Wilson bounds are well-formed.
        ci_lo = float(r["ci95_low"])
        ci_hi = float(r["ci95_high"])
        assert 0.0 <= ci_lo <= ci_hi <= 1.0
        # Exactly one side of each row is flagged as default.
        assert {r["is_default_a"], r["is_default_b"]} == {"0", "1"}


def test_championship_real_skips_pairwise_when_default_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # If the configured default isn't in the (shrunk) registry, phase 2
    # is a no-op — but warmup still runs and exits cleanly.
    tiny = {k: CHAMPIONSHIP_STRATEGIES[k] for k in ("metausage+matchup_aware", "random+random")}
    monkeypatch.setattr(mod, "CHAMPIONSHIP_STRATEGIES", tiny)
    monkeypatch.setattr(mod, "CHAMPIONSHIP_DEFAULT", "not_in_registry")

    output_path = tmp_path / "championship.csv"
    rc = mod.run_championship_real_tournament(
        n_battles=2,
        output_path=output_path,
        epochs=2,
        seed=42,
        roster_size=8,
        n_moves=16,
    )
    assert rc == 0
    # File may contain just the header (if it was newly created) or nothing
    # appended beyond it.
    with output_path.open() as f:
        rows = list(csv.DictReader(f))
    assert rows == []


def test_balance_smoke_writes_one_row_per_strategy(tmp_path: Path) -> None:
    output_path = tmp_path / "balance.csv"
    rc = mod.run_balance_smoke(output_path)
    assert rc == 0
    with output_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(BALANCE_STRATEGIES)
    for r in rows:
        assert r["track"] == "balance"
        assert r["validated"] == "1"
        assert r["note"].startswith("construction-")


def test_main_routes_to_subcommands(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # End-to-end CLI smoke: parse argv, dispatch to the right subcommand.
    monkeypatch.setattr(
        mod, "BATTLE_STRATEGIES", {k: BATTLE_STRATEGIES[k] for k in ("random", "greedy")}
    )
    output = tmp_path / "battle.csv"
    rc = mod.main(["battle", "--n", "1", "--output", str(output)])
    assert rc == 0
    assert output.exists()
