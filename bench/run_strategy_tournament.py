"""All-vs-all strategy tournament per track.

Drives the strategy registry's compounds head-to-head and appends rows to
``bench/strategies/{track}.csv``. The reviewer loop (PR 3) reads these
CSVs and decides when a candidate has dethroned the current default.

One subcommand per track:

- ``battle`` — Per-pair duel via ``vgc_ai.eval.duel.duel``. Reads
  ``BATTLE_STRATEGIES``; battle policy is the only differentiator.
- ``championship`` — Per-pair competitor-vs-competitor head-to-head via
  ``vgc2.competition.match.Match`` with ``gen=gen_team`` (routes to
  ``Match._run_random``). **Does NOT call ``set_meta``** — every
  meta-aware compound runs with empty meta on this subcommand. Kept as
  a "no-meta baseline" and for backward compatibility with the VM bench
  loop; prefer ``championship_real`` for actual Championship Track
  evaluation.
- ``championship_real`` — Two-phase Championship-based bench. Phase 1
  registers every ``CHAMPIONSHIP_STRATEGIES`` competitor into a
  ``vgc2.competition.ecosystem.Championship`` with a fresh ``BasicMeta``
  and runs ``--epochs`` epochs to populate the meta (``set_meta`` IS
  called per match via ``Match._run_non_random``). Phase 2 plays
  default-vs-challenger pairwise matches with the warmed meta and teams
  rebuilt against current usage data. Output schema matches the
  ``championship`` subcommand so the reviewer / proposer / pr_auto_handler
  consume it unchanged.
- ``balance`` — Smoke validation: each ``DesignCompetitor`` constructs
  and exposes a legal policy. Reads ``BALANCE_STRATEGIES``. A real
  evaluator-driven bench waits on ``MetaEvaluator`` / ``RuleEvaluator``
  being filled in by Reis (still stubs in vgc2 v2.1.1).

Each invocation runs ONE round: iterate ordered pairs, append rows, exit.
The VM bench loop calls this once per cycle. The reviewer pools across
recent rows to clear the noise floor.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from vgc2.balance.meta import BasicMeta
from vgc2.battle_engine.constants import BattleRuleParam
from vgc2.battle_engine.security import sanitized_team_build_decision
from vgc2.competition import (
    Competitor,
    CompetitorManager,
    DesignCompetitor,
)
from vgc2.competition.ecosystem import Championship, Strategy, build_team, label_roster
from vgc2.competition.match import Match
from vgc2.util.generator import gen_move_set, gen_pkm_roster, gen_team

from vgc_ai.eval.duel import duel
from vgc_ai.policies.battle import VgcAiBattlePolicy
from vgc_ai.strategies import (
    BALANCE_DEFAULT,
    BALANCE_STRATEGIES,
    BATTLE_DEFAULT,
    BATTLE_STRATEGIES,
    CHAMPIONSHIP_DEFAULT,
    CHAMPIONSHIP_STRATEGIES,
    BalanceStrategy,
    ChampionshipStrategy,
)

DEFAULT_BATTLE_N = 200
DEFAULT_CHAMPIONSHIP_N = 50
DEFAULT_CHAMPIONSHIP_REAL_N = 20
DEFAULT_CHAMPIONSHIP_REAL_EPOCHS = 30
DEFAULT_CHAMPIONSHIP_REAL_ROSTER_SIZE = 30
DEFAULT_CHAMPIONSHIP_REAL_N_MOVES = 60
DEFAULT_OUTPUT_DIR = Path("bench/strategies")

BATTLE_HEADER = [
    "timestamp",
    "track",
    "strategy_a",
    "strategy_b",
    "n_battles",
    "wins_a",
    "wins_b",
    "ties",
    "win_rate_a",
    "ci95_low",
    "ci95_high",
    "elapsed_sec",
    "avg_battle_ms",
    "avg_turn_ms_a",
    "avg_turn_ms_b",
    "is_default_a",
    "is_default_b",
]

CHAMPIONSHIP_HEADER = [
    "timestamp",
    "track",
    "strategy_a",
    "strategy_b",
    "n_battles",
    "wins_a",
    "wins_b",
    "win_rate_a",
    "ci95_low",
    "ci95_high",
    "elapsed_sec",
    "is_default_a",
    "is_default_b",
]

BALANCE_HEADER = [
    "timestamp",
    "track",
    "strategy",
    "validated",
    "elapsed_sec",
    "note",
    "is_default",
]


def wilson_ci_95(wins: int, n: int) -> tuple[float, float]:
    """Wilson 95% confidence interval on a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def _ordered_pairs(names: list[str]) -> Iterator[tuple[str, str]]:
    """All ordered pairs ``(a, b)`` with ``a != b``.

    Ordered (not combinations) because head-to-head win rates aren't strictly
    symmetric across orderings — selection orders members, leads differ.
    """
    for a in names:
        for b in names:
            if a != b:
                yield (a, b)


def _ensure_parent_and_header(path: Path, header: list[str]) -> None:
    """Create ``path``'s parent dir if missing and write ``header`` if the file is new.

    ``csv.writer`` returns an opaque internal type that doesn't compose well
    under ``mypy --strict``; rather than smuggle it through a helper, callers
    keep their own ``csv.writer(f)`` inside a ``with`` block and rely on this
    helper for the one-time header guarantee.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def run_battle_tournament(n_battles: int, output_path: Path) -> int:
    """All-vs-all duels between ``BATTLE_STRATEGIES``."""
    _ensure_parent_and_header(output_path, BATTLE_HEADER)
    names = list(BATTLE_STRATEGIES)
    with output_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for a_name, b_name in _ordered_pairs(names):
            strat_a = BATTLE_STRATEGIES[a_name]
            strat_b = BATTLE_STRATEGIES[b_name]
            t0 = time.perf_counter()
            res = duel(strat_a.battle_policy, strat_b.battle_policy, n_battles=n_battles)
            elapsed = time.perf_counter() - t0
            decided = res.wins_a + res.wins_b
            ci_low, ci_high = wilson_ci_95(res.wins_a, decided) if decided else (0.0, 0.0)
            w.writerow(
                [
                    _now(),
                    "battle",
                    a_name,
                    b_name,
                    res.n_battles,
                    res.wins_a,
                    res.wins_b,
                    res.ties,
                    round(res.win_rate_a, 4),
                    round(ci_low, 4),
                    round(ci_high, 4),
                    round(elapsed, 3),
                    round(elapsed * 1000.0 / max(1, res.n_battles), 2),
                    round(res.avg_turn_ms_a, 3),
                    round(res.avg_turn_ms_b, 3),
                    int(a_name == BATTLE_DEFAULT),
                    int(b_name == BATTLE_DEFAULT),
                ]
            )
            f.flush()
            print(
                f"  battle: {a_name} vs {b_name}: "
                f"{res.wins_a}-{res.wins_b}-{res.ties} "
                f"(wr={res.win_rate_a:.1%}, ci95=[{ci_low:.3f},{ci_high:.3f}])",
                file=sys.stderr,
            )
    return 0


class _CompoundCompetitor(Competitor):  # type: ignore[misc]
    """vgc2 Competitor parametrized by a ChampionshipStrategy.

    Battle policy is held to the project's current battle default; the
    Championship Track tests team-build + selection, not the battle policy.
    """

    def __init__(self, name: str, strategy: ChampionshipStrategy) -> None:
        self._name = name
        self._battle = VgcAiBattlePolicy()
        self._teambuild = strategy.team_build_policy()
        self._selection = strategy.selection_policy()

    @property
    def battlepolicy(self) -> Any:
        return self._battle

    @property
    def selectionpolicy(self) -> Any:
        return self._selection

    @property
    def teambuildpolicy(self) -> Any:
        return self._teambuild

    @property
    def name(self) -> str:
        return self._name


def run_championship_tournament(n_battles: int, output_path: Path) -> int:
    """All-vs-all head-to-head Match between ``CHAMPIONSHIP_STRATEGIES``."""
    _ensure_parent_and_header(output_path, CHAMPIONSHIP_HEADER)
    names = list(CHAMPIONSHIP_STRATEGIES)
    with output_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for a_name, b_name in _ordered_pairs(names):
            a = _CompoundCompetitor(a_name, CHAMPIONSHIP_STRATEGIES[a_name])
            b = _CompoundCompetitor(b_name, CHAMPIONSHIP_STRATEGIES[b_name])
            cm = (CompetitorManager(a), CompetitorManager(b))
            # Match runs two battles per loop iteration (one per side ordering),
            # so n_battles=N//2 yields ~N actual battles. A tiebreaker may push
            # the total slightly higher; we record the actual total below.
            match = Match(
                cm,
                n_active=2,
                n_battles=max(1, n_battles // 2),
                gen=gen_team,
                params=BattleRuleParam(),
            )
            t0 = time.perf_counter()
            match.run()
            elapsed = time.perf_counter() - t0
            wins_a, wins_b = match.wins
            total = wins_a + wins_b
            ci_low, ci_high = wilson_ci_95(wins_a, total) if total else (0.0, 0.0)
            win_rate_a = wins_a / total if total else 0.0
            w.writerow(
                [
                    _now(),
                    "championship",
                    a_name,
                    b_name,
                    total,
                    wins_a,
                    wins_b,
                    round(win_rate_a, 4),
                    round(ci_low, 4),
                    round(ci_high, 4),
                    round(elapsed, 3),
                    int(a_name == CHAMPIONSHIP_DEFAULT),
                    int(b_name == CHAMPIONSHIP_DEFAULT),
                ]
            )
            f.flush()
            print(
                f"  championship: {a_name} vs {b_name}: "
                f"{wins_a}-{wins_b} "
                f"(wr={win_rate_a:.1%}, ci95=[{ci_low:.3f},{ci_high:.3f}])",
                file=sys.stderr,
            )
    return 0


def run_championship_real_tournament(
    *,
    n_battles: int,
    output_path: Path,
    epochs: int = DEFAULT_CHAMPIONSHIP_REAL_EPOCHS,
    seed: int | None = None,
    roster_size: int = DEFAULT_CHAMPIONSHIP_REAL_ROSTER_SIZE,
    n_moves: int = DEFAULT_CHAMPIONSHIP_REAL_N_MOVES,
    max_team_size: int = 4,
    max_pkm_moves: int = 4,
    n_active: int = 2,
) -> int:
    """Championship Track bench with REAL meta accumulation.

    Two phases:

    1. **Warmup**: register every ``CHAMPIONSHIP_STRATEGIES`` competitor
       into a ``Championship`` with a fresh ``BasicMeta`` and run
       ``epochs`` epochs. ``Match._run_non_random`` is invoked under the
       hood, which DOES call ``set_meta`` on every selector + battle
       agent (line 114-118 of vgc2/competition/match.py). The meta
       accumulates real usage data across epochs.
    2. **Pairwise**: with the warmed meta, run a fresh ``Match`` for
       each ordered ``(default, challenger)`` pair using teams rebuilt
       against the current meta. One CSV row per pair in the existing
       ``CHAMPIONSHIP_HEADER`` schema so the reviewer / proposer /
       pr_auto_handler keep working unchanged.

    Why this exists: ``run_championship_tournament`` (above) uses
    ``Match(..., gen=gen_team)`` which routes to ``Match._run_random``,
    a code path that **never calls set_meta**. Every meta-aware compound
    in the registry has been bench-tested with empty meta. This
    subcommand is the structural fix; the old one is preserved for
    backward compatibility (and as a baseline for "performance with no
    meta").
    """
    _ensure_parent_and_header(output_path, CHAMPIONSHIP_HEADER)

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    move_set = gen_move_set(n_moves)
    roster = gen_pkm_roster(roster_size, move_set)
    label_roster(move_set, roster)
    meta = BasicMeta(move_set, roster)

    champ = Championship(
        roster,
        meta,
        epochs=epochs,
        n_active=n_active,
        n_battles=n_battles,
        max_team_size=max_team_size,
        max_pkm_moves=max_pkm_moves,
        strategy=Strategy.RANDOM_PAIRING,
        client=None,
    )
    managers: dict[str, CompetitorManager] = {}
    for name, strategy in CHAMPIONSHIP_STRATEGIES.items():
        comp = _CompoundCompetitor(name, strategy)
        cm = CompetitorManager(comp)
        champ.register(cm)
        managers[name] = cm

    print(
        f"  championship-real WARMUP: {len(managers)} competitors, "
        f"epochs={epochs}, n_battles={n_battles}",
        file=sys.stderr,
    )
    warmup_t0 = time.perf_counter()
    champ.run()
    warmup_elapsed = time.perf_counter() - warmup_t0
    print(
        f"  championship-real WARMUP done in {warmup_elapsed:.1f}s, "
        f"meta has {len(meta.record)} matches recorded",
        file=sys.stderr,
    )

    if CHAMPIONSHIP_DEFAULT not in managers:
        print(
            f"  championship-real: default {CHAMPIONSHIP_DEFAULT!r} not in registry, skipping pairwise phase",
            file=sys.stderr,
        )
        return 0

    challengers = [n for n in managers if n != CHAMPIONSHIP_DEFAULT]
    with output_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for a_name, b_name in _ordered_pairs([CHAMPIONSHIP_DEFAULT, *challengers]):
            # Only emit rows touching the default — keeps cycle time
            # bounded (2*(N-1) pairs instead of N*(N-1)) and matches the
            # reviewer's gate (strategy_a == cand AND strategy_b == default).
            if CHAMPIONSHIP_DEFAULT not in (a_name, b_name):
                continue
            cm_a, cm_b = managers[a_name], managers[b_name]
            # Rebuild teams against the warmed meta so each pair's
            # decision reflects current usage data, not the team frozen
            # at championship epoch ``epochs-1``.
            cm_a.team = build_team(
                sanitized_team_build_decision(
                    cm_a.competitor.teambuildpolicy,
                    roster,
                    meta,
                    max_team_size,
                    max_pkm_moves,
                    n_active,
                ),
                roster,
            )
            cm_b.team = build_team(
                sanitized_team_build_decision(
                    cm_b.competitor.teambuildpolicy,
                    roster,
                    meta,
                    max_team_size,
                    max_pkm_moves,
                    n_active,
                ),
                roster,
            )
            match = Match(
                (cm_a, cm_b),
                n_active=n_active,
                n_battles=max(1, n_battles // 2),
                max_team_size=max_team_size,
                max_pkm_moves=max_pkm_moves,
                meta=meta,
                params=BattleRuleParam(),
            )
            t0 = time.perf_counter()
            match.run()
            elapsed = time.perf_counter() - t0
            wins_a, wins_b = match.wins
            total = wins_a + wins_b
            ci_low, ci_high = wilson_ci_95(wins_a, total) if total else (0.0, 0.0)
            win_rate_a = wins_a / total if total else 0.0
            w.writerow(
                [
                    _now(),
                    "championship",
                    a_name,
                    b_name,
                    total,
                    wins_a,
                    wins_b,
                    round(win_rate_a, 4),
                    round(ci_low, 4),
                    round(ci_high, 4),
                    round(elapsed, 3),
                    int(a_name == CHAMPIONSHIP_DEFAULT),
                    int(b_name == CHAMPIONSHIP_DEFAULT),
                ]
            )
            f.flush()
            print(
                f"  championship-real: {a_name} vs {b_name}: "
                f"{wins_a}-{wins_b} "
                f"(wr={win_rate_a:.1%}, ci95=[{ci_low:.3f},{ci_high:.3f}])",
                file=sys.stderr,
            )
    return 0


class _BalanceDesignCompetitor(DesignCompetitor):  # type: ignore[misc]
    def __init__(self, name: str, strategy: BalanceStrategy) -> None:
        self._name = name
        self._meta = strategy.meta_balance_policy()
        self._rule = strategy.rule_balance_policy()

    @property
    def metabalancepolicy(self) -> Any:
        return self._meta

    @property
    def rulebalancepolicy(self) -> Any:
        return self._rule

    @property
    def name(self) -> str:
        return self._name


def run_balance_smoke(output_path: Path) -> int:
    """Smoke-validate each balance strategy.

    The Balance Track needs MetaEvaluator / RuleEvaluator implementations,
    still empty stubs in vgc2 v2.1.1. Until those land, we verify each
    strategy's DesignCompetitor constructs and exposes both policies.
    """
    _ensure_parent_and_header(output_path, BALANCE_HEADER)
    with output_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for name, strat in BALANCE_STRATEGIES.items():
            t0 = time.perf_counter()
            note: str
            validated: bool
            try:
                comp = _BalanceDesignCompetitor(name, strat)
                if comp.metabalancepolicy is None or comp.rulebalancepolicy is None:
                    raise RuntimeError("policy is None")
                validated = True
                note = "construction-ok"
            except Exception as e:
                validated = False
                note = f"construction-failed: {e}"
            elapsed = time.perf_counter() - t0
            w.writerow(
                [
                    _now(),
                    "balance",
                    name,
                    int(validated),
                    round(elapsed, 4),
                    note,
                    int(name == BALANCE_DEFAULT),
                ]
            )
            f.flush()
            print(f"  balance: {name}: validated={validated} ({note})", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.run_strategy_tournament")
    sub = p.add_subparsers(dest="track", required=True)

    b = sub.add_parser("battle", help="Battle Track all-vs-all duels")
    b.add_argument("--n", type=int, default=DEFAULT_BATTLE_N)
    b.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "battle.csv")

    c = sub.add_parser(
        "championship", help="Championship Track all-vs-all Match (no meta — see championship_real)"
    )
    c.add_argument("--n", type=int, default=DEFAULT_CHAMPIONSHIP_N)
    c.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "championship.csv")

    cr = sub.add_parser(
        "championship_real",
        help="Championship Track with REAL meta accumulation (Championship-based, calls set_meta)",
    )
    cr.add_argument("--n", type=int, default=DEFAULT_CHAMPIONSHIP_REAL_N)
    cr.add_argument("--epochs", type=int, default=DEFAULT_CHAMPIONSHIP_REAL_EPOCHS)
    cr.add_argument("--seed", type=int, default=None)
    cr.add_argument("--roster-size", type=int, default=DEFAULT_CHAMPIONSHIP_REAL_ROSTER_SIZE)
    cr.add_argument("--n-moves", type=int, default=DEFAULT_CHAMPIONSHIP_REAL_N_MOVES)
    cr.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "championship.csv")

    bal = sub.add_parser("balance", help="Balance Track construction smoke")
    bal.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "balance.csv")

    args = p.parse_args(argv)
    if args.track == "battle":
        return run_battle_tournament(args.n, args.output)
    if args.track == "championship":
        return run_championship_tournament(args.n, args.output)
    if args.track == "championship_real":
        return run_championship_real_tournament(
            n_battles=args.n,
            output_path=args.output,
            epochs=args.epochs,
            seed=args.seed,
            roster_size=args.roster_size,
            n_moves=args.n_moves,
        )
    if args.track == "balance":
        return run_balance_smoke(args.output)
    p.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
