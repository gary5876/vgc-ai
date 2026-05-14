"""Head-to-head A/B for team-build variants on the same roster.

Drives ``vgc2.competition.match.Match`` between two competitors that differ
*only* in their ``TeamBuildPolicy``. Counts individual *battles* (not match
winners) for statistical power, the same way ``bench.run_selection_match``
does, and reports a Wilson 95% confidence interval.

Use case: A/B the per-species-optimized ``MetaUsageTeamBuildPolicy`` against
its prior flat-default version on the same Random-roster opponent. The
flat-default class is reconstructed inline here so the bench can isolate
the optimization without depending on a separate branch / commit.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from typing import TypedDict

from vgc2.agent import TeamBuildCommand, TeamBuildPolicy
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import RandomSelectionPolicy
from vgc2.balance.meta import Meta, Roster
from vgc2.battle_engine.modifiers import Nature
from vgc2.competition import Competitor, CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_team

from vgc_ai.policies.teambuild import (
    MetaUsageTeamBuildPolicy,
    _move_priority,
    _species_priority,
)


class _FlatMetaUsageTeamBuildPolicy(TeamBuildPolicy):  # type: ignore[misc]
    """Pre-optimization MetaUsage: flat (85,)*6 EVs, Nature.SERIOUS, top moves."""

    _FLAT_EVS: tuple[int, int, int, int, int, int] = (85, 85, 85, 85, 85, 85)
    _FLAT_IVS: tuple[int, int, int, int, int, int] = (31, 31, 31, 31, 31, 31)

    def decision(
        self,
        roster: Roster,
        meta: Meta | None,
        max_team_size: int,
        max_pkm_moves: int,
        n_active: int,
    ) -> TeamBuildCommand:
        ranked = _species_priority(roster, meta)[:max_team_size]
        cmds: TeamBuildCommand = []
        for species_idx in ranked:
            species = roster[species_idx]
            move_idx = _move_priority(species)[:max_pkm_moves]
            cmds.append((species_idx, self._FLAT_EVS, self._FLAT_IVS, Nature.SERIOUS, move_idx))
        return cmds


class _TeamBuildABCompetitor(Competitor):  # type: ignore[misc]
    """Greedy battle + Random selection + parametrized teambuild."""

    def __init__(self, name: str, teambuild_policy: TeamBuildPolicy) -> None:
        self._name = name
        self._battle = GreedyBattlePolicy()
        self._selection = RandomSelectionPolicy()
        self._teambuild = teambuild_policy

    @property
    def battlepolicy(self):  # type: ignore[no-untyped-def]
        return self._battle

    @property
    def selectionpolicy(self):  # type: ignore[no-untyped-def]
        return self._selection

    @property
    def teambuildpolicy(self):  # type: ignore[no-untyped-def]
        return self._teambuild

    @property
    def name(self) -> str:
        return self._name


def wilson_ci_95(wins: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


class TeamBuildABResult(TypedDict):
    timestamp: str
    n_matches: int
    n_battles_per_match: int
    n_battles_total: int
    n_active: int
    max_team_size: int
    max_pkm_moves: int
    wins_a: int
    wins_b: int
    win_rate_a: float
    ci95_low: float
    ci95_high: float
    elapsed_sec: float


def run_teambuild_ab(
    *,
    n_matches: int,
    n_battles_per_match: int,
    n_active: int,
    max_team_size: int,
    max_pkm_moves: int,
) -> TeamBuildABResult:
    a = _TeamBuildABCompetitor("metausage-optimized", MetaUsageTeamBuildPolicy())
    b = _TeamBuildABCompetitor("metausage-flat", _FlatMetaUsageTeamBuildPolicy())
    cm = (CompetitorManager(a), CompetitorManager(b))

    wins_a = 0
    wins_b = 0
    t0 = time.perf_counter()
    for _ in range(n_matches):
        match = Match(
            cm,
            n_active=n_active,
            n_battles=n_battles_per_match,
            max_team_size=max_team_size,
            max_pkm_moves=max_pkm_moves,
            gen=gen_team,
        )
        match.run()
        a_battles, b_battles = match.wins
        wins_a += a_battles
        wins_b += b_battles
        match.wins = [0, 0]
    elapsed = time.perf_counter() - t0

    decided = wins_a + wins_b
    ci_low, ci_high = wilson_ci_95(wins_a, decided) if decided else (0.0, 0.0)
    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "n_matches": n_matches,
        "n_battles_per_match": n_battles_per_match,
        "n_battles_total": decided,
        "n_active": n_active,
        "max_team_size": max_team_size,
        "max_pkm_moves": max_pkm_moves,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "win_rate_a": round(wins_a / decided, 4) if decided else 0.0,
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "elapsed_sec": round(elapsed, 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.run_teambuild_ab")
    p.add_argument("--n-matches", type=int, default=100)
    p.add_argument("--n-battles-per-match", type=int, default=3)
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument("--max-team-size", type=int, default=4)
    p.add_argument("--max-pkm-moves", type=int, default=4)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--min-ci95-low", type=float, default=0.5)
    args = p.parse_args(argv)

    result = run_teambuild_ab(
        n_matches=args.n_matches,
        n_battles_per_match=args.n_battles_per_match,
        n_active=args.n_active,
        max_team_size=args.max_team_size,
        max_pkm_moves=args.max_pkm_moves,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)

    if result["ci95_low"] <= args.min_ci95_low:
        print(
            f"FAIL: ci95_low={result['ci95_low']} <= {args.min_ci95_low}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
