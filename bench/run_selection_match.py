"""Match-based bench isolating the selection policy as the only differentiator.

Drives ``vgc2.competition.match.Match`` between two competitors that share
identical battle and team-build policies (Greedy + Random respectively); only
the selection policy differs (``MatchupAwareSelectionPolicy`` vs the
framework's ``RandomSelectionPolicy``). Battle policy is intentionally Greedy
rather than ``heuristic_det`` to keep the bench fast — the variable under
test is selection, not battle.

We count individual *battles* across all matches (not match winners) so each
random team draw contributes a binary outcome; collapsing 3 battles into one
match-winner throws away statistical power. Outputs Wilson 95% confidence
interval on per-battle win rate. Acceptance gate: ``ci95_low > 0.5``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from typing import TypedDict

from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import RandomSelectionPolicy
from vgc2.agent.teambuild import RandomTeamBuildPolicy
from vgc2.competition import Competitor, CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_team

from vgc_ai.policies.selection import MatchupAwareSelectionPolicy


class _SelectionTestCompetitor(Competitor):  # type: ignore[misc]
    """Greedy battle + Random teambuild + parametrized selection."""

    def __init__(self, name: str, selection_policy: object) -> None:
        self._name = name
        self._battle = GreedyBattlePolicy()
        self._selection = selection_policy
        self._teambuild = RandomTeamBuildPolicy()

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


class SelectionMatchResult(TypedDict):
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


def run_selection_match(
    *,
    n_matches: int,
    n_battles_per_match: int,
    n_active: int,
    max_team_size: int,
    max_pkm_moves: int,
) -> SelectionMatchResult:
    a = _SelectionTestCompetitor("matchup-aware", MatchupAwareSelectionPolicy())
    b = _SelectionTestCompetitor("random-selection", RandomSelectionPolicy())
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
        match.wins = [0, 0]  # reset per-match for the next iteration
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
    p = argparse.ArgumentParser(prog="bench.run_selection_match")
    p.add_argument("--n-matches", type=int, default=200)
    p.add_argument("--n-battles-per-match", type=int, default=3)
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument("--max-team-size", type=int, default=4)
    p.add_argument("--max-pkm-moves", type=int, default=4)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--min-ci95-low", type=float, default=0.5)
    args = p.parse_args(argv)

    result = run_selection_match(
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
