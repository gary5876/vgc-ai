"""Championship A/B harness for selection-policy variants.

History: PR #19 ran a doubles-matchup-table SelectionPolicy through this
harness and recorded a -86 ELO mean regression vs the type-chart proxy.
The table-using policy was discarded; only its negative-result write-up
survives in ``vgc_ai.policies.selection``'s module docstring.

**Current state (round-robin audit, 2026-05-19): both A and B are
functionally identical.** ``MatchupAwareSelectionPolicy`` (in
``selection.py``) and the inline ``_TypeChartOnlySelectionPolicy`` here
both score via ``_selection_score`` / ``_type_chart_score`` (the same
function — line 86 of ``selection.py`` aliases one to the other). Running
this script now produces only Championship-variance noise; the docstring
claim that A "uses the doubles matchup table when meta is provided" has
been stale since merge time.

The harness is retained because the natural next experiment — an
LP-minimax SelectionPolicy over the doubles table (see
``vgc_ai.policies.selection``'s module docstring for the proposed shape)
— will need exactly this competitor-shape. A future PR should add the new
variant as a named ``SelectionPolicy`` class and have B import it instead
of the now-redundant inline shim.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import UTC, datetime
from typing import TypedDict

import numpy as np
from vgc2.agent import SelectionCommand, SelectionPolicy, TeamBuildPolicy
from vgc2.balance.meta import BasicMeta
from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.team import Team
from vgc2.competition import Competitor, CompetitorManager
from vgc2.competition.ecosystem import Championship, Strategy, label_roster
from vgc2.util.generator import gen_move_set, gen_pkm_roster

from vgc_ai.policies.battle import VgcAiBattlePolicy
from vgc_ai.policies.selection import (
    MatchupAwareSelectionPolicy,
    _type_chart_score,
)
from vgc_ai.policies.teambuild import MatchupTableTeamBuildPolicy

MIN_ELO_DELTA = 50.0


class _TypeChartOnlySelectionPolicy(SelectionPolicy):  # type: ignore[misc]
    """Inline copy of the type-chart scorer; identical to ``MatchupAwareSelectionPolicy``.

    Currently functionally equivalent to the policy it's benched against —
    see the module docstring for the audit note. Retained as the slot a
    future LP-minimax-over-doubles-table variant will fill.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        params: BattleRuleParam = self.params
        scored = [
            (-_type_chart_score(p, opp_team, params), i) for i, p in enumerate(my_team.members)
        ]
        scored.sort()
        return [i for _, i in scored][:max_size]


class _SelectionABCompetitor(Competitor):  # type: ignore[misc]
    """Shared battle + teambuild; parametrized selection."""

    def __init__(
        self,
        name: str,
        selection_policy: SelectionPolicy,
        teambuild_policy: TeamBuildPolicy,
    ) -> None:
        self._name = name
        self._battle = VgcAiBattlePolicy()
        self._selection = selection_policy
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


class DoublesSelectionABResult(TypedDict):
    timestamp: str
    epochs: int
    n_battles: int
    n_active: int
    max_team_size: int
    max_pkm_moves: int
    roster_size: int
    n_moves: int
    doubles_table_elo: int
    typechart_elo: int
    elo_delta: float
    elapsed_sec: float


def run_doubles_selection_ab(
    *,
    epochs: int,
    n_battles: int,
    n_active: int,
    max_team_size: int,
    max_pkm_moves: int,
    roster_size: int,
    n_moves: int,
    seed: int | None,
) -> DoublesSelectionABResult:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    move_set = gen_move_set(n_moves)
    roster = gen_pkm_roster(roster_size, move_set)
    label_roster(move_set, roster)
    meta = BasicMeta(move_set, roster)

    a = _SelectionABCompetitor(
        "selection-doubles-table",
        MatchupAwareSelectionPolicy(),
        MatchupTableTeamBuildPolicy(),
    )
    b = _SelectionABCompetitor(
        "selection-typechart",
        _TypeChartOnlySelectionPolicy(),
        MatchupTableTeamBuildPolicy(),
    )

    championship = Championship(
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
    cm_a = CompetitorManager(a)
    cm_b = CompetitorManager(b)
    championship.register(cm_a)
    championship.register(cm_b)

    t0 = time.perf_counter()
    championship.run()
    elapsed = time.perf_counter() - t0

    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "epochs": epochs,
        "n_battles": n_battles,
        "n_active": n_active,
        "max_team_size": max_team_size,
        "max_pkm_moves": max_pkm_moves,
        "roster_size": roster_size,
        "n_moves": n_moves,
        "doubles_table_elo": int(cm_a.elo),
        "typechart_elo": int(cm_b.elo),
        "elo_delta": round(cm_a.elo - cm_b.elo, 2),
        "elapsed_sec": round(elapsed, 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.run_selection_doubles_ab")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--n-battles", type=int, default=3)
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument("--max-team-size", type=int, default=4)
    p.add_argument("--max-pkm-moves", type=int, default=4)
    p.add_argument("--roster-size", type=int, default=30)
    p.add_argument("--n-moves", type=int, default=60)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--min-elo-delta", type=float, default=MIN_ELO_DELTA)
    args = p.parse_args(argv)

    result = run_doubles_selection_ab(
        epochs=args.epochs,
        n_battles=args.n_battles,
        n_active=args.n_active,
        max_team_size=args.max_team_size,
        max_pkm_moves=args.max_pkm_moves,
        roster_size=args.roster_size,
        n_moves=args.n_moves,
        seed=args.seed,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)

    if result["elo_delta"] < args.min_elo_delta:
        print(
            f"FAIL: elo_delta={result['elo_delta']} < {args.min_elo_delta}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
