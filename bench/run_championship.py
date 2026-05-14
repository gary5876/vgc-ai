"""In-process Championship bench.

Drives ``vgc2.competition.ecosystem.Championship`` directly with two
``CompetitorManager`` instances — no ``ProxyCompetitor`` / socket protocol.
One side is ``VgcAiCompetitor`` (our policies); the other side is a control
competitor that shares our battle + selection policies but uses a
parametrized team-build policy (``--control random|metausage``) so the
team-build delta is the only differentiator.

Acceptance gate: VgcAiCompetitor's final ELO must exceed the control's by
at least ``--min-elo-delta``. Prints final ranking; exits 0 on pass, 1 on
miss.
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
from vgc2.agent import TeamBuildPolicy
from vgc2.agent.teambuild import RandomTeamBuildPolicy
from vgc2.balance.meta import BasicMeta
from vgc2.competition import Competitor, CompetitorManager
from vgc2.competition.ecosystem import Championship, Strategy, label_roster
from vgc2.util.generator import gen_move_set, gen_pkm_roster

from vgc_ai.competitor import VgcAiCompetitor
from vgc_ai.policies.battle import VgcAiBattlePolicy
from vgc_ai.policies.selection import VgcAiSelectionPolicy
from vgc_ai.policies.teambuild import (
    MatchupTableTeamBuildPolicy,
    MetaUsageTeamBuildPolicy,
    MinimaxTeamBuildPolicy,
)

MIN_ELO_DELTA = 50.0

CONTROL_TEAMBUILDERS: dict[str, type[TeamBuildPolicy]] = {
    "random": RandomTeamBuildPolicy,
    "metausage": MetaUsageTeamBuildPolicy,
    "matchup_table": MatchupTableTeamBuildPolicy,
    "minimax": MinimaxTeamBuildPolicy,
}


class _ControlCompetitor(Competitor):  # type: ignore[misc]
    """Same battle + selection as our submission; parametrized team builder.

    Isolates the team-build policy as the single differentiator vs.
    ``VgcAiCompetitor`` in head-to-head championship play.
    """

    def __init__(self, name: str, teambuild_policy: TeamBuildPolicy) -> None:
        self._name = name
        self._battle = VgcAiBattlePolicy()
        self._selection = VgcAiSelectionPolicy()
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


class ChampionshipResult(TypedDict):
    timestamp: str
    epochs: int
    n_battles: int
    n_active: int
    max_team_size: int
    max_pkm_moves: int
    roster_size: int
    n_moves: int
    control: str
    vgc_ai_elo: int
    control_elo: int
    elo_delta: float
    elapsed_sec: float


def run_championship(
    *,
    epochs: int,
    n_battles: int,
    n_active: int,
    max_team_size: int,
    max_pkm_moves: int,
    roster_size: int,
    n_moves: int,
    seed: int | None,
    control: str = "random",
) -> ChampionshipResult:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    if control not in CONTROL_TEAMBUILDERS:
        raise SystemExit(f"unknown control: {control!r} (known: {sorted(CONTROL_TEAMBUILDERS)})")

    move_set = gen_move_set(n_moves)
    roster = gen_pkm_roster(roster_size, move_set)
    label_roster(move_set, roster)
    meta = BasicMeta(move_set, roster)

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
    ours = VgcAiCompetitor(name="vgc-ai")
    control_policy = CONTROL_TEAMBUILDERS[control]()
    control_competitor = _ControlCompetitor(
        name=f"control-{control}", teambuild_policy=control_policy
    )
    cm_ours = CompetitorManager(ours)
    cm_ctrl = CompetitorManager(control_competitor)
    championship.register(cm_ours)
    championship.register(cm_ctrl)

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
        "control": control,
        "vgc_ai_elo": int(cm_ours.elo),
        "control_elo": int(cm_ctrl.elo),
        "elo_delta": round(cm_ours.elo - cm_ctrl.elo, 2),
        "elapsed_sec": round(elapsed, 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.run_championship")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--n-battles", type=int, default=3)
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument("--max-team-size", type=int, default=4)
    p.add_argument("--max-pkm-moves", type=int, default=4)
    p.add_argument("--roster-size", type=int, default=30)
    p.add_argument("--n-moves", type=int, default=60)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    p.add_argument(
        "--control",
        choices=sorted(CONTROL_TEAMBUILDERS.keys()),
        default="random",
        help="Control's team-build policy. 'random' is the framework baseline; 'metausage', 'matchup_table', 'minimax' are head-to-head A/Bs vs prior defaults.",
    )
    p.add_argument(
        "--min-elo-delta",
        type=float,
        default=MIN_ELO_DELTA,
        help="Minimum ELO advantage required for the run to be considered a pass.",
    )
    args = p.parse_args(argv)

    result = run_championship(
        epochs=args.epochs,
        n_battles=args.n_battles,
        n_active=args.n_active,
        max_team_size=args.max_team_size,
        max_pkm_moves=args.max_pkm_moves,
        roster_size=args.roster_size,
        n_moves=args.n_moves,
        seed=args.seed,
        control=args.control,
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
