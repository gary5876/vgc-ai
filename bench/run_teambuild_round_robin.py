"""4-way team-build round-robin via Championship ELO.

Registers one ``_ControlCompetitor`` per entry in
``run_championship.CONTROL_TEAMBUILDERS`` against the same shared battle +
selection policies. All four compete in a single ``Championship`` with
``Strategy.RANDOM_PAIRING``; the ELO each accumulates is the all-vs-all
ranking signal for the team-build layer.

Output: a JSON ranking + a Markdown table on stdout.
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
from vgc2.balance.meta import BasicMeta
from vgc2.competition import CompetitorManager
from vgc2.competition.ecosystem import Championship, Strategy, label_roster
from vgc2.util.generator import gen_move_set, gen_pkm_roster

from bench.run_championship import CONTROL_TEAMBUILDERS, _ControlCompetitor


class TeamBuildRanking(TypedDict):
    name: str
    elo: int


class TeamBuildRoundRobinResult(TypedDict):
    timestamp: str
    epochs: int
    n_battles: int
    n_active: int
    max_team_size: int
    max_pkm_moves: int
    roster_size: int
    n_moves: int
    ranking: list[TeamBuildRanking]
    elapsed_sec: float


def run_teambuild_round_robin(
    *,
    epochs: int,
    n_battles: int,
    n_active: int,
    max_team_size: int,
    max_pkm_moves: int,
    roster_size: int,
    n_moves: int,
    seed: int | None,
) -> TeamBuildRoundRobinResult:
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

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
    managers: list[tuple[str, CompetitorManager]] = []
    for name, factory in CONTROL_TEAMBUILDERS.items():
        comp = _ControlCompetitor(name=name, teambuild_policy=factory())
        cm = CompetitorManager(comp)
        championship.register(cm)
        managers.append((name, cm))

    t0 = time.perf_counter()
    championship.run()
    elapsed = time.perf_counter() - t0

    ranking: list[TeamBuildRanking] = sorted(
        [{"name": name, "elo": int(cm.elo)} for name, cm in managers],
        key=lambda r: -r["elo"],
    )

    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "epochs": epochs,
        "n_battles": n_battles,
        "n_active": n_active,
        "max_team_size": max_team_size,
        "max_pkm_moves": max_pkm_moves,
        "roster_size": roster_size,
        "n_moves": n_moves,
        "ranking": ranking,
        "elapsed_sec": round(elapsed, 2),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.run_teambuild_round_robin")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--n-battles", type=int, default=3)
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument("--max-team-size", type=int, default=4)
    p.add_argument("--max-pkm-moves", type=int, default=4)
    p.add_argument("--roster-size", type=int, default=30)
    p.add_argument("--n-moves", type=int, default=60)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output", type=str, default=None)
    args = p.parse_args(argv)

    result = run_teambuild_round_robin(
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

    print("\n### Team-build ELO ranking", file=sys.stderr)
    for entry in result["ranking"]:
        print(f"- {entry['name']}: {entry['elo']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
