"""Run a single benchmark duel between two named policies.

Output is a JSON object on stdout, optionally also written to a file. The
schema is the row schema of ``bench/leaderboard.csv`` (see ``run_continuous``).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from typing import TypedDict

from vgc_ai.cli import POLICIES
from vgc_ai.eval.duel import duel


class BenchResult(TypedDict):
    timestamp: str
    policy_a: str
    policy_b: str
    n_battles: int
    wins_a: int
    wins_b: int
    ties: int
    win_rate_a: float
    ci95_low: float
    ci95_high: float
    elapsed_sec: float
    avg_battle_ms: float
    avg_turn_ms_a: float
    avg_turn_ms_b: float


def wilson_ci_95(wins: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def run_once(
    policy_a: str,
    policy_b: str,
    n_battles: int,
    *,
    team_size: int = 4,
    n_active: int = 2,
    fixed_team_seed: int | None = None,
) -> BenchResult:
    if policy_a not in POLICIES:
        raise SystemExit(f"unknown policy_a: {policy_a!r} (known: {sorted(POLICIES)})")
    if policy_b not in POLICIES:
        raise SystemExit(f"unknown policy_b: {policy_b!r} (known: {sorted(POLICIES)})")

    t0 = time.perf_counter()
    result = duel(
        POLICIES[policy_a],
        POLICIES[policy_b],
        n_battles=n_battles,
        team_size=team_size,
        n_active=n_active,
        fixed_team_seed=fixed_team_seed,
    )
    elapsed = time.perf_counter() - t0

    decided = result.wins_a + result.wins_b
    ci_low, ci_high = wilson_ci_95(result.wins_a, decided) if decided else (0.0, 0.0)

    return {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "policy_a": policy_a,
        "policy_b": policy_b,
        "n_battles": result.n_battles,
        "wins_a": result.wins_a,
        "wins_b": result.wins_b,
        "ties": result.ties,
        "win_rate_a": round(result.win_rate_a, 4),
        "ci95_low": round(ci_low, 4),
        "ci95_high": round(ci_high, 4),
        "elapsed_sec": round(elapsed, 3),
        "avg_battle_ms": round(elapsed * 1000.0 / result.n_battles, 2),
        "avg_turn_ms_a": round(result.avg_turn_ms_a, 3),
        "avg_turn_ms_b": round(result.avg_turn_ms_b, 3),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.run_once")
    p.add_argument("--a", required=True, help=f"Policy A name (one of {sorted(POLICIES)})")
    p.add_argument("--b", required=True, help="Policy B name")
    p.add_argument("--n", type=int, default=100, help="Number of battles")
    p.add_argument("--team-size", type=int, default=4)
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument(
        "--fixed-team-seed",
        type=int,
        default=None,
        help=(
            "If set, replay the same teams + engine rolls across all N battles. "
            "Reduces variance when comparing different policies on identical battles."
        ),
    )
    p.add_argument("--output", type=str, default=None, help="Optional path to write the JSON to")
    args = p.parse_args(argv)

    result = run_once(
        args.a,
        args.b,
        args.n,
        team_size=args.team_size,
        n_active=args.n_active,
        fixed_team_seed=args.fixed_team_seed,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
