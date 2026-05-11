"""One round of pairwise benchmarking; appends to ``bench/leaderboard.csv``.

Wrap in a shell loop for continuous benching:

    while true; do uv run python -m bench.run_continuous; sleep 60; done

Each invocation re-imports ``vgc_ai.cli.POLICIES`` so newly-added policies
are picked up at the start of the next round.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from bench.run_once import BenchResult, run_once
from vgc_ai.cli import POLICIES

LEADERBOARD = Path("bench/leaderboard.csv")
CSV_COLUMNS: list[str] = [
    "timestamp",
    "policy_a",
    "policy_b",
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
]
SLOW_POLICIES: frozenset[str] = frozenset({"tree"})
BASELINE_ORDER: tuple[str, ...] = ("greedy", "random")


def migrate_schema() -> None:
    """Rewrite ``leaderboard.csv`` with the current header if it has older columns.

    Existing rows are preserved; new columns are written as empty strings for
    rows from older schema versions.
    """
    if not LEADERBOARD.exists():
        return
    with LEADERBOARD.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing = reader.fieldnames or []
        if existing == CSV_COLUMNS:
            return
        rows = list(reader)
    with LEADERBOARD.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def pairs_to_bench(include_slow: bool) -> list[tuple[str, str]]:
    available = [p for p in POLICIES if include_slow or p not in SLOW_POLICIES]
    baselines = [p for p in BASELINE_ORDER if p in available]
    challengers = [p for p in available if p not in baselines]

    pairs: list[tuple[str, str]] = [(c, b) for c in challengers for b in baselines]
    if "greedy" in baselines and "random" in baselines:
        pairs.append(("greedy", "random"))
    return pairs


def append_row(row: BenchResult) -> None:
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    new_file = not LEADERBOARD.exists()
    with LEADERBOARD.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: row[k] for k in CSV_COLUMNS})  # type: ignore[literal-required]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.run_continuous")
    p.add_argument("--n", type=int, default=50, help="Battles per matchup per round")
    p.add_argument(
        "--include-slow",
        action="store_true",
        help=f"Include slow policies: {sorted(SLOW_POLICIES)}",
    )
    args = p.parse_args(argv)

    pairs = pairs_to_bench(args.include_slow)
    if not pairs:
        print("No pairs to bench.", file=sys.stderr)
        return 1

    migrate_schema()

    print(f"[bench] round start: {len(pairs)} matchups, n={args.n}", flush=True)
    round_start = time.perf_counter()
    for a, b in pairs:
        t0 = time.perf_counter()
        try:
            result = run_once(a, b, args.n)
        except Exception as exc:
            print(f"[bench] {a} vs {b} FAILED: {exc}", file=sys.stderr, flush=True)
            continue
        append_row(result)
        print(
            f"[bench] {a:>10} vs {b:<10} "
            f"-> {result['wins_a']}-{result['wins_b']}-{result['ties']} "
            f"({result['win_rate_a']:.1%}, "
            f"{time.perf_counter() - t0:.1f}s)",
            flush=True,
        )
    print(f"[bench] round done in {time.perf_counter() - round_start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
