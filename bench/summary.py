"""Aggregate the last N rows of ``bench/leaderboard.csv`` into a Markdown table.

Rows are policies; columns are ``vs <baseline>`` for each known baseline.
Cells show ``win_rate (n=decided_games)`` aggregated across the matched rows.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

LEADERBOARD = Path("bench/leaderboard.csv")
BASELINES: tuple[str, ...] = ("greedy", "random")
DEFAULT_LAST = 50

Totals = dict[tuple[str, str], tuple[int, int]]


def load_rows(path: Path, last: int) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if last < 1:
        return rows
    return rows[-last:]


def aggregate(rows: list[dict[str, str]]) -> tuple[list[str], Totals]:
    """Return ``(policies_seen, {(policy, baseline) -> (wins, decided)})``.

    Each leaderboard row contributes to two ``(policy, baseline)`` cells when
    ``policy_b`` (or ``policy_a``) is a known baseline: once for each side.
    """
    totals: Totals = defaultdict(lambda: (0, 0))
    policies: set[str] = set()
    for row in rows:
        a = row.get("policy_a", "")
        b = row.get("policy_b", "")
        if not a or not b:
            continue
        try:
            wa = int(row["wins_a"])
            wb = int(row["wins_b"])
        except (KeyError, ValueError):
            continue
        decided = wa + wb
        if decided == 0:
            continue
        policies.add(a)
        policies.add(b)
        for policy, baseline, wins in ((a, b, wa), (b, a, wb)):
            if baseline in BASELINES and policy != baseline:
                pw, pd = totals[(policy, baseline)]
                totals[(policy, baseline)] = (pw + wins, pd + decided)
    return sorted(policies, key=_policy_sort_key), dict(totals)


def _policy_sort_key(policy: str) -> tuple[int, str]:
    # Challengers first (group 0), then baselines (group 1), each alphabetical.
    return (1 if policy in BASELINES else 0, policy)


def render(policies: list[str], totals: Totals) -> str:
    header = ["policy"] + [f"vs {b}" for b in BASELINES]
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    for policy in policies:
        cells = [policy]
        for baseline in BASELINES:
            if policy == baseline:
                cells.append("—")
                continue
            wins, decided = totals.get((policy, baseline), (0, 0))
            if decided == 0:
                cells.append("—")
            else:
                rate = wins / decided
                cells.append(f"{rate:.2%} (n={decided})")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(path: Path, last: int) -> str:
    rows = load_rows(path, last)
    if not rows:
        return "no data yet"
    policies, totals = aggregate(rows)
    if not policies:
        return "no data yet"
    return render(policies, totals)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.summary")
    p.add_argument(
        "--last",
        type=int,
        default=DEFAULT_LAST,
        help="Number of most-recent leaderboard rows to aggregate (default: 50)",
    )
    p.add_argument(
        "--path",
        type=str,
        default=str(LEADERBOARD),
        help="Path to leaderboard.csv",
    )
    args = p.parse_args(argv)
    print(summarize(Path(args.path), args.last))
    return 0


if __name__ == "__main__":
    sys.exit(main())
