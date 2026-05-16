"""All-vs-all battle-policy round-robin.

Runs every unique unordered pair of policies in ``--policies`` via
``bench.run_once.run_once`` and emits:

- Per-pair JSON rows (one per pair) under ``bench/results/round-robin/``.
- A consolidated CSV ``bench/round_robin_battle.csv`` with all pair rows.
- A Markdown win-rate matrix on stdout (and optionally to ``--matrix-out``).

The driver complements ``run_continuous`` (which compares challengers vs a
fixed baseline) by producing a true all-vs-all ranking. Each cell ``M[a][b]``
is policy ``a``'s win rate vs policy ``b`` at the given ``--n``.

Slow policies (``tree``) are excluded by default; pass ``--include-slow`` to
include them. Pair sample size for slow-vs-* is overridden to ``--slow-n``
(default 20) so the run completes in finite time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from bench.run_once import BenchResult, run_once
from vgc_ai.cli import POLICIES

SLOW_POLICIES: frozenset[str] = frozenset({"tree"})
RESULTS_DIR = Path("bench/results/round-robin")
CSV_OUT = Path("bench/round_robin_battle.csv")
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


def _unique_pairs(names: list[str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            out.append((a, b))
    return out


def _n_for_pair(a: str, b: str, n_default: int, n_slow: int) -> int:
    if a in SLOW_POLICIES or b in SLOW_POLICIES:
        return n_slow
    return n_default


def _format_matrix(names: list[str], wr: dict[tuple[str, str], float]) -> str:
    header = "| vs | " + " | ".join(names) + " |"
    sep = "|" + "|".join(["---"] * (len(names) + 1)) + "|"
    lines = [header, sep]
    for a in names:
        row_cells: list[str] = [a]
        for b in names:
            if a == b:
                row_cells.append("—")
            else:
                row_cells.append(f"{wr[(a, b)]:.3f}")
        lines.append("| " + " | ".join(row_cells) + " |")
    return "\n".join(lines)


def _summarize(names: list[str], wr: dict[tuple[str, str], float]) -> list[tuple[str, float]]:
    """Mean win rate of each policy across its off-diagonal column."""
    out: list[tuple[str, float]] = []
    for a in names:
        opponents = [wr[(a, b)] for b in names if b != a]
        out.append((a, sum(opponents) / len(opponents) if opponents else 0.0))
    out.sort(key=lambda kv: -kv[1])
    return out


def run_round_robin(
    names: list[str],
    *,
    n_default: int,
    n_slow: int,
    team_size: int = 4,
    n_active: int = 2,
    fixed_team_seed: int | None = None,
) -> list[BenchResult]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[BenchResult] = []
    for a, b in _unique_pairs(names):
        n = _n_for_pair(a, b, n_default, n_slow)
        print(f"[round-robin] {a} vs {b} (n={n})", file=sys.stderr, flush=True)
        result = run_once(
            a,
            b,
            n,
            team_size=team_size,
            n_active=n_active,
            fixed_team_seed=fixed_team_seed,
        )
        rows.append(result)
        out = RESULTS_DIR / f"{a}__vs__{b}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return rows


def write_csv(rows: list[BenchResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CSV_COLUMNS})  # type: ignore[literal-required]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="bench.run_round_robin")
    p.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help=(
            "Names of policies to enter. Defaults to all in POLICIES minus "
            "slow ones; use --include-slow to keep them."
        ),
    )
    p.add_argument("--n", type=int, default=200, help="Battles per pair (non-slow).")
    p.add_argument(
        "--slow-n",
        type=int,
        default=20,
        help="Battles per pair when either side is in SLOW_POLICIES.",
    )
    p.add_argument("--include-slow", action="store_true", help="Include slow policies (e.g. tree).")
    p.add_argument("--team-size", type=int, default=4)
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument("--fixed-team-seed", type=int, default=None)
    p.add_argument("--csv-out", type=str, default=str(CSV_OUT))
    p.add_argument("--matrix-out", type=str, default=None)
    args = p.parse_args(argv)

    if args.policies is None:
        names = sorted(POLICIES)
        if not args.include_slow:
            names = [n for n in names if n not in SLOW_POLICIES]
    else:
        names = list(args.policies)
        for n in names:
            if n not in POLICIES:
                raise SystemExit(f"unknown policy {n!r} (known: {sorted(POLICIES)})")

    if len(names) < 2:
        raise SystemExit(f"need >=2 policies; got {names}")

    rows = run_round_robin(
        names,
        n_default=args.n,
        n_slow=args.slow_n,
        team_size=args.team_size,
        n_active=args.n_active,
        fixed_team_seed=args.fixed_team_seed,
    )

    # Build a directed win-rate map (a, b) -> wr_a; symmetric counterpart filled.
    wr: dict[tuple[str, str], float] = {}
    for r in rows:
        wr[(r["policy_a"], r["policy_b"])] = r["win_rate_a"]
        wr[(r["policy_b"], r["policy_a"])] = 1.0 - r["win_rate_a"] - (r["ties"] / r["n_battles"])

    write_csv(rows, Path(args.csv_out))

    matrix = _format_matrix(names, wr)
    print(matrix)
    if args.matrix_out:
        Path(args.matrix_out).write_text(matrix + "\n", encoding="utf-8")

    print("\n### Mean win rate (off-diagonal)")
    for name, mean in _summarize(names, wr):
        print(f"- {name}: {mean:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
