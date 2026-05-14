"""vgc-ai command-line interface."""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial

from vgc2.agent.battle import GreedyBattlePolicy, RandomBattlePolicy, TreeSearchBattlePolicy

from vgc_ai.eval.duel import duel
from vgc_ai.policies.heuristic_det import HeuristicDetBattlePolicy
from vgc_ai.policies.tabular_mc import TabularMCBattlePolicy

# tabular_mc auto-loads ``models/tabular_mc.json`` if present (the file is
# gitignored — produced by ``scripts/train_tabular_mc.py``); when absent, the
# constructor still returns a zero-knowledge policy that plays uniformly at
# random over legal joint actions, so unit tests and bench rounds without a
# trained checkpoint continue to work.
POLICIES = {
    "random": RandomBattlePolicy,
    "greedy": GreedyBattlePolicy,
    "tree": TreeSearchBattlePolicy,
    "tabular_mc": partial(TabularMCBattlePolicy, model_path="models/tabular_mc.json"),
    "heuristic_det": HeuristicDetBattlePolicy,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vgc-ai")
    sub = parser.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval", help="Duel two battle policies and report the win rate.")
    eval_p.add_argument("--a", choices=POLICIES.keys(), default="tree")
    eval_p.add_argument("--b", choices=POLICIES.keys(), default="random")
    eval_p.add_argument("--n", type=int, default=20, help="Number of battles.")
    eval_p.add_argument("--team-size", type=int, default=4)
    eval_p.add_argument("--n-active", type=int, default=2)
    eval_p.add_argument("--max-pkm-moves", type=int, default=4)
    return parser


def cmd_eval(args: argparse.Namespace) -> int:
    factory_a = POLICIES[args.a]
    factory_b = POLICIES[args.b]
    t0 = time.perf_counter()
    result = duel(
        factory_a,
        factory_b,
        n_battles=args.n,
        team_size=args.team_size,
        n_active=args.n_active,
        max_pkm_moves=args.max_pkm_moves,
    )
    elapsed = time.perf_counter() - t0
    print(
        f"{args.a} vs {args.b}: "
        f"{result.wins_a}-{result.wins_b}-{result.ties} "
        f"(a win rate: {result.win_rate_a:.1%}) "
        f"in {elapsed:.1f}s"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "eval":
        return cmd_eval(args)
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
