"""vgc-ai command-line interface."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid

from vgc_ai.agents.random_agent import DEFAULT_DOUBLES_FORMAT, make_random_doubles_player
from vgc_ai.eval.runner import make_run_dir, run_battles


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vgc-ai", description="Run vgc-ai battles.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play = subparsers.add_parser("play", help="Run battles between two agents.")
    play.add_argument("--vs", choices=["random"], default="random", help="Opponent type.")
    play.add_argument("--n", type=int, default=10, help="Number of battles to run.")
    play.add_argument(
        "--format",
        default=DEFAULT_DOUBLES_FORMAT,
        help="Showdown battle format ID.",
    )
    return parser


async def _cmd_play(args: argparse.Namespace) -> int:
    suffix = uuid.uuid4().hex[:6]
    p1 = make_random_doubles_player(f"vgcai-p1-{suffix}", battle_format=args.format)
    p2 = make_random_doubles_player(f"vgcai-p2-{suffix}", battle_format=args.format)
    run_dir = make_run_dir()
    log_path = await run_battles(p1, p2, args.n, run_dir)
    print(f"Wrote {log_path}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "play":
        return asyncio.run(_cmd_play(args))
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
