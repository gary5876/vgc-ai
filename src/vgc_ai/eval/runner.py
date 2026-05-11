"""Run N battles between two players and log per-battle summaries to JSONL.

Per-turn telemetry is deliberately out of scope for milestone 1 — it requires
a custom Player subclass that captures move-by-move data, which is only
meaningful once we have a non-random policy.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poke_env.player import Player


def make_run_dir(root: Path | str = "runs") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = Path(root) / timestamp
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def _summarize(player: Player) -> list[dict[str, Any]]:
    return [
        {
            "battle_tag": tag,
            "won": battle.won,
            "lost": battle.lost,
            "finished": battle.finished,
            "turn": battle.turn,
        }
        for tag, battle in player.battles.items()
    ]


async def run_battles(
    player: Player,
    opponent: Player,
    n_battles: int,
    run_dir: Path,
) -> Path:
    await player.battle_against(opponent, n_battles=n_battles)
    log_path = run_dir / "episodes.jsonl"
    write_jsonl(log_path, _summarize(player))
    return log_path
