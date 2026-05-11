"""Running Elo ratings per battle policy.

After each call to :mod:`bench.run_continuous`, every matchup row updates the
running Elo for both policies (K=32, R0=1500) using the matchup's overall
score share — ``(wins + 0.5 * ties) / decided``. One Elo update per matchup
result, not per individual battle.

The history is appended to ``bench/elo.csv``; each row records a policy's
state at a given timestamp. The latest row per policy is the current rating.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import NamedTuple

ELO_K: float = 32.0
ELO_R0: float = 1500.0
ELO_CSV: Path = Path("bench/elo.csv")
CSV_COLUMNS: list[str] = ["timestamp", "policy", "games", "wins", "losses", "elo"]


class EloState(NamedTuple):
    games: int
    wins: int
    losses: int
    elo: float


def initial_state() -> EloState:
    return EloState(games=0, wins=0, losses=0, elo=ELO_R0)


def expected_score(rating_a: float, rating_b: float) -> float:
    return float(1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0)))


def update_pair(
    state_a: EloState,
    state_b: EloState,
    wins_a: int,
    wins_b: int,
    ties: int,
) -> tuple[EloState, EloState]:
    n = wins_a + wins_b + ties
    if n == 0:
        return state_a, state_b
    score_a = (wins_a + 0.5 * ties) / n
    expected = expected_score(state_a.elo, state_b.elo)
    delta = ELO_K * (score_a - expected)
    new_a = EloState(
        games=state_a.games + n,
        wins=state_a.wins + wins_a,
        losses=state_a.losses + wins_b,
        elo=state_a.elo + delta,
    )
    new_b = EloState(
        games=state_b.games + n,
        wins=state_b.wins + wins_b,
        losses=state_b.losses + wins_a,
        elo=state_b.elo - delta,
    )
    return new_a, new_b


def load_current_ratings(path: Path = ELO_CSV) -> dict[str, EloState]:
    if not path.exists():
        return {}
    states: dict[str, EloState] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            states[row["policy"]] = EloState(
                games=int(row["games"]),
                wins=int(row["wins"]),
                losses=int(row["losses"]),
                elo=float(row["elo"]),
            )
    return states


def append_rating_rows(
    policies: list[str],
    states: dict[str, EloState],
    timestamp: str,
    path: Path = ELO_CSV,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            w.writeheader()
        for policy in policies:
            s = states[policy]
            w.writerow(
                {
                    "timestamp": timestamp,
                    "policy": policy,
                    "games": s.games,
                    "wins": s.wins,
                    "losses": s.losses,
                    "elo": round(s.elo, 2),
                }
            )
