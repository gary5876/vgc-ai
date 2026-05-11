"""Tests for ``bench.elo``: Elo math + CSV round-trip."""

from __future__ import annotations

from pathlib import Path

from bench.elo import (
    ELO_K,
    ELO_R0,
    EloState,
    append_rating_rows,
    expected_score,
    initial_state,
    load_current_ratings,
    update_pair,
)


def test_expected_score_equal_ratings_is_half() -> None:
    assert expected_score(1500.0, 1500.0) == 0.5


def test_expected_score_symmetric() -> None:
    e_a = expected_score(1600.0, 1500.0)
    e_b = expected_score(1500.0, 1600.0)
    assert abs(e_a + e_b - 1.0) < 1e-12


def test_update_pair_two_rounds_matches_hand_computed() -> None:
    a = initial_state()
    b = initial_state()

    # Round 1: a wins 18, b wins 2, no ties — score_a = 0.9, expected = 0.5
    a1, b1 = update_pair(a, b, wins_a=18, wins_b=2, ties=0)
    delta_1 = ELO_K * (0.9 - 0.5)  # 12.8
    assert abs(a1.elo - (ELO_R0 + delta_1)) < 1e-9
    assert abs(b1.elo - (ELO_R0 - delta_1)) < 1e-9
    assert a1.games == 20 and a1.wins == 18 and a1.losses == 2
    assert b1.games == 20 and b1.wins == 2 and b1.losses == 18

    # Round 2: same matchup, ratings have diverged so expected changes
    a2, b2 = update_pair(a1, b1, wins_a=18, wins_b=2, ties=0)
    expected_2 = 1.0 / (1.0 + 10.0 ** ((b1.elo - a1.elo) / 400.0))
    delta_2 = ELO_K * (0.9 - expected_2)
    assert abs(a2.elo - (a1.elo + delta_2)) < 1e-9
    assert abs(b2.elo - (b1.elo - delta_2)) < 1e-9
    # Net Elo is conserved (zero-sum)
    assert abs((a2.elo + b2.elo) - 2 * ELO_R0) < 1e-9
    assert a2.games == 40 and b2.games == 40


def test_update_pair_with_ties_counts_half() -> None:
    a = initial_state()
    b = initial_state()
    a1, b1 = update_pair(a, b, wins_a=4, wins_b=4, ties=2)
    # score_a = (4 + 0.5*2)/10 = 0.5; expected = 0.5 → no change
    assert a1.elo == ELO_R0
    assert b1.elo == ELO_R0
    assert a1.games == 10 and a1.wins == 4 and a1.losses == 4


def test_update_pair_zero_games_is_noop() -> None:
    a = initial_state()
    b = initial_state()
    a1, b1 = update_pair(a, b, wins_a=0, wins_b=0, ties=0)
    assert a1 == a and b1 == b


def test_50_rounds_at_95pct_exceeds_200_elo_gap() -> None:
    """Acceptance: 50 rounds of one-sided benching gives a > 200 Elo gap."""
    a = initial_state()
    b = initial_state()
    for _ in range(50):
        a, b = update_pair(a, b, wins_a=19, wins_b=1, ties=0)
    assert (a.elo - b.elo) > 200.0


def test_load_and_append_roundtrip(tmp_path: Path) -> None:
    csv_path = tmp_path / "elo.csv"
    assert load_current_ratings(csv_path) == {}

    states = {
        "greedy": EloState(games=20, wins=18, losses=2, elo=1512.8),
        "random": EloState(games=20, wins=2, losses=18, elo=1487.2),
    }
    append_rating_rows(
        ["greedy", "random"], states, timestamp="2026-05-11T00:00:00+00:00", path=csv_path
    )

    loaded = load_current_ratings(csv_path)
    assert set(loaded) == {"greedy", "random"}
    assert loaded["greedy"].elo == 1512.8
    assert loaded["greedy"].games == 20
    assert loaded["random"].losses == 18


def test_load_returns_latest_per_policy(tmp_path: Path) -> None:
    csv_path = tmp_path / "elo.csv"
    s1 = {"greedy": EloState(20, 18, 2, 1512.8)}
    append_rating_rows(["greedy"], s1, timestamp="2026-05-11T00:00:00+00:00", path=csv_path)
    s2 = {"greedy": EloState(40, 36, 4, 1525.5)}
    append_rating_rows(["greedy"], s2, timestamp="2026-05-11T01:00:00+00:00", path=csv_path)

    loaded = load_current_ratings(csv_path)
    assert loaded["greedy"].elo == 1525.5
    assert loaded["greedy"].games == 40
