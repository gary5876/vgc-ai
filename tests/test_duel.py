"""Smoke test for the duel utility — uses random policies for speed."""

from vgc2.agent.battle import GreedyBattlePolicy, RandomBattlePolicy

from vgc_ai.eval.duel import duel


def test_duel_counts_sum_to_n_battles() -> None:
    result = duel(RandomBattlePolicy, RandomBattlePolicy, n_battles=2)
    assert result.n_battles == 2
    assert result.wins_a + result.wins_b + result.ties == 2


def test_duel_win_rate_a_is_proportion_of_decided_games() -> None:
    result = duel(RandomBattlePolicy, RandomBattlePolicy, n_battles=4)
    decided = result.wins_a + result.wins_b
    if decided:
        expected = result.wins_a / decided
        assert abs(result.win_rate_a - expected) < 1e-9
    else:
        assert result.win_rate_a == 0.0


def test_duel_reports_per_side_turn_timing() -> None:
    result = duel(RandomBattlePolicy, RandomBattlePolicy, n_battles=2)
    assert result.decisions_a > 0
    assert result.decisions_b > 0
    assert result.avg_turn_ms_a > 0.0
    assert result.avg_turn_ms_b > 0.0


def test_duel_is_deterministic_with_fixed_team_seed() -> None:
    r1 = duel(GreedyBattlePolicy, RandomBattlePolicy, n_battles=5, fixed_team_seed=42)
    r2 = duel(GreedyBattlePolicy, RandomBattlePolicy, n_battles=5, fixed_team_seed=42)
    r3 = duel(GreedyBattlePolicy, RandomBattlePolicy, n_battles=5, fixed_team_seed=42)
    assert (r1.wins_a, r1.wins_b, r1.ties) == (r2.wins_a, r2.wins_b, r2.ties)
    assert (r2.wins_a, r2.wins_b, r2.ties) == (r3.wins_a, r3.wins_b, r3.ties)
