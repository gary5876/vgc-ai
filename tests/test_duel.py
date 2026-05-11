"""Smoke test for the duel utility — uses random policies for speed."""

from vgc2.agent.battle import RandomBattlePolicy

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
