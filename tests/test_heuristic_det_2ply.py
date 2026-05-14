"""Smoke tests for the 2-ply deterministic-rollout heuristic policy."""

from __future__ import annotations

from vgc2.agent.battle import RandomBattlePolicy

from vgc_ai.cli import POLICIES
from vgc_ai.eval.duel import duel
from vgc_ai.policies.heuristic_det_2ply import HeuristicDet2plyBattlePolicy


def test_heuristic_det_2ply_is_registered() -> None:
    assert POLICIES["heuristic_det_2ply"] is HeuristicDet2plyBattlePolicy


def test_heuristic_det_2ply_runs_a_battle() -> None:
    result = duel(HeuristicDet2plyBattlePolicy, RandomBattlePolicy, n_battles=2, fixed_team_seed=42)
    assert result.n_battles == 2
    assert result.wins_a + result.wins_b + result.ties == 2


def test_heuristic_det_2ply_is_deterministic_with_fixed_team_seed() -> None:
    r1 = duel(HeuristicDet2plyBattlePolicy, RandomBattlePolicy, n_battles=4, fixed_team_seed=7)
    r2 = duel(HeuristicDet2plyBattlePolicy, RandomBattlePolicy, n_battles=4, fixed_team_seed=7)
    assert (r1.wins_a, r1.wins_b, r1.ties) == (r2.wins_a, r2.wins_b, r2.ties)
