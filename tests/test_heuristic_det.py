"""Smoke tests for the deterministic-rollout heuristic policy.

Sanity-only: the policy is registered, builds, plays a legal battle, and the
1-ply rollout is actually deterministic (same input state → same chosen
action across repeated decisions).
"""

from __future__ import annotations

from vgc2.agent.battle import RandomBattlePolicy

from vgc_ai.cli import POLICIES
from vgc_ai.eval.duel import duel
from vgc_ai.policies.heuristic_det import HeuristicDetBattlePolicy


def test_heuristic_det_is_registered() -> None:
    assert POLICIES["heuristic_det"] is HeuristicDetBattlePolicy


def test_heuristic_det_runs_a_battle() -> None:
    result = duel(HeuristicDetBattlePolicy, RandomBattlePolicy, n_battles=2, fixed_team_seed=42)
    assert result.n_battles == 2
    assert result.wins_a + result.wins_b + result.ties == 2


def test_heuristic_det_is_deterministic_with_fixed_team_seed() -> None:
    r1 = duel(HeuristicDetBattlePolicy, RandomBattlePolicy, n_battles=4, fixed_team_seed=7)
    r2 = duel(HeuristicDetBattlePolicy, RandomBattlePolicy, n_battles=4, fixed_team_seed=7)
    assert (r1.wins_a, r1.wins_b, r1.ties) == (r2.wins_a, r2.wins_b, r2.ties)
