"""Smoke tests for GreedyWithSwitchBBattlePolicy.

Behavior gate (win rate vs greedy) is enforced by the bench harness, not here.
These tests verify the policy registers and runs without error.
"""

from __future__ import annotations

from vgc2.agent.battle import GreedyBattlePolicy

from vgc_ai.cli import POLICIES
from vgc_ai.eval.duel import duel
from vgc_ai.policies.battle import GreedyWithSwitchBBattlePolicy


def test_greedy_switch_b_is_registered() -> None:
    assert "greedy_switch_b" in POLICIES
    assert POLICIES["greedy_switch_b"] is GreedyWithSwitchBBattlePolicy


def test_greedy_switch_b_runs_against_greedy() -> None:
    result = duel(GreedyWithSwitchBBattlePolicy, GreedyBattlePolicy, n_battles=2)
    assert result.n_battles == 2
    assert result.wins_a + result.wins_b + result.ties == 2
