"""Smoke tests for GreedyWithSwitchCBattlePolicy.

Behavior gate (win rate vs greedy) is enforced by the bench harness, not here.
These tests verify the policy registers and runs without error.
"""

from __future__ import annotations

from vgc2.agent.battle import GreedyBattlePolicy

from vgc_ai.cli import POLICIES
from vgc_ai.eval.duel import duel
from vgc_ai.policies.battle import GreedyWithSwitchCBattlePolicy


def test_greedy_switch_c_is_registered() -> None:
    assert "greedy_switch_c" in POLICIES
    assert POLICIES["greedy_switch_c"] is GreedyWithSwitchCBattlePolicy


def test_greedy_switch_c_runs_against_greedy() -> None:
    result = duel(GreedyWithSwitchCBattlePolicy, GreedyBattlePolicy, n_battles=2)
    assert result.n_battles == 2
    assert result.wins_a + result.wins_b + result.ties == 2
