"""Smoke tests for GreedyWithSwitchDBattlePolicy.

Behavior gate (win rate vs greedy) is enforced by the bench harness, not here.
These tests verify the policy registers and runs without error.
"""

from __future__ import annotations

from vgc2.agent.battle import GreedyBattlePolicy

from vgc_ai.cli import POLICIES
from vgc_ai.eval.duel import duel
from vgc_ai.policies.battle import GreedyWithSwitchDBattlePolicy


def test_greedy_switch_d_is_registered() -> None:
    assert "greedy_switch_d" in POLICIES
    assert POLICIES["greedy_switch_d"] is GreedyWithSwitchDBattlePolicy


def test_greedy_switch_d_runs_against_greedy() -> None:
    result = duel(GreedyWithSwitchDBattlePolicy, GreedyBattlePolicy, n_battles=2)
    assert result.n_battles == 2
    assert result.wins_a + result.wins_b + result.ties == 2
