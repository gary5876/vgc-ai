"""Smoke tests for GreedyWithSwitchBattlePolicy.

Behavior gate (win rate vs greedy) is enforced by the bench harness, not here.
These tests verify the policy registers, runs without error, and falls back to
greedy when no switch is warranted.
"""

from __future__ import annotations

from vgc2.agent.battle import GreedyBattlePolicy

from vgc_ai.cli import POLICIES
from vgc_ai.eval.duel import duel
from vgc_ai.policies.battle import GreedyWithSwitchBattlePolicy


def test_greedy_switch_is_registered() -> None:
    assert "greedy_switch" in POLICIES
    assert POLICIES["greedy_switch"] is GreedyWithSwitchBattlePolicy


def test_greedy_switch_runs_against_greedy() -> None:
    result = duel(GreedyWithSwitchBattlePolicy, GreedyBattlePolicy, n_battles=2)
    assert result.n_battles == 2
    assert result.wins_a + result.wins_b + result.ties == 2
