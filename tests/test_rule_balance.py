"""Smoke tests for the Rules Balance Track submission scaffold."""

from __future__ import annotations

from vgc2.balance.rules.constraints import RuleConstraints
from vgc2.battle_engine import BattleRuleParam

from vgc_ai.design_competitor import VgcAiDesignCompetitor
from vgc_ai.policies.rule_balance import (
    IdentityRuleBalancePolicy,
    VgcAiRuleBalancePolicy,
)


def test_default_name() -> None:
    assert VgcAiDesignCompetitor().name == "vgc-ai"


def test_custom_name() -> None:
    assert VgcAiDesignCompetitor(name="alt").name == "alt"


def test_alias_points_to_identity_default() -> None:
    assert VgcAiRuleBalancePolicy is IdentityRuleBalancePolicy


def test_design_competitor_wires_to_rule_balance_policy() -> None:
    c = VgcAiDesignCompetitor()
    assert isinstance(c.rulebalancepolicy, IdentityRuleBalancePolicy)


def test_decision_returns_battle_rule_param() -> None:
    policy = IdentityRuleBalancePolicy()
    constraints = RuleConstraints()
    params = policy.decision([], constraints)
    assert isinstance(params, BattleRuleParam)


def test_decision_returns_default_param() -> None:
    """Identity policy must return a fresh default BattleRuleParam, not a
    mutated singleton — callers may set rule attributes on the returned
    object and we don't want cross-call interference."""
    policy = IdentityRuleBalancePolicy()
    constraints = RuleConstraints()
    a = policy.decision([], constraints)
    b = policy.decision([], constraints)
    # Distinct instances
    assert a is not b
    # Same defaults
    assert a.STAB_MODIFIER == BattleRuleParam().STAB_MODIFIER
    assert a.PARALYSIS_MODIFIER == BattleRuleParam().PARALYSIS_MODIFIER
