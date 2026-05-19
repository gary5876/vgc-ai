"""Smoke test — instantiate VgcAiDesignCompetitor and exercise its policies."""

from vgc2.util.param import BattleRuleParam

from vgc_ai.design_competitor import VgcAiDesignCompetitor
from vgc_ai.policies.meta_balance import NoOpMetaBalancePolicy
from vgc_ai.policies.rule_balance import DefaultRuleBalancePolicy


def test_default_name() -> None:
    assert VgcAiDesignCompetitor().name == "vgc-ai-design"


def test_custom_name() -> None:
    assert VgcAiDesignCompetitor(name="alt-design").name == "alt-design"


def test_policies_wired_to_expected_shells() -> None:
    c = VgcAiDesignCompetitor()
    assert isinstance(c.metabalancepolicy, NoOpMetaBalancePolicy)
    assert isinstance(c.rulebalancepolicy, DefaultRuleBalancePolicy)


def test_meta_balance_decision_returns_no_changes() -> None:
    policy = NoOpMetaBalancePolicy()
    move_set, roster = policy.decision(move_set=[], roster=[], constraints=None)  # type: ignore[arg-type]  # MetaConstraints is empty stub
    assert move_set == []
    assert roster == []


def test_rule_balance_decision_returns_default_params() -> None:
    policy = DefaultRuleBalancePolicy()
    params = policy.decision(team_pairs=[], constraints=None)  # type: ignore[arg-type]  # RuleConstraints is empty stub
    assert isinstance(params, BattleRuleParam)
