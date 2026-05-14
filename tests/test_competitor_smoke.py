"""Smoke test — instantiate VgcAiCompetitor and check policies are wired."""

from vgc_ai.competitor import VgcAiCompetitor
from vgc_ai.policies.heuristic_det import HeuristicDetBattlePolicy
from vgc_ai.policies.selection import MatchupAwareSelectionPolicy
from vgc_ai.policies.teambuild import MetaUsageTeamBuildPolicy


def test_default_name() -> None:
    assert VgcAiCompetitor().name == "vgc-ai"


def test_custom_name() -> None:
    assert VgcAiCompetitor(name="alt").name == "alt"


def test_policies_wired_to_expected_baselines() -> None:
    c = VgcAiCompetitor()
    assert isinstance(c.battlepolicy, HeuristicDetBattlePolicy)
    assert isinstance(c.selectionpolicy, MatchupAwareSelectionPolicy)
    assert isinstance(c.teambuildpolicy, MetaUsageTeamBuildPolicy)
