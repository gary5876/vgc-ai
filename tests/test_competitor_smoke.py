"""Smoke test — instantiate VgcAiCompetitor and check policies are wired."""

from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.selection import RandomSelectionPolicy
from vgc2.agent.teambuild import RandomTeamBuildPolicy

from vgc_ai.competitor import VgcAiCompetitor


def test_default_name() -> None:
    assert VgcAiCompetitor().name == "vgc-ai"


def test_custom_name() -> None:
    assert VgcAiCompetitor(name="alt").name == "alt"


def test_policies_wired_to_expected_baselines() -> None:
    c = VgcAiCompetitor()
    assert isinstance(c.battlepolicy, GreedyBattlePolicy)
    assert isinstance(c.selectionpolicy, RandomSelectionPolicy)
    assert isinstance(c.teambuildpolicy, RandomTeamBuildPolicy)
