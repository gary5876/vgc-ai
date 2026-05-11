"""Smoke test — instantiate VgcAiCompetitor and check policies are wired."""

from vgc2.agent.battle import RandomBattlePolicy
from vgc2.agent.selection import RandomSelectionPolicy
from vgc2.agent.teambuild import RandomTeamBuildPolicy

from vgc_ai.competitor import VgcAiCompetitor


def test_default_name() -> None:
    assert VgcAiCompetitor().name == "vgc-ai"


def test_custom_name() -> None:
    assert VgcAiCompetitor(name="alt").name == "alt"


def test_policies_default_to_random_baselines() -> None:
    c = VgcAiCompetitor()
    assert isinstance(c.battlepolicy, RandomBattlePolicy)
    assert isinstance(c.selectionpolicy, RandomSelectionPolicy)
    assert isinstance(c.teambuildpolicy, RandomTeamBuildPolicy)
