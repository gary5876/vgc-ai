"""Smoke test — instantiate VgcAiCompetitor and check policies are wired."""

from numpy.random import default_rng
from vgc2.competition import CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_rule_set, gen_team

from vgc_ai.competitor import VgcAiCompetitor
from vgc_ai.policies.heuristic_det import HeuristicDetBattlePolicy
from vgc_ai.policies.selection import MatchupAwareSelectionPolicy
from vgc_ai.policies.teambuild import MinimaxTeamBuildPolicy


def test_default_name() -> None:
    assert VgcAiCompetitor().name == "vgc-ai"


def test_custom_name() -> None:
    assert VgcAiCompetitor(name="alt").name == "alt"


def test_policies_wired_to_expected_baselines() -> None:
    c = VgcAiCompetitor()
    assert isinstance(c.battlepolicy, HeuristicDetBattlePolicy)
    assert isinstance(c.selectionpolicy, MatchupAwareSelectionPolicy)
    assert isinstance(c.teambuildpolicy, MinimaxTeamBuildPolicy)


def test_competitor_runs_under_random_rules() -> None:
    """2026 Battle Track shifts to general game-playing: rules generated
    per tournament via gen_rule_set. Confirm a single doubles battle
    completes cleanly under a perturbed BattleRuleParam.
    """
    params = gen_rule_set(rng=default_rng(0))
    a = VgcAiCompetitor(name="rules-A")
    b = VgcAiCompetitor(name="rules-B")
    cm = (CompetitorManager(a), CompetitorManager(b))
    match = Match(cm, n_active=2, n_battles=1, gen=gen_team, params=params)
    match.run()
    wins_a, wins_b = match.wins
    assert wins_a + wins_b >= 1
