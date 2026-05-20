"""Strategy registry — compound policy stacks per track.

Each ``*Strategy`` dataclass holds zero-arg factories that produce *fresh*
policy instances on every call. Several of our policies (e.g.
``MatchupAwareSelectionPolicy``, ``MatchupTableTeamBuildPolicy``,
``TabularMCBattlePolicy``) carry per-instance caches or learned state, so
sharing one instance across competitor entries would silently entangle
their decisions. Factories sidestep that — each ``Competitor`` /
``DesignCompetitor`` constructor calls the factory to get its own copy.

Adding a candidate is one tuple in the matching registry — no per-strategy
boilerplate. The reviewer loop appends here when promoting a new compound.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vgc2.agent import (
    BattlePolicy,
    MetaBalancePolicy,
    RuleBalancePolicy,
    SelectionPolicy,
    TeamBuildPolicy,
)
from vgc2.agent.battle import GreedyBattlePolicy, RandomBattlePolicy
from vgc2.agent.selection import RandomSelectionPolicy
from vgc2.agent.teambuild import RandomTeamBuildPolicy

from vgc_ai.policies.heuristic_det import HeuristicDetBattlePolicy
from vgc_ai.policies.library_teambuild import LibraryTeamBuildPolicy
from vgc_ai.policies.meta_balance import NoOpMetaBalancePolicy
from vgc_ai.policies.rule_balance import DefaultRuleBalancePolicy
from vgc_ai.policies.selection import (
    MatchupAwareSelectionPolicy,
    MetaThreatAwareSelectionPolicy,
    MetaWeightedSelectionPolicy,
)
from vgc_ai.policies.tabular_mc import TabularMCBattlePolicy
from vgc_ai.policies.teambuild import (
    MatchupTableTeamBuildPolicy,
    MetaCoverageTeamBuildPolicy,
    MetaUsageTeamBuildPolicy,
    MinimaxTeamBuildPolicy,
    PrincipledCoverageTeamBuildPolicy,
)

BattlePolicyFactory = Callable[[], BattlePolicy]
TeamBuildPolicyFactory = Callable[[], TeamBuildPolicy]
SelectionPolicyFactory = Callable[[], SelectionPolicy]
MetaBalancePolicyFactory = Callable[[], MetaBalancePolicy]
RuleBalancePolicyFactory = Callable[[], RuleBalancePolicy]


@dataclass(frozen=True)
class BattleStrategy:
    name: str
    battle_policy: BattlePolicyFactory


@dataclass(frozen=True)
class ChampionshipStrategy:
    name: str
    team_build_policy: TeamBuildPolicyFactory
    selection_policy: SelectionPolicyFactory


@dataclass(frozen=True)
class BalanceStrategy:
    name: str
    meta_balance_policy: MetaBalancePolicyFactory
    rule_balance_policy: RuleBalancePolicyFactory


BATTLE_STRATEGIES: dict[str, BattleStrategy] = {
    s.name: s
    for s in (
        BattleStrategy(name="heuristic_det", battle_policy=HeuristicDetBattlePolicy),
        BattleStrategy(name="greedy", battle_policy=GreedyBattlePolicy),
        BattleStrategy(name="random", battle_policy=RandomBattlePolicy),
        # tabular_mc is currently below random vs greedy (~0.04, see TASKS.md
        # blocked entries); kept in the registry as a known-weak baseline so
        # the next canonicalization fix lands as a replace-by-name, not a
        # new entry.
        BattleStrategy(name="tabular_mc", battle_policy=TabularMCBattlePolicy),
    )
}

BATTLE_DEFAULT = "heuristic_det"


CHAMPIONSHIP_STRATEGIES: dict[str, ChampionshipStrategy] = {
    s.name: s
    for s in (
        ChampionshipStrategy(
            name="minimax+matchup_aware",
            team_build_policy=MinimaxTeamBuildPolicy,
            selection_policy=MatchupAwareSelectionPolicy,
        ),
        ChampionshipStrategy(
            name="matchup_table+matchup_aware",
            team_build_policy=MatchupTableTeamBuildPolicy,
            selection_policy=MatchupAwareSelectionPolicy,
        ),
        ChampionshipStrategy(
            name="metausage+matchup_aware",
            team_build_policy=MetaUsageTeamBuildPolicy,
            selection_policy=MatchupAwareSelectionPolicy,
        ),
        ChampionshipStrategy(
            name="random+random",
            team_build_policy=RandomTeamBuildPolicy,
            selection_policy=RandomSelectionPolicy,
        ),
        # Same minimax team builder as the current default; the differentiator
        # is the selection layer, which now consumes meta.usage_rate_pokemon
        # via the Championship Track set_meta hook (proposed task
        # policy-selection-set-meta-prior-usage-weighting). Strict
        # generalisation of MatchupAwareSelectionPolicy — falls back to the
        # type-chart parent when the meta is empty (epoch 0, or no usable
        # usage data), so the worst case is parity with the default.
        ChampionshipStrategy(
            name="minimax+meta_weighted_selection",
            team_build_policy=MinimaxTeamBuildPolicy,
            selection_policy=MetaWeightedSelectionPolicy,
        ),
        # Same LP-minimax team builder as the current default; the
        # differentiator is the selection layer, which composes two
        # single-axis improvements: usage-weighted offense (the
        # meta-weighted insight -- high-usage opp species drive the
        # offense signal more than rare ones) AND max-threat defense
        # (a single 2x super-effective opp one-shots the lead in
        # doubles, so worst-case survival dominates average matchup).
        # Falls back to (uniform_mean_offense - max_defense) when the
        # meta has no usable data (epoch 0 of every championship), so
        # the worst case at epoch 0 is the uniform threat-aware
        # baseline.
        ChampionshipStrategy(
            name="minimax+meta_threat_aware_selection",
            team_build_policy=MinimaxTeamBuildPolicy,
            selection_policy=MetaThreatAwareSelectionPolicy,
        ),
        # Meta-usage-weighted greedy coverage team builder. With uniform
        # weights (meta is None / epoch 0 / no usage signal) the first
        # pick matches MatchupTableTeamBuildPolicy (both argmax the row
        # mean); later picks may differ because MetaCoverage doesn't mask
        # already-picked species from the score (the meta is the opponent
        # distribution, not "opponents we haven't picked"). When the meta
        # has signal, picks shift toward counters of high-usage opponents.
        # Strategic insight: real opponents cluster around the empirical
        # meta, not adversarially as MinimaxTeamBuildPolicy assumes.
        ChampionshipStrategy(
            name="meta_coverage+matchup_aware",
            team_build_policy=MetaCoverageTeamBuildPolicy,
            selection_policy=MatchupAwareSelectionPolicy,
        ),
        # Pre-computed library lookup with MetaCoverage fallback.
        # data/team_library.json holds hand-curated teams keyed by
        # roster fingerprint (Claude offline, 15-principle checklist).
        # On a roster miss, falls back to MetaCoverageTeamBuildPolicy,
        # which itself falls back to MatchupTable behaviour at epoch 0
        # — so the worst case is parity with MetaCoverage above.
        ChampionshipStrategy(
            name="library+matchup_aware",
            team_build_policy=LibraryTeamBuildPolicy,
            selection_policy=MatchupAwareSelectionPolicy,
        ),
        # MetaCoverage + encoded subset of the 15-principle doubles
        # checklist as additive bonuses (speed-control redundancy,
        # status coverage, phys/spec split, type diversity, glass-cannon
        # penalty). Same matchup-table cache, same meta weighting; bonuses
        # bias close calls without overriding strong matchup scores.
        # Sibling of meta_coverage+matchup_aware so the bench A/Bs them
        # against the default; principles that vgc2 can't express
        # (Fake Out, redirection, abilities, spread blockers) are
        # intentionally omitted.
        ChampionshipStrategy(
            name="principled_coverage+matchup_aware",
            team_build_policy=PrincipledCoverageTeamBuildPolicy,
            selection_policy=MatchupAwareSelectionPolicy,
        ),
    )
}

CHAMPIONSHIP_DEFAULT = "minimax+matchup_aware"


BALANCE_STRATEGIES: dict[str, BalanceStrategy] = {
    s.name: s
    for s in (
        BalanceStrategy(
            name="noop+default",
            meta_balance_policy=NoOpMetaBalancePolicy,
            rule_balance_policy=DefaultRuleBalancePolicy,
        ),
    )
}

BALANCE_DEFAULT = "noop+default"


__all__ = [
    "BALANCE_DEFAULT",
    "BALANCE_STRATEGIES",
    "BATTLE_DEFAULT",
    "BATTLE_STRATEGIES",
    "CHAMPIONSHIP_DEFAULT",
    "CHAMPIONSHIP_STRATEGIES",
    "BalanceStrategy",
    "BattleStrategy",
    "ChampionshipStrategy",
]
