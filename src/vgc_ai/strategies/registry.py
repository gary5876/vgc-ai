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
    PairCoverageSelectionPolicy,
    SpeedPairCoverageSelectionPolicy,
    SpeedTierAwareSelectionPolicy,
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
        # Same LP-minimax team builder as the current default; the
        # differentiator is the selection layer, which scores *pairs*
        # of leads jointly instead of independent members. For each
        # candidate pair (a, b) and each opp k, pair offense is
        # ``max(off(a, k), off(b, k))`` (the better-positioned lead
        # targets k) and pair defense is ``max(off(k, a), off(k, b))``
        # (opp focus-fires whichever of our pair is more vulnerable).
        # The argmax across all C(n, 2) candidate pairs is the lead
        # pair; remaining team slots fill by singleton score. This is
        # the single-axis change vs the type-chart parent -- same
        # primitive, non-decomposable pair-level aggregation. Two
        # leads scored identically as singletons can compose into very
        # different pairs depending on which opp threats they cover
        # jointly: a 2x-vs-Fire + 2x-vs-Water pair beats a redundant
        # 2x-vs-Fire + 2x-vs-Fire pair against a Fire/Water opp duo,
        # which every existing policy gets wrong by ranking both Fire
        # leads at the same singleton score. Falls back to the
        # singleton parent at n_active=1 (singles regime), so the
        # worst case at the degenerate case is parity with the default.
        ChampionshipStrategy(
            name="minimax+pair_coverage_selection",
            team_build_policy=MinimaxTeamBuildPolicy,
            selection_policy=PairCoverageSelectionPolicy,
        ),
        # Same LP-minimax team builder as the current default; the
        # differentiator is the selection layer, which adds a per-opp
        # speed-tier initiative bonus to the type-chart score. Every
        # other selection policy throws away the most influential
        # single stat in doubles: a faster lead may KO a threat before
        # being hit, so an outspeed effectively reduces the incoming
        # threat that the static type-chart defense term treats as
        # inevitable. Reads ``Pokemon.stats[Stat.SPEED]`` -- the
        # post-EV/IV/nature value -- so the team-build's JOLLY/TIMID +
        # 252 SPE spread choices flow through into selection. Bonus
        # magnitude (0.25) tuned so the speed signal biases close
        # calls without overriding a real 2x super-effective matchup.
        # Strict refinement of MatchupAware: same structure when
        # matchups dominate, tiebroken by speed when they don't.
        ChampionshipStrategy(
            name="minimax+speed_tier_selection",
            team_build_policy=MinimaxTeamBuildPolicy,
            selection_policy=SpeedTierAwareSelectionPolicy,
        ),
        # Composes the two strongest single-axis selection improvements
        # already in the registry -- PairCoverage (joint pair coverage,
        # 70% pooled win rate vs current default) and SpeedTier (per-opp
        # initiative bonus, 73.3% pooled win rate vs current default) --
        # into one pair scorer. Each lead contributes independently to
        # the initiative term per opp, matching the doubles reality that
        # both leads attack and defend every turn (two outspeeds vs one
        # opp is genuinely twice as valuable as one outspeed). Strict
        # refinement of both parents: pair-coverage when all speeds tie,
        # speed-tier in the singles fallback. Same LP-minimax team
        # builder as the current default so the comparison isolates the
        # selection-layer composition. Ignores meta on purpose -- this
        # is the pure type-chart + speed-tier improvement, leaving
        # meta-weighting for a future stack.
        ChampionshipStrategy(
            name="minimax+speed_pair_coverage_selection",
            team_build_policy=MinimaxTeamBuildPolicy,
            selection_policy=SpeedPairCoverageSelectionPolicy,
        ),
    )
}

CHAMPIONSHIP_DEFAULT = "minimax+meta_threat_aware_selection"


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
