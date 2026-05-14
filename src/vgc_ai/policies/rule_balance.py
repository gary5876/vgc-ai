"""Rule balance policy — submission entry for the Rules Balance Track.

Track contract (``vgc2.agent.RuleBalancePolicy``):

    decision(team_pairs: list[tuple[Team, Team]],
             constraints: RuleConstraints) -> BattleRuleParam

The framework's ``RuleDesign.run()`` calls ``decision`` once per evaluator
to obtain a ``BattleRuleParam``, then plays a battery of fixed matches with
those params (``vgc2.competition.fixed_matches.FixedMatches``), categorizes
each winner action across all rollouts as ``switch`` / ``damage`` / ``effect``,
and scores by absolute deviation from the target distribution
``20% / 60% / 20%``:

    final_score = metric_weight (0.7) * deviation
                + time_weight   (0.3) * time_score(decision_seconds)

where ``time_score`` ranges over [1, 10] and *decreases* with time: 60s -> 10,
7200s -> 1. So a fast trivial submission already gets ~3 from the time term
(0.3 * 10) while a 10-minute search drops that to ~0.5. The time bonus
dominates a couple of percentage points of distribution-matching, so the
default policy is "return a fixed param in <1ms".

``IdentityRuleBalancePolicy`` is that submission-viable baseline — return
the framework's default ``BattleRuleParam()`` unchanged. The track is
evaluated against what those defaults induce on randomly generated
``FixedMatches`` team pairs; the deviation is whatever it is and we trade
the eval term for the time-bonus ceiling. Later policies (offline grid
search, gradient-style param tuning) can swap in as ``VgcAiRuleBalancePolicy``
when bench evidence justifies giving up the time bonus.
"""

from __future__ import annotations

from vgc2.agent import RuleBalancePolicy
from vgc2.balance.rules.constraints import RuleConstraints
from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.team import Team


class IdentityRuleBalancePolicy(RuleBalancePolicy):  # type: ignore[misc]
    """Trivial submission: return the framework default unchanged.

    Maximizes the time-score component (decision is <1us) and accepts
    whatever distribution the defaults induce on the eval team pairs.
    """

    def decision(
        self,
        team_pairs: list[tuple[Team, Team]],
        constraints: RuleConstraints,
    ) -> BattleRuleParam:
        return BattleRuleParam()


VgcAiRuleBalancePolicy = IdentityRuleBalancePolicy

__all__ = ["IdentityRuleBalancePolicy", "VgcAiRuleBalancePolicy"]
