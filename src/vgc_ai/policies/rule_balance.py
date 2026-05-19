"""Rule-balance policy (Balance Track).

A ``RuleBalancePolicy`` proposes a ``BattleRuleParam`` (the engine's
ruleset: damage multipliers, STAB, status modifiers, etc.) to shape
battle dynamics — e.g. incentivising non-damaging moves.

``DefaultRuleBalancePolicy`` returns an unmodified default
``BattleRuleParam`` — i.e. "leave the rules alone." Legal submission,
trivial strategy; here to stand up the Balance Track shell.

``RuleConstraints`` is still an empty stub in vgc2 v2.1.1 (framework TODO
in ``vgc2.agent.rule_balance``). Once filled in, the policy will need to
honour real constraints.
"""

from __future__ import annotations

from vgc2.agent import RuleBalancePolicy, RuleConstraints
from vgc2.battle_engine.team import Team
from vgc2.util.param import BattleRuleParam


class DefaultRuleBalancePolicy(RuleBalancePolicy):  # type: ignore[misc]
    def decision(
        self,
        team_pairs: list[tuple[Team, Team]],
        constraints: RuleConstraints,
    ) -> BattleRuleParam:
        return BattleRuleParam()
