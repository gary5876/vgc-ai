"""Design competitor — submission entry for the Rules Balance Track.

The Battle / Championship tracks use ``vgc2.competition.Competitor``;
the Rules Balance and Meta Balance tracks use ``DesignCompetitor`` instead.
Different ABC, different property names, different evaluator (``RuleDesign``
/ ``MetaDesign`` rather than ``Match`` / ``Championship``). This module
exposes ``VgcAiDesignCompetitor`` for the rules-balance entry; meta-balance
is a future track if we choose to pursue it.
"""

from __future__ import annotations

from vgc2.agent import RuleBalancePolicy
from vgc2.competition import DesignCompetitor

from vgc_ai.policies.rule_balance import VgcAiRuleBalancePolicy


class VgcAiDesignCompetitor(DesignCompetitor):  # type: ignore[misc]
    """vgc-ai's Rules Balance Track submission shell."""

    def __init__(self, name: str = "vgc-ai") -> None:
        self._name = name
        self._rule_balance: RuleBalancePolicy = VgcAiRuleBalancePolicy()

    @property
    def rulebalancepolicy(self) -> RuleBalancePolicy | None:
        return self._rule_balance

    @property
    def name(self) -> str:
        return self._name


__all__ = ["VgcAiDesignCompetitor"]
