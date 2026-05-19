"""Balance Track entry — parallel to ``VgcAiCompetitor`` but for the
Meta + Rule Balance Track introduced in the 2026 4th edition.

The Balance Track uses a different ABC (``vgc2.competition.DesignCompetitor``)
exposing ``metabalancepolicy`` + ``rulebalancepolicy`` instead of the
battle/selection/teambuild triplet. The shell ships with no-op policies
so we can submit, exercise the wiring, and iterate from there.
"""

from __future__ import annotations

from vgc2.agent import MetaBalancePolicy, RuleBalancePolicy
from vgc2.competition import DesignCompetitor

from vgc_ai.policies.meta_balance import NoOpMetaBalancePolicy
from vgc_ai.policies.rule_balance import DefaultRuleBalancePolicy


class VgcAiDesignCompetitor(DesignCompetitor):  # type: ignore[misc]  # vgc2 untyped under --strict
    def __init__(self, name: str = "vgc-ai-design") -> None:
        self._name = name
        self._meta_balance_policy: MetaBalancePolicy = NoOpMetaBalancePolicy()
        self._rule_balance_policy: RuleBalancePolicy = DefaultRuleBalancePolicy()

    @property
    def metabalancepolicy(self) -> MetaBalancePolicy | None:
        return self._meta_balance_policy

    @property
    def rulebalancepolicy(self) -> RuleBalancePolicy | None:
        return self._rule_balance_policy

    @property
    def name(self) -> str:
        return self._name
