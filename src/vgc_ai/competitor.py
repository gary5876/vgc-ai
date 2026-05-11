"""Competitor entry for the IEEE VGC AI Competition.

The three policies start as the framework's random baselines. They will be
replaced one at a time as later milestones land (MCTS for battle, GA / LP
for team building, etc.). Only this file is the public entry point — the
``policies/`` module owns each policy's implementation.
"""

from __future__ import annotations

from vgc2.agent import BattlePolicy, SelectionPolicy, TeamBuildPolicy
from vgc2.competition import Competitor

from vgc_ai.policies.battle import VgcAiBattlePolicy
from vgc_ai.policies.selection import VgcAiSelectionPolicy
from vgc_ai.policies.teambuild import VgcAiTeamBuildPolicy


class VgcAiCompetitor(Competitor):  # type: ignore[misc]  # vgc2 is untyped; Competitor resolves as Any under --strict
    def __init__(self, name: str = "vgc-ai") -> None:
        self._name = name
        self._battle_policy: BattlePolicy = VgcAiBattlePolicy()
        self._selection_policy: SelectionPolicy = VgcAiSelectionPolicy()
        self._team_build_policy: TeamBuildPolicy = VgcAiTeamBuildPolicy()

    @property
    def battlepolicy(self) -> BattlePolicy | None:
        return self._battle_policy

    @property
    def selectionpolicy(self) -> SelectionPolicy | None:
        return self._selection_policy

    @property
    def teambuildpolicy(self) -> TeamBuildPolicy | None:
        return self._team_build_policy

    @property
    def name(self) -> str:
        return self._name
