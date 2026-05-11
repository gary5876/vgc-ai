"""Run head-to-head battles between two BattlePolicies.

The framework's ``Match`` class evaluates full ``Competitor``s (with selection
and team-build policies). This helper isolates just the battle policy — same
randomly generated teams on both sides per battle, no selection or
team-build influence. Useful for ranking policies during development.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vgc2.agent import BattlePolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam, State
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import label_teams, run_battle
from vgc2.util.generator import gen_team

PolicyFactory = Callable[[], BattlePolicy]


@dataclass(frozen=True)
class DuelResult:
    n_battles: int
    wins_a: int
    wins_b: int
    ties: int

    @property
    def win_rate_a(self) -> float:
        decided = self.wins_a + self.wins_b
        return self.wins_a / decided if decided else 0.0


def duel(
    policy_a: PolicyFactory,
    policy_b: PolicyFactory,
    n_battles: int,
    *,
    team_size: int = 4,
    n_active: int = 2,
    max_pkm_moves: int = 4,
    params: BattleRuleParam | None = None,
) -> DuelResult:
    params = params or BattleRuleParam()
    wins_a = 0
    wins_b = 0
    ties = 0
    for _ in range(n_battles):
        team = gen_team(team_size, max_pkm_moves), gen_team(team_size, max_pkm_moves)
        label_teams(team)
        team_view = TeamView(team[0]), TeamView(team[1])
        state = State(get_battle_teams(team, n_active))
        state_view = (
            StateView(state, 0, team_view),
            StateView(state, 1, team_view),
        )
        engine = BattleEngine(state, debug=False)

        a = policy_a()
        b = policy_b()
        a.set_params(params)
        b.set_params(params)

        winner = run_battle(engine, (a, b), team_view, state_view, client=None)
        if winner == 0:
            wins_a += 1
        elif winner == 1:
            wins_b += 1
        else:
            ties += 1
    return DuelResult(n_battles=n_battles, wins_a=wins_a, wins_b=wins_b, ties=ties)
