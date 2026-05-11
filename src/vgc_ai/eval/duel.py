"""Run head-to-head battles between two BattlePolicies.

The framework's ``Match`` class evaluates full ``Competitor``s (with selection
and team-build policies). This helper isolates just the battle policy — same
randomly generated teams on both sides per battle, no selection or
team-build influence. Useful for ranking policies during development.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from vgc2.agent import BattlePolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam, State
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import label_teams, run_battle
from vgc2.util.generator import gen_team

PolicyFactory = Callable[[], BattlePolicy]


class _TimedBattlePolicy:
    """Decorator around a ``BattlePolicy`` that accumulates ``decision`` wall-clock time.

    ``run_battle`` calls ``decision`` once per turn per side, so the accumulated
    seconds divided by the call count is per-turn latency for that side. Uses
    composition (not subclassing) because ``BattlePolicy`` is untyped in vgc2.
    """

    def __init__(self, wrapped: BattlePolicy) -> None:
        self._wrapped = wrapped
        self.total_decision_sec: float = 0.0
        self.decision_count: int = 0

    def decision(self, state: State, opp_view: TeamView | None = None) -> Any:
        t0 = time.perf_counter()
        try:
            return self._wrapped.decision(state, opp_view)
        finally:
            self.total_decision_sec += time.perf_counter() - t0
            self.decision_count += 1

    def on_new_battle(self) -> None:
        self._wrapped.on_new_battle()


@dataclass(frozen=True)
class DuelResult:
    n_battles: int
    wins_a: int
    wins_b: int
    ties: int
    total_decision_sec_a: float = 0.0
    total_decision_sec_b: float = 0.0
    decisions_a: int = 0
    decisions_b: int = 0

    @property
    def win_rate_a(self) -> float:
        decided = self.wins_a + self.wins_b
        return self.wins_a / decided if decided else 0.0

    @property
    def avg_turn_ms_a(self) -> float:
        if not self.decisions_a:
            return 0.0
        return self.total_decision_sec_a * 1000.0 / self.decisions_a

    @property
    def avg_turn_ms_b(self) -> float:
        if not self.decisions_b:
            return 0.0
        return self.total_decision_sec_b * 1000.0 / self.decisions_b


def duel(
    policy_a: PolicyFactory,
    policy_b: PolicyFactory,
    n_battles: int,
    *,
    team_size: int = 4,
    n_active: int = 2,
    max_pkm_moves: int = 4,
    params: BattleRuleParam | None = None,
    fixed_team_seed: int | None = None,
) -> DuelResult:
    params = params or BattleRuleParam()
    # When fixed_team_seed is set, derive deterministic team/engine RNGs and seed
    # numpy's legacy global state so RandomBattlePolicy (which uses
    # numpy.random.choice) and any other legacy-RNG consumers are also
    # reproducible. Variance across policies is reduced because matchups
    # replay the same teams and same engine rolls.
    rng: np.random.Generator | None = None
    engine_rngs: (
        tuple[
            tuple[np.random.Generator, ...],
            tuple[np.random.Generator, ...],
        ]
        | None
    ) = None
    if fixed_team_seed is not None:
        rng = np.random.default_rng(fixed_team_seed)
        # vgc2 mixes three RNG sources: a Generator passed via constructor /
        # gen_team kwargs, numpy.random.choice (legacy global state, used by
        # RandomBattlePolicy), and random.sample/shuffle (stdlib, used by the
        # team generator and other policies). Seed all three for full
        # reproducibility.
        np.random.seed(fixed_team_seed)
        random.seed(fixed_team_seed)
        side_rngs = tuple(rng for _ in range(n_active))
        engine_rngs = (side_rngs, side_rngs)
    wins_a = 0
    wins_b = 0
    ties = 0
    total_decision_sec_a = 0.0
    total_decision_sec_b = 0.0
    decisions_a = 0
    decisions_b = 0
    for _ in range(n_battles):
        if rng is not None:
            team = (
                gen_team(team_size, max_pkm_moves, rng=rng),
                gen_team(team_size, max_pkm_moves, rng=rng),
            )
        else:
            team = gen_team(team_size, max_pkm_moves), gen_team(team_size, max_pkm_moves)
        label_teams(team)
        team_view = TeamView(team[0]), TeamView(team[1])
        state = State(get_battle_teams(team, n_active))
        state_view = (
            StateView(state, 0, team_view),
            StateView(state, 1, team_view),
        )
        if engine_rngs is not None:
            engine = BattleEngine(
                state,
                debug=False,
                acc_rng=engine_rngs,
                eff_rng=engine_rngs,
                sta_rng=engine_rngs,
            )
        else:
            engine = BattleEngine(state, debug=False)

        a_raw = policy_a()
        b_raw = policy_b()
        a_raw.set_params(params)
        b_raw.set_params(params)
        a = _TimedBattlePolicy(a_raw)
        b = _TimedBattlePolicy(b_raw)

        winner = run_battle(engine, (a, b), team_view, state_view, client=None)
        if winner == 0:
            wins_a += 1
        elif winner == 1:
            wins_b += 1
        else:
            ties += 1
        total_decision_sec_a += a.total_decision_sec
        total_decision_sec_b += b.total_decision_sec
        decisions_a += a.decision_count
        decisions_b += b.decision_count
    return DuelResult(
        n_battles=n_battles,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        total_decision_sec_a=total_decision_sec_a,
        total_decision_sec_b=total_decision_sec_b,
        decisions_a=decisions_a,
        decisions_b=decisions_b,
    )
