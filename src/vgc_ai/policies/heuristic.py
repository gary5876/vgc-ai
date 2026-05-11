"""One-ply forward-search policy guided by ``vgc_ai.eval.heuristic.evaluate``.

For each of our possible action combos, simulate one turn forward (assuming
the opponent plays Greedy), score the resulting state with ``evaluate``, and
pick the combo with the highest score. Stochastic outcomes are sampled once
per action — no RNG branching — so the per-turn cost stays bounded.
"""

from __future__ import annotations

from vgc2.agent import BattlePolicy
from vgc2.agent.battle import GreedyBattlePolicy, get_actions
from vgc2.battle_engine import BattleCommand, BattleRuleParam, State
from vgc2.battle_engine.view import TeamView
from vgc2.util.forward import copy_state, forward

from vgc_ai.eval.heuristic import evaluate


class HeuristicBattlePolicy(BattlePolicy):  # type: ignore[misc]
    """Picks the action combo maximizing ``evaluate`` one ply ahead.

    The opponent is approximated by ``GreedyBattlePolicy`` (same approach as
    ``TreeSearchBattlePolicy``). Falls back to a greedy action if no legal
    combos are enumerable (e.g. all moves disabled).
    """

    def __init__(self) -> None:
        self._opp = GreedyBattlePolicy()
        self._fallback = GreedyBattlePolicy()

    def set_params(self, params: BattleRuleParam) -> None:
        super().set_params(params)
        self._opp.set_params(params)
        self._fallback.set_params(params)

    def decision(
        self,
        state: State,
        opp_view: TeamView | None = None,
    ) -> list[BattleCommand]:
        actions = get_actions((state.sides[0].team, state.sides[1].team))
        if not actions:
            fallback: list[BattleCommand] = self._fallback.decision(state, opp_view)
            return fallback

        # The opponent's greedy decision depends only on the current state
        # (one-ply lookahead), so compute it once and reuse for every action.
        opp_state = State((state.sides[1], state.sides[0]))
        opp_action: list[BattleCommand] = self._opp.decision(opp_state, None)

        best_value = float("-inf")
        best_action: list[BattleCommand] | None = None
        for action in actions:
            simulated = copy_state(state)
            forward(simulated, (list(action), opp_action), self.params)
            value = evaluate(simulated, self.params)
            if value > best_value:
                best_value = value
                best_action = list(action)

        if best_action is not None:
            return best_action
        fallback = self._fallback.decision(state, opp_view)
        return fallback
