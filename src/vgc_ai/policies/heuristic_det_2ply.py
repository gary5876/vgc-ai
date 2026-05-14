"""Two-ply forward-search policy with **deterministic** rollouts.

Deepens :class:`vgc_ai.policies.heuristic_det.HeuristicDetBattlePolicy` by one
additional ply. The structure is "opponent-best-response then re-evaluate":

- ply 1: for each of my joint actions ``a``, apply ``(a, opp_greedy(state))``
  via the deterministic forward sim and record the 1-ply heuristic value of
  the resulting state ``s1``.
- ply 2: for the top ``TOP_K_PLY1`` ply-1 actions by value, look one more
  turn ahead — at ``s1``, search over my joint actions ``a'`` (with the
  opponent again playing greedy) and take the max heuristic value. This
  becomes the 2-ply value of the original ``a``.
- argmax over the ply-1 candidates of the 2-ply value.

The opponent model is :class:`GreedyBattlePolicy` (matches the bench
baseline; see CLAUDE.md note that this is the "realistic, matches bench
baseline" choice over a paranoid same-heuristic opponent).

**Why pruning at ply 1 is required**: the parent task observed ~100 joint
actions per state on doubles startup positions. Full minimax at depth 2 is
~K² = 10000 forward+eval calls per turn; at the parent's measured 0.25 ms
per forward+eval that's ~2500 ms/turn — far over the 500 ms/turn budget.
Pruning to ``TOP_K_PLY1`` cuts the worst case to ~K + K·TOP_K_PLY1 calls,
which at K=100, TOP_K_PLY1=10 gives ~1100 calls ≈ 275 ms — under budget
with margin.
"""

from __future__ import annotations

from vgc2.agent import BattlePolicy
from vgc2.agent.battle import GreedyBattlePolicy, get_actions
from vgc2.battle_engine import BattleCommand, BattleRuleParam, State
from vgc2.battle_engine.view import TeamView
from vgc2.util.forward import copy_state, forward
from vgc2.util.rng import ONE_RNG, ZERO_RNG

from vgc_ai.eval.heuristic import evaluate

_ACC_RNG = ((ZERO_RNG, ZERO_RNG), (ZERO_RNG, ZERO_RNG))
_EFF_RNG = ((ONE_RNG, ONE_RNG), (ONE_RNG, ONE_RNG))
_STA_RNG = ((ONE_RNG, ONE_RNG), (ONE_RNG, ONE_RNG))

TOP_K_PLY1 = 10


class HeuristicDet2plyBattlePolicy(BattlePolicy):  # type: ignore[misc]
    """Heuristic 2-ply policy with deterministic forward sim.

    Opponent is approximated by :class:`GreedyBattlePolicy` at both plies.
    Falls back to a greedy action if no legal joint actions are enumerable.
    """

    def __init__(self, top_k_ply1: int = TOP_K_PLY1) -> None:
        self._opp = GreedyBattlePolicy()
        self._fallback = GreedyBattlePolicy()
        self._top_k_ply1 = top_k_ply1

    def set_params(self, params: BattleRuleParam) -> None:
        super().set_params(params)
        self._opp.set_params(params)
        self._fallback.set_params(params)

    def _ply2_value(self, s1: State) -> float:
        """Best heuristic value reachable from ``s1`` after one more turn.

        Opponent is greedy. Returns ``evaluate(s1)`` directly if the state
        is terminal or no joint actions are enumerable (no further lookahead
        possible).
        """
        if s1.terminal():
            return evaluate(s1, self.params)
        actions2 = get_actions((s1.sides[0].team, s1.sides[1].team))
        if not actions2:
            return evaluate(s1, self.params)
        opp_state2 = State((s1.sides[1], s1.sides[0]))
        opp_action2: list[BattleCommand] = self._opp.decision(opp_state2, None)
        best_v2 = float("-inf")
        for a2 in actions2:
            s2 = copy_state(s1)
            forward(
                s2,
                (list(a2), opp_action2),
                self.params,
                acc_rng=_ACC_RNG,
                eff_rng=_EFF_RNG,
                sta_rng=_STA_RNG,
            )
            v = evaluate(s2, self.params)
            if v > best_v2:
                best_v2 = v
        return best_v2

    def decision(
        self,
        state: State,
        opp_view: TeamView | None = None,
    ) -> list[BattleCommand]:
        actions = get_actions((state.sides[0].team, state.sides[1].team))
        if not actions:
            fallback: list[BattleCommand] = self._fallback.decision(state, opp_view)
            return fallback

        opp_state = State((state.sides[1], state.sides[0]))
        opp_action: list[BattleCommand] = self._opp.decision(opp_state, None)

        ply1: list[tuple[float, list[BattleCommand], State]] = []
        for action in actions:
            simulated = copy_state(state)
            forward(
                simulated,
                (list(action), opp_action),
                self.params,
                acc_rng=_ACC_RNG,
                eff_rng=_EFF_RNG,
                sta_rng=_STA_RNG,
            )
            v1 = evaluate(simulated, self.params)
            ply1.append((v1, list(action), simulated))

        ply1.sort(key=lambda item: item[0], reverse=True)
        candidates = ply1[: self._top_k_ply1]

        best_value = float("-inf")
        best_action: list[BattleCommand] | None = None
        for _v1, action, s1 in candidates:
            v2 = self._ply2_value(s1)
            if v2 > best_value:
                best_value = v2
                best_action = action

        if best_action is not None:
            return best_action
        fallback = self._fallback.decision(state, opp_view)
        return fallback
