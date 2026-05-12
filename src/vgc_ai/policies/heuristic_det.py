"""One-ply forward-search policy with **deterministic** rollouts.

Same heuristic-evaluation lookahead as the non-deterministic variant, but the
1-ply forward simulation is run with vgc2's deterministic-RNG generators
(``ZERO_RNG`` / ``ONE_RNG`` from ``vgc2.util.rng``) instead of the default
stochastic ``Generator``. The previous heuristic policy hit
``win_rate_a=0.545 (ci95_low=0.4758)`` against Greedy at n=200 — positive but
below the gate; the hypothesis being tested here is that the gap is caused by
1-ply sim noise rather than weight calibration.

RNG choice per channel:

- ``acc_rng = ZERO_RNG``: ``random() == 0`` makes ``random() >= threshold``
  false (for any threshold > 0), so moves always **hit**. Matches the
  convention used by ``TreeSearchBattlePolicy.get_states`` for the "hit"
  branch.
- ``eff_rng = ONE_RNG``: ``random() ~= 1 - eps`` makes ``random() < prob``
  false for any ``prob < 1``, so probabilistic secondary effects do **not**
  fire. Conservative: most secondary effects have ``prob`` ~0.1-0.3, so "no
  proc" is the modal outcome.
- ``sta_rng = ONE_RNG``: ``(1 - eps) < threshold`` is false for the default
  ``PARALYSIS_THRESHOLD = 0.25``, so a paralyzed attacker is **not** blocked.
  Frozen attackers also stay frozen unless the move is FIRE-type (handled
  separately by the engine).

Net effect: the rollout collapses to the most-likely deterministic outcome
(move hits, no surprise side-effects, no status-block), removing the
per-action variance that motivated this task.
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


class HeuristicDetBattlePolicy(BattlePolicy):  # type: ignore[misc]
    """Heuristic 1-ply policy with deterministic forward sim.

    Identical to ``HeuristicBattlePolicy`` modulo the RNG passed to
    :func:`vgc2.util.forward.forward`. The opponent is approximated by
    ``GreedyBattlePolicy`` (same approach as ``TreeSearchBattlePolicy``).
    Falls back to a greedy action if no legal combos are enumerable.
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

        opp_state = State((state.sides[1], state.sides[0]))
        opp_action: list[BattleCommand] = self._opp.decision(opp_state, None)

        best_value = float("-inf")
        best_action: list[BattleCommand] | None = None
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
            value = evaluate(simulated, self.params)
            if value > best_value:
                best_value = value
                best_action = list(action)

        if best_action is not None:
            return best_action
        fallback = self._fallback.decision(state, opp_view)
        return fallback
