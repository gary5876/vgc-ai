"""Selection policy.

``MatchupAwareSelectionPolicy`` ranks our team members by net type matchup
against the opponent's full team and returns indices in descending-score
order. Per ``vgc2.battle_engine.game_state.get_battle_teams``, the first
``n_active`` selected members start as ACTIVE; the remainder become
RESERVES. So even when ``max_size == len(team)`` (the degenerate "which to
bring" case in current Match defaults), the **order** of the selection
still controls who leads — a real lever in doubles.

Score for our member ``i``:

    offense(i) - defense(i)

where, averaged over the opponent's members ``j``:

- ``offense(i)`` = best ``type_effectiveness_modifier(move.pkm_type, j.types)``
  across i's damaging moves (``base_power > 0``).
- ``defense(i)`` = best ``type_effectiveness_modifier(move.pkm_type, i.types)``
  across j's damaging moves.

A member with a 2x advantage move and 1x defensive matchup scores +1.0;
a member that's 2x weak with neutral offense scores -1.0. Ties broken by
original index (stable).

We don't model the opponent's selection — they choose simultaneously, so
their leads are unknown. Averaging over their full team is the conservative
substitute.

A matchup-table-based variant of this scoring (using actual simulated win
rates from ``vgc_ai.eval.matchup_table``) was tried and benched as a
regression vs the type-chart proxy (mean -90 ELO over 5 seeds x 10 epochs
in a championship A/B). Likely cause: the matchup table is built from
singleton ``n_active=1`` battles, so it doesn't capture the doubles lead
positioning and double-targeting that actually drive selection outcomes;
its noise + format mismatch swamps any signal gain. The type-chart proxy
stays as the default.
"""

from __future__ import annotations

from vgc2.agent import SelectionCommand, SelectionPolicy
from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.damage_calculator import type_effectiveness_modifier
from vgc2.battle_engine.pokemon import Pokemon
from vgc2.battle_engine.team import Team


def _best_offense_multiplier(
    attacker: Pokemon, defender: Pokemon, params: BattleRuleParam
) -> float:
    """Best type-effectiveness multiplier from attacker's damaging moves vs defender.

    Returns 1.0 if the attacker has no damaging moves (status-only kit) — the
    neutral baseline. Skips moves with ``base_power == 0`` (status moves).
    """
    best = 1.0
    for move in attacker.moves:
        if move.base_power == 0:
            continue
        m = type_effectiveness_modifier(params, move.pkm_type, defender.species.types)
        if m > best:
            best = m
    return best


def _selection_score(my_pkm: Pokemon, opp_team: Team, params: BattleRuleParam) -> float:
    """Net (offense - defense) advantage of my_pkm averaged over opp_team."""
    if not opp_team.members:
        return 0.0
    offense = 0.0
    defense = 0.0
    for opp in opp_team.members:
        offense += _best_offense_multiplier(my_pkm, opp, params)
        defense += _best_offense_multiplier(opp, my_pkm, params)
    n = len(opp_team.members)
    return (offense - defense) / n


class MatchupAwareSelectionPolicy(SelectionPolicy):  # type: ignore[misc]
    """Order team members by net type matchup vs the opponent's team."""

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        params: BattleRuleParam = self.params
        scored = [
            (
                -_selection_score(p, opp_team, params),
                i,
            )  # negate for descending sort, index for stable tiebreak
            for i, p in enumerate(my_team.members)
        ]
        scored.sort()
        ordered = [i for _, i in scored]
        return ordered[:max_size]


VgcAiSelectionPolicy = MatchupAwareSelectionPolicy

__all__ = ["MatchupAwareSelectionPolicy", "VgcAiSelectionPolicy"]
