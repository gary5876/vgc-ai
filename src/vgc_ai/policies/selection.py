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
their leads are unknown. Averaging over their full team is the
conservative substitute.

Negative results recorded (so future tuners don't repeat them):

- Singleton (``n_active=1``) matchup table for scoring: -90 ELO mean over
  5 seeds x 10 epochs vs this type-chart proxy (PR #18). Likely cause: the
  singleton table doesn't capture doubles lead positioning.
- Doubles (``n_active=2``, paired-with-sampled-teammate) matchup table:
  -86 ELO mean over 5 seeds x 10 epochs (this PR's experiment). The
  doubles signal is *also* dominated by championship-level ELO variance;
  the smooth type-chart score outperforms the noisy simulated win rates
  at all sample sizes we've tried. Next leverage on selection is
  game-theoretic (LP-minimax over the doubles table for an opponent-
  uncertainty-aware mixed strategy), not more pointwise scoring variants.
"""

from __future__ import annotations

from vgc2.agent import SelectionCommand, SelectionPolicy
from vgc2.balance.meta import Meta
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


# Public alias used by bench/run_selection_doubles_ab.py to import the
# canonical scorer without depending on a private name.
_type_chart_score = _selection_score


class MatchupAwareSelectionPolicy(SelectionPolicy):  # type: ignore[misc]
    """Order team members by net type matchup vs the opponent's team."""

    def __init__(self) -> None:
        self._meta: Meta | None = None

    def set_meta(self, meta: Meta) -> None:
        # v2.1.x Championship Track hook — store the meta for later
        # consumption in scoring (usage-weighted priors). Plumbing only;
        # _selection_score still uses the type-chart proxy.
        super().set_meta(meta)
        self._meta = meta

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
