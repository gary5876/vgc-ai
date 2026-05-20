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

``MetaWeightedSelectionPolicy`` extends the same primitive: when the
championship meta has populated usage data, the uniform mean over the
opponent's team is replaced by a ``usage_rate_pokemon``-weighted mean
so high-usage opp species drive the score more than rare ones. Falls
back to ``MatchupAwareSelectionPolicy``'s uniform behavior whenever the
meta is absent / empty / yields ``ZeroDivisionError`` (same epoch-0
defense as ``teambuild._species_priority``).

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
from vgc2.battle_engine.modifiers import Category, Stat
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


def _opp_usage_weights(meta: Meta | None, opp_team: Team) -> list[float] | None:
    """Return per-opp-member usage weights normalised to sum to 1.

    Returns ``None`` when the meta is absent, the opponent team is empty,
    ``BasicMeta.usage_rate_pokemon`` raises ``ZeroDivisionError`` (epoch 0
    of every championship — same pattern guarded by
    ``teambuild._species_priority``), or all weights are zero (meta has
    matches recorded but none touched any of these species yet). In every
    fallback case the caller should drop back to uniform scoring.
    """
    if meta is None or not opp_team.members:
        return None
    try:
        raw = [meta.usage_rate_pokemon(opp.species) for opp in opp_team.members]
    except ZeroDivisionError:
        return None
    total = sum(raw)
    if total <= 0.0:
        return None
    return [w / total for w in raw]


def _meta_weighted_selection_score(
    my_pkm: Pokemon,
    opp_team: Team,
    weights: list[float],
    params: BattleRuleParam,
) -> float:
    """Usage-weighted variant of ``_selection_score``.

    ``weights`` must already sum to 1; computed once per ``decision`` call
    by ``_opp_usage_weights``. Same (offense - defense) signal as the
    type-chart baseline, just with non-uniform per-opp contributions —
    high-usage species drive the score more than rare ones.
    """
    if not opp_team.members:
        return 0.0
    score = 0.0
    for opp, w in zip(opp_team.members, weights, strict=True):
        offense = _best_offense_multiplier(my_pkm, opp, params)
        defense = _best_offense_multiplier(opp, my_pkm, params)
        score += w * (offense - defense)
    return score


class MetaWeightedSelectionPolicy(MatchupAwareSelectionPolicy):
    """Weight opponent members by ``meta.usage_rate_pokemon`` in the score.

    Same (offense - defense) primitive as ``MatchupAwareSelectionPolicy``,
    but the uniform mean over the opponent's team is replaced with a
    usage-weighted mean once the championship meta is populated. Rationale:
    the framework hands us the meta via ``set_meta`` so we can prioritise
    leads that counter the opponents most likely to be played; a uniform
    mean discards that signal and treats a 50%-usage staple identically
    to a 5%-usage curiosity sharing the same team slot.

    Strict generalisation of the parent: when the meta is absent OR has
    no usable data yet (epoch 0, or ``ZeroDivisionError`` from
    ``BasicMeta.usage_rate_pokemon``, or all-zero weights) we delegate
    back to ``MatchupAwareSelectionPolicy.decision`` — so the worst case
    is parity, not regression.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        weights = _opp_usage_weights(self._meta, opp_team)
        if weights is None:
            return super().decision(teams, max_size)
        params: BattleRuleParam = self.params
        scored = [
            (
                -_meta_weighted_selection_score(p, opp_team, weights, params),
                i,
            )
            for i, p in enumerate(my_team.members)
        ]
        scored.sort()
        ordered = [i for _, i in scored]
        return ordered[:max_size]


def _meta_threat_aware_selection_score(
    my_pkm: Pokemon,
    opp_team: Team,
    weights: list[float],
    params: BattleRuleParam,
) -> float:
    """Usage-weighted offense minus worst-case threat.

    Composes two single-axis enhancements over ``MatchupAwareSelectionPolicy``:

    - offense term: ``sum_j w_j * best_offense(my, opp_j)`` -- the same
      usage-weighted mean that ``MetaWeightedSelectionPolicy`` uses, so
      high-usage opp species drive the offense signal more than rare ones
      (the opp is more likely to actually field a high-usage species).
    - defense term: ``max_j best_offense(opp_j, my)`` -- worst-case
      threat across the opp team. The max isn't usage-weighted on
      purpose: a 2x super-effective threat that one-shots the lead
      still removes the lead even if the opp's usage rate of it is
      below average -- damage doesn't get diluted by usage probability
      once the species is on the field.

    ``weights`` must already sum to 1; computed once per ``decision`` call
    by ``_opp_usage_weights``. Returns 0.0 on an empty opp team (matches
    the parent for the degenerate case).
    """
    if not opp_team.members:
        return 0.0
    weighted_offense = 0.0
    max_defense = 0.0
    for opp, w in zip(opp_team.members, weights, strict=True):
        weighted_offense += w * _best_offense_multiplier(my_pkm, opp, params)
        threat = _best_offense_multiplier(opp, my_pkm, params)
        if threat > max_defense:
            max_defense = threat
    return weighted_offense - max_defense


def _threat_aware_uniform_score(
    my_pkm: Pokemon,
    opp_team: Team,
    params: BattleRuleParam,
) -> float:
    """(mean_offense - max_defense) net score; the meta-absent fallback path.

    Same offense signal as ``_selection_score`` -- mean of best-multiplier
    vs each opp -- but the defense signal is the worst-case max rather
    than the mean. Used by ``MetaThreatAwareSelectionPolicy`` whenever the
    meta has no usable data yet (epoch 0). Returns 0.0 on an empty opp
    team.
    """
    if not opp_team.members:
        return 0.0
    offense_total = 0.0
    max_defense = 0.0
    for opp in opp_team.members:
        offense_total += _best_offense_multiplier(my_pkm, opp, params)
        threat = _best_offense_multiplier(opp, my_pkm, params)
        if threat > max_defense:
            max_defense = threat
    return (offense_total / len(opp_team.members)) - max_defense


class MetaThreatAwareSelectionPolicy(MatchupAwareSelectionPolicy):
    """Usage-weighted offense minus worst-case threat for the selection score.

    Composes two single-axis improvements over the uniform mean-mean
    parent (``MatchupAwareSelectionPolicy``):

    - Usage-weighted offense (the ``MetaWeightedSelectionPolicy`` insight):
      when the championship meta has populated usage data, weight the
      opponent's members by ``meta.usage_rate_pokemon`` so the offense
      term reflects which opp species the opponent is actually likely
      to field, not a uniform mean over the listed roster.
    - Worst-case (max) threat defense: in doubles a single 2x
      super-effective opp one-shots the lead, so worst-case survival
      dominates average matchup. Damage isn't diluted by the threat's
      usage rate once it's on the field, so the max stays uniform
      across opp members (the offense's usage weight doesn't carry
      through to the defense term).

    Falls back to (uniform_mean_offense - max_defense) whenever the
    meta is absent or has no usable data yet (epoch 0,
    ``ZeroDivisionError`` from ``BasicMeta.usage_rate_pokemon``, or
    all-zero weights). So the worst case at epoch 0 is the uniform
    threat-aware baseline, never a regression to the symmetric mean-mean
    parent.

    Theoretical leverage over the existing default:

    - Offense: a 90%-usage staple drives the score 9x more than a
      10%-usage curiosity, so leads that counter the actually-played
      threats rank higher than under the uniform mean.
    - Defense: a single 2x super-effective threat costs the full 2.0,
      not its 1/N share of the mean -- correct because one threat is
      enough to remove the lead in doubles.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        weights = _opp_usage_weights(self._meta, opp_team)
        params: BattleRuleParam = self.params
        if weights is None:
            scored_fb = [
                (
                    -_threat_aware_uniform_score(p, opp_team, params),
                    i,
                )
                for i, p in enumerate(my_team.members)
            ]
            scored_fb.sort()
            return [i for _, i in scored_fb][:max_size]
        scored = [
            (
                -_meta_threat_aware_selection_score(p, opp_team, weights, params),
                i,
            )
            for i, p in enumerate(my_team.members)
        ]
        scored.sort()
        ordered = [i for _, i in scored]
        return ordered[:max_size]


def _damage_fraction(attacker: Pokemon, defender: Pokemon, params: BattleRuleParam) -> float:
    """Max fraction of defender's HP the attacker can strip in one hit.

    Mirrors ``vgc2.battle_engine.damage_calculator.calculate_damage`` for the
    multiplicative side of the formula, dropping additive level constants and
    state-dependent modifiers (boosts, weather, status, light screen) that
    aren't available pre-battle. For each of the attacker's damaging moves:

        damage = base_power * STAB * type_mod * attack_stat / defense_stat

    where ``attack_stat`` is the attacker's ATK (PHYSICAL move) or SPA
    (SPECIAL); ``defense_stat`` is the defender's DEF / SPD respectively.
    OTHER-category moves and ``base_power == 0`` (status) moves are skipped.
    The returned value is the best such damage divided by defender's MAX_HP --
    so it sits on a ``[0, ~few]`` scale where 1.0 means "this move OHKOs the
    defender at full HP".

    Returns 0.0 when no damaging move exists. This is intentionally lower
    than ``_best_offense_multiplier``'s 1.0 baseline -- a status-only kit
    really *cannot* dent the opp's HP, and the selection score should
    reflect that rather than pretend neutral damage exists.
    """
    if defender.stats[Stat.MAX_HP] <= 0:
        return 0.0
    best = 0.0
    for move in attacker.moves:
        if move.base_power == 0:
            continue
        if move.category == Category.PHYSICAL:
            attack_stat = attacker.stats[Stat.ATTACK]
            defense_stat = defender.stats[Stat.DEFENSE]
        elif move.category == Category.SPECIAL:
            attack_stat = attacker.stats[Stat.SPECIAL_ATTACK]
            defense_stat = defender.stats[Stat.SPECIAL_DEFENSE]
        else:
            continue
        if defense_stat <= 0:
            continue
        stab = params.STAB_MODIFIER if move.pkm_type in attacker.species.types else 1.0
        type_mod = type_effectiveness_modifier(params, move.pkm_type, defender.species.types)
        dmg = move.base_power * stab * type_mod * attack_stat / defense_stat
        frac = dmg / defender.stats[Stat.MAX_HP]
        if frac > best:
            best = frac
    return best


def _damage_aware_score(my_pkm: Pokemon, opp_team: Team, params: BattleRuleParam) -> float:
    """Net (offense - defense) HP-fraction advantage averaged over opp_team.

    Same uniform-mean structure as ``_selection_score``, but offense and
    defense are estimated via ``_damage_fraction`` instead of pure type-chart
    multipliers. The signal is "fraction of opp HP we strip per hit minus
    fraction of our HP they strip per hit" -- positive means we win the
    asymptotic 1v1 trade.
    """
    if not opp_team.members:
        return 0.0
    offense = 0.0
    defense = 0.0
    for opp in opp_team.members:
        offense += _damage_fraction(my_pkm, opp, params)
        defense += _damage_fraction(opp, my_pkm, params)
    n = len(opp_team.members)
    return (offense - defense) / n


class DamageAwareSelectionPolicy(MatchupAwareSelectionPolicy):
    """Order team members by HP-fraction damage averaged over the opponent.

    The existing siblings (``MatchupAwareSelectionPolicy``,
    ``MetaWeightedSelectionPolicy``, ``MetaThreatAwareSelectionPolicy``)
    all score via the same pure type-chart multiplier primitive
    (``_best_offense_multiplier``). That collapses stat differences: a
    frail 30-ATK attacker with a 2x move looks indistinguishable from a
    tanky 250-ATK attacker with a 1x move, even though their realised
    damage outputs differ by an order of magnitude.

    This policy uses ``_damage_fraction`` -- the actual game-formula
    damage divided by defender max HP -- so:

    - Offensive stats matter: a high-ATK / -SPA lead outranks a low-stat
      lead carrying the same type advantage.
    - Defensive bulk matters: a bulky lead absorbs more incoming damage
      from the same opp threat than a frail one, so the defense term is
      naturally scaled by HP and DEF / SPD.
    - Move category matters: PHYSICAL moves are tempered by DEF, SPECIAL
      by SPD -- a mixed defensive profile correctly distinguishes the two
      attacker categories instead of collapsing them.

    Ignores ``meta`` on purpose -- this is the single-axis change vs the
    type-chart parent, isolating the *damage-prediction* improvement from
    any usage-prior or LP-minimax composition. The clean A/B against
    ``minimax+matchup_aware`` is exactly the same uniform-mean formula
    with the offense / defense primitive swapped out.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        params: BattleRuleParam = self.params
        scored = [
            (-_damage_aware_score(p, opp_team, params), i) for i, p in enumerate(my_team.members)
        ]
        scored.sort()
        ordered = [i for _, i in scored]
        return ordered[:max_size]


VgcAiSelectionPolicy = MatchupAwareSelectionPolicy

__all__ = [
    "DamageAwareSelectionPolicy",
    "MatchupAwareSelectionPolicy",
    "MetaThreatAwareSelectionPolicy",
    "MetaWeightedSelectionPolicy",
    "VgcAiSelectionPolicy",
]
