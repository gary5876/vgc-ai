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

``PairCoverageSelectionPolicy`` takes the orthogonal angle: every other
policy here scores team members independently, then sorts the singletons.
That aggregation collapses the doubles game state to a singles-equivalent
-- two identical 2x-Fire leads rank the same regardless of whether
they're paired against an all-Fire or all-Water opp. PairCoverage scores
*pairs* (i, j) jointly with max-of-pair offense and max-of-pair defense,
then picks the best-pair-by-coverage as the leads. Non-decomposable
signal -- two leads scored identically can compose into very different
pairs depending on which opp threats they cover jointly.
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


def _pair_coverage_score(
    me_a: Pokemon,
    me_b: Pokemon,
    opp_team: Team,
    params: BattleRuleParam,
) -> float:
    """Best-of-pair offense minus worst-of-pair defense, averaged over opp_team.

    Captures the doubles reality that both active leads attack and defend
    every turn:

    - Offense vs opp k: ``max(off(a, k), off(b, k))`` -- the better-
      positioned lead targets k. Rewards *complementary* coverage; a
      2x-vs-Fire + 2x-vs-Water pair covers both threats, where a
      2x-vs-Fire + 2x-vs-Fire pair has a redundant slot.
    - Defense vs opp k: ``max(off(k, a), off(k, b))`` -- opp focus-fires
      whichever of our pair is more vulnerable to k. Penalises pairs
      where any member is one-shot by a common opp threat; one tanky
      partner does not paper over a frail one being removed.

    Returns the mean of (best-offense - worst-defense) over the opp
    team. Aggregation is non-decomposable: two leads scored identically
    as singletons can compose into very different pairs depending on
    which opp threats they cover jointly. Returns 0.0 on an empty opp
    team (matches ``_selection_score`` for the degenerate case).
    """
    if not opp_team.members:
        return 0.0
    offense = 0.0
    defense = 0.0
    for opp in opp_team.members:
        off_a = _best_offense_multiplier(me_a, opp, params)
        off_b = _best_offense_multiplier(me_b, opp, params)
        def_a = _best_offense_multiplier(opp, me_a, params)
        def_b = _best_offense_multiplier(opp, me_b, params)
        offense += off_a if off_a > off_b else off_b
        defense += def_a if def_a > def_b else def_b
    n = len(opp_team.members)
    return (offense - defense) / n


class PairCoverageSelectionPolicy(MatchupAwareSelectionPolicy):
    """Order team members by joint-pair coverage of the opponent's team.

    Every other selection policy in the module scores team members
    *independently* and sorts those singleton scores. That collapses
    the doubles game state to a singles-equivalent: two identical
    2x-Fire leads rank the same regardless of whether they share a
    weakness or counter different opp threats. This policy is the
    single-axis change vs ``MatchupAwareSelectionPolicy`` -- same
    type-chart primitive (``_best_offense_multiplier``), pair-level
    aggregation:

    - For each candidate pair (a, b), score it via
      ``_pair_coverage_score`` -- best-of-pair offense per opp,
      worst-of-pair defense per opp, averaged.
    - The lead pair is the argmax across all ``C(n, 2)`` pairs.
      ``n`` is the team size (typically <= 6) so the quadratic is
      cheap and the search exhaustive.
    - Within the chosen pair, the higher-singleton-score member takes
      the very first slot (stable on ties: keep the original
      ``(i, j)`` order so the index tiebreak matches the parent's
      behaviour at degenerate matchups).
    - Remaining team slots fill by singleton score (same comparator
      as the parent) so the reserve order stays comparable across
      A/B benches.

    Theoretical leverage over the parent:

    - Complementary coverage: a 2x-vs-Fire + 2x-vs-Water pair beats a
      2x-vs-Fire + 2x-vs-Fire pair against a Fire/Water opp duo,
      because the singleton-ranker would happily promote the second
      Fire lead even though it duplicates coverage the first one
      already provides.
    - Worst-case defense: a single 2x weakness shared by both leads
      costs the pair the full 2.0 (max), whereas the singleton-ranker
      averages it 1/N across opponents and may under-penalise
      structural fragility.

    Falls back to ``MatchupAwareSelectionPolicy.decision`` when the
    team is smaller than 2 or ``max_size < 2`` (singles regime) -- so
    the worst case at ``n_active=1`` is parity with the default.

    Ignores ``meta`` on purpose -- this is the single-axis change vs
    the type-chart parent, isolating the *pair-aggregation* improvement
    from any usage-prior or LP-minimax composition.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        params: BattleRuleParam = self.params
        n = len(my_team.members)
        if n < 2 or max_size < 2:
            return super().decision(teams, max_size)
        best_score = -float("inf")
        best_pair: tuple[int, int] = (0, 1)
        for i in range(n):
            me_i = my_team.members[i]
            for j in range(i + 1, n):
                s = _pair_coverage_score(me_i, my_team.members[j], opp_team, params)
                if s > best_score:
                    best_score = s
                    best_pair = (i, j)
        i, j = best_pair
        si = _selection_score(my_team.members[i], opp_team, params)
        sj = _selection_score(my_team.members[j], opp_team, params)
        if sj > si:
            i, j = j, i
        leads = [i, j]
        used = {i, j}
        rest = sorted(
            (k for k in range(n) if k not in used),
            key=lambda k: (
                -_selection_score(my_team.members[k], opp_team, params),
                k,
            ),
        )
        return (leads + rest)[:max_size]


VgcAiSelectionPolicy = MatchupAwareSelectionPolicy

__all__ = [
    "MatchupAwareSelectionPolicy",
    "MetaThreatAwareSelectionPolicy",
    "MetaWeightedSelectionPolicy",
    "PairCoverageSelectionPolicy",
    "VgcAiSelectionPolicy",
]
