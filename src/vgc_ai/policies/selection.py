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

``SpeedTierAwareSelectionPolicy`` adds the most influential single stat
the type-chart aggregators ignore: speed. A faster lead may KO a key
threat before being hit, so an outspeed effectively reduces the
incoming threat that the static type-chart defense term treats as
inevitable. Single-axis change vs ``MatchupAwareSelectionPolicy`` --
same ``_best_offense_multiplier`` primitive, plus a per-opp +/-
``_INITIATIVE_BONUS`` based on ``Pokemon.stats[Stat.SPEED]`` (the
post-EV/IV/nature value), so the JOLLY/TIMID + 252 SPE team-build
spread choices flow through into selection.

``SpeedPairCoverageSelectionPolicy`` composes the two strongest
single-axis selection improvements -- pair-level coverage aggregation
(``PairCoverageSelectionPolicy``) and per-opp speed-tier initiative
(``SpeedTierAwareSelectionPolicy``) -- into one pair scorer. Each
lead in the pair contributes independently to the initiative term per
opp, matching the doubles reality that both leads attack and defend
every turn. Strict refinement of the pair-coverage parent: recovers
the parent when all speeds tie.

``BulkAwareDamageThreatSelectionPolicy`` refines the damage-aware
primitive of ``DamageThreatSelectionPolicy`` along the *bulk* axis: the
BP-only scorer treats every defender as if it had identical HP / DEF /
SPD, so a 110-BP STAB super-effective hit ranks the same against a
frail support and a defensive tank. The vgc2 damage formula is linear
in ``(attacker_offense_stat / defender_defense_stat)`` and the final
HP determines how many such hits the defender survives, so normalising
``BP * STAB * type_eff * (atk / def_)`` by ``defender_max_hp`` turns
the damage proxy into a real KO-likelihood signal. Mean over opp HP
for offense, max over opp HP for defense -- same shape as the
damage-threat parent, just on the per-HP scale.
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


_INITIATIVE_BONUS: float = 0.25
"""Per-opp net bonus applied to ``offense - defense`` when our lead outspeeds.

Tuned so the speed bonus biases close calls (matchup deltas <~0.25) without
overriding a real type advantage. A 2x super-effective matchup is worth
+1.0, a 1x trade is 0.0, and a 0.5x resist is -0.5. Initiative is one
of those secondary signals: an outspeed turns a 1x trade into a slight
favour for us (we may KO before being hit). Symmetric: an outsped lead
loses the same magnitude per opp.

Set to 0.25 by inspection: stronger than a 0.5x resist but weaker than a
2x super-effective hit, so a 2x weakness from the opp still beats a
clean speed advantage. Calibrated to bias selection without overriding
type-chart structure."""


def _speed_tier_score(
    my_pkm: Pokemon,
    opp_team: Team,
    params: BattleRuleParam,
    initiative_bonus: float = _INITIATIVE_BONUS,
) -> float:
    """Net (offense - defense) plus an initiative term per opp, averaged.

    Same primitive as ``_selection_score`` (``_best_offense_multiplier``
    on the type chart), plus a per-opp speed-tier term:

    - +``initiative_bonus`` when our final Speed stat strictly exceeds
      the opp's. Captures the doubles reality that a faster lead may KO
      a key threat before being hit, mitigating an incoming attack the
      static type-chart score would otherwise treat as inevitable.
    - -``initiative_bonus`` when the opp strictly outspeeds. A slow
      lead pays for being out-paced -- in doubles a 2x super-effective
      threat that moves first removes the lead before our turn.
    - 0.0 on a tie. Speed ties roll a coin in vgc2; we don't claim
      either side gets the bonus.

    Reads ``Pokemon.stats[Stat.SPEED]``, the post-EV/IV/nature value
    computed at construction time. Bonus magnitude tuned (see
    ``_INITIATIVE_BONUS``) so the signal biases close calls without
    overriding the type-chart structure.

    Returns 0.0 on an empty opp team (matches ``_selection_score`` for
    the degenerate case).
    """
    if not opp_team.members:
        return 0.0
    my_speed = my_pkm.stats[Stat.SPEED]
    offense = 0.0
    defense = 0.0
    initiative = 0.0
    for opp in opp_team.members:
        offense += _best_offense_multiplier(my_pkm, opp, params)
        defense += _best_offense_multiplier(opp, my_pkm, params)
        opp_speed = opp.stats[Stat.SPEED]
        if my_speed > opp_speed:
            initiative += initiative_bonus
        elif my_speed < opp_speed:
            initiative -= initiative_bonus
    n = len(opp_team.members)
    return (offense - defense + initiative) / n


class SpeedTierAwareSelectionPolicy(MatchupAwareSelectionPolicy):
    """Type-chart selection with a speed-tier initiative bonus.

    Every other policy in this module scores selection purely on
    *static* type-chart matchups -- ``_best_offense_multiplier`` on the
    opponent's defensive types vs our offensive moves and vice versa.
    Speed is the most influential single stat in doubles that those
    scores throw away: a faster lead may KO a threat before being hit,
    so an outspeed effectively reduces the incoming threat that the
    pure type-chart defense term treats as inevitable.

    Single-axis change vs ``MatchupAwareSelectionPolicy``:

    - Same primitive (``_best_offense_multiplier`` on the type chart),
      same average-over-opp aggregation.
    - Adds a per-opp initiative term: +``_INITIATIVE_BONUS`` when our
      ``stats[Stat.SPEED]`` strictly exceeds the opp's, -``_INITIATIVE_BONUS``
      when the opp strictly outspeeds, 0.0 on a tie. Stat is the
      final value (post-EV/IV/nature) computed at ``Pokemon`` build
      time, so the team-build's JOLLY/TIMID + 252 SPE spread choices
      flow through into selection.

    Theoretical leverage over the default:

    - Two leads scored identically on the type chart (same offense, same
      defense averaged across opp) can have very different practical
      threat exposure depending on speed tier. A 1.0-1.0 trade where we
      outspeed lets us strike first; the type-chart score is blind.
    - The bonus is small enough (default 0.25) that it never flips a
      real 2x super-effective matchup -- a 2x weakness still scores
      worse than a clean outspeed. So the policy is a strict
      refinement of the parent: same structure when matchups dominate,
      tiebroken by speed when they don't.

    Ignores ``meta`` on purpose -- this is the single-axis change vs
    the type-chart parent, isolating the *speed-tier* improvement from
    any usage-prior composition. Future compounds can stack speed
    awareness on top of the meta-weighted offense signal.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        params: BattleRuleParam = self.params
        scored = [
            (
                -_speed_tier_score(p, opp_team, params),
                i,
            )
            for i, p in enumerate(my_team.members)
        ]
        scored.sort()
        ordered = [i for _, i in scored]
        return ordered[:max_size]


def _speed_pair_coverage_score(
    me_a: Pokemon,
    me_b: Pokemon,
    opp_team: Team,
    params: BattleRuleParam,
    initiative_bonus: float = _INITIATIVE_BONUS,
) -> float:
    """Pair-coverage score plus per-opp speed-tier initiative.

    Composes the two strongest single-axis selection improvements:

    - ``_pair_coverage_score``: max-of-pair offense and max-of-pair
      defense per opp -- captures complementary type coverage and
      worst-case threat focus.
    - ``_speed_tier_score`` initiative: +/- ``initiative_bonus`` per
      lead-opp pair based on the ``stats[Stat.SPEED]`` comparison.
      Each lead in the pair contributes independently because both
      leads attack and defend every turn in doubles -- two outspeeds
      vs one opp is genuinely twice as valuable as one outspeed.

    Returns the average over the opp team of
    ``(best-offense - worst-defense + lead_a_init + lead_b_init)``.
    Strict refinement of ``_pair_coverage_score``: recovers the parent
    when all speeds tie (initiative term is zero). Returns 0.0 on an
    empty opp team.
    """
    if not opp_team.members:
        return 0.0
    a_speed = me_a.stats[Stat.SPEED]
    b_speed = me_b.stats[Stat.SPEED]
    offense = 0.0
    defense = 0.0
    initiative = 0.0
    for opp in opp_team.members:
        off_a = _best_offense_multiplier(me_a, opp, params)
        off_b = _best_offense_multiplier(me_b, opp, params)
        def_a = _best_offense_multiplier(opp, me_a, params)
        def_b = _best_offense_multiplier(opp, me_b, params)
        offense += off_a if off_a > off_b else off_b
        defense += def_a if def_a > def_b else def_b
        opp_speed = opp.stats[Stat.SPEED]
        if a_speed > opp_speed:
            initiative += initiative_bonus
        elif a_speed < opp_speed:
            initiative -= initiative_bonus
        if b_speed > opp_speed:
            initiative += initiative_bonus
        elif b_speed < opp_speed:
            initiative -= initiative_bonus
    n = len(opp_team.members)
    return (offense - defense + initiative) / n


class SpeedPairCoverageSelectionPolicy(MatchupAwareSelectionPolicy):
    """Pair-coverage selection scoring with per-lead speed-tier initiative.

    Composes the two strongest single-axis selection improvements --
    ``PairCoverageSelectionPolicy`` (joint pair coverage) and
    ``SpeedTierAwareSelectionPolicy`` (per-opp speed-tier initiative) --
    into one pair scorer. Each lead contributes independently to the
    initiative term per opp, matching the doubles reality that both
    leads attack and defend every turn (two outspeeds vs one opp is
    genuinely twice as valuable as one outspeed).

    Decision shape mirrors ``PairCoverageSelectionPolicy``:

    - Argmax over all ``C(n, 2)`` candidate pairs by
      ``_speed_pair_coverage_score``.
    - Within the chosen pair, the higher-speed-tier-singleton-score
      member takes the very first slot (stable on ties: original
      ``(i, j)`` order so the index tiebreak matches the parent).
    - Remaining slots fill by ``_speed_tier_score`` so the reserve
      order also reflects the speed-aware signal.

    Falls back to ``_speed_tier_score`` singleton ranking when the
    team is smaller than 2 or ``max_size < 2`` (singles regime). Both
    parents are strict refinements: pair-coverage when speeds tie,
    speed-tier in the singles fallback.

    Ignores ``meta`` on purpose -- this composes the two type-chart
    structural improvements without entangling the usage-prior axis.
    Future compounds can stack meta-weighting on top of this signal.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        params: BattleRuleParam = self.params
        n = len(my_team.members)
        if n < 2 or max_size < 2:
            scored_single = [
                (-_speed_tier_score(p, opp_team, params), i) for i, p in enumerate(my_team.members)
            ]
            scored_single.sort()
            return [i for _, i in scored_single][:max_size]
        best_score = -float("inf")
        best_pair: tuple[int, int] = (0, 1)
        for i in range(n):
            me_i = my_team.members[i]
            for j in range(i + 1, n):
                s = _speed_pair_coverage_score(me_i, my_team.members[j], opp_team, params)
                if s > best_score:
                    best_score = s
                    best_pair = (i, j)
        i, j = best_pair
        si = _speed_tier_score(my_team.members[i], opp_team, params)
        sj = _speed_tier_score(my_team.members[j], opp_team, params)
        if sj > si:
            i, j = j, i
        leads = [i, j]
        used = {i, j}
        rest = sorted(
            (k for k in range(n) if k not in used),
            key=lambda k: (
                -_speed_tier_score(my_team.members[k], opp_team, params),
                k,
            ),
        )
        return (leads + rest)[:max_size]


def _meta_threat_pair_coverage_score(
    me_a: Pokemon,
    me_b: Pokemon,
    opp_team: Team,
    weights: list[float],
    params: BattleRuleParam,
) -> float:
    """Meta-weighted pair-coverage offense minus worst-case pair threat.

    Three-way composition of the strongest selection signals already in the
    module:

    - Pair-coverage aggregation (the ``PairCoverageSelectionPolicy``
      insight): per-opp ``pair_off_k = max(off(a, k), off(b, k))`` rewards
      complementary type coverage between the two leads.
    - Meta-weighted offense (the ``MetaWeightedSelectionPolicy`` insight):
      offense aggregated as ``sum_k w_k * pair_off_k`` so high-usage opp
      species drive the offense signal more than rare ones.
    - Worst-case (max) threat defense (the ``MetaThreatAwareSelectionPolicy``
      insight): ``defense = max_k max(off(k, a), off(k, b))``. The outer
      max isn't usage-weighted on purpose -- a 2x super-effective threat
      that one-shots whichever lead is more vulnerable still removes the
      lead even if the opp's usage rate of it is below average, and the
      damage doesn't get diluted by usage probability once the species is
      on the field.

    ``weights`` must already sum to 1; computed once per ``decision`` call
    by ``_opp_usage_weights``. Returns 0.0 on an empty opp team.
    """
    if not opp_team.members:
        return 0.0
    weighted_offense = 0.0
    max_defense = 0.0
    for opp, w in zip(opp_team.members, weights, strict=True):
        off_a = _best_offense_multiplier(me_a, opp, params)
        off_b = _best_offense_multiplier(me_b, opp, params)
        pair_offense = off_a if off_a > off_b else off_b
        weighted_offense += w * pair_offense
        def_a = _best_offense_multiplier(opp, me_a, params)
        def_b = _best_offense_multiplier(opp, me_b, params)
        pair_threat = def_a if def_a > def_b else def_b
        if pair_threat > max_defense:
            max_defense = pair_threat
    return weighted_offense - max_defense


def _threat_aware_pair_coverage_score(
    me_a: Pokemon,
    me_b: Pokemon,
    opp_team: Team,
    params: BattleRuleParam,
) -> float:
    """(mean pair-offense - max pair-defense); meta-absent fallback for the pair scorer.

    Same pair-aggregation primitive as ``_meta_threat_pair_coverage_score``
    but the offense is a uniform mean across the opp team instead of a
    usage-weighted sum. Used whenever the meta has no usable data yet
    (epoch 0 of every championship). Returns 0.0 on an empty opp team.
    """
    if not opp_team.members:
        return 0.0
    offense_total = 0.0
    max_defense = 0.0
    for opp in opp_team.members:
        off_a = _best_offense_multiplier(me_a, opp, params)
        off_b = _best_offense_multiplier(me_b, opp, params)
        offense_total += off_a if off_a > off_b else off_b
        def_a = _best_offense_multiplier(opp, me_a, params)
        def_b = _best_offense_multiplier(opp, me_b, params)
        pair_threat = def_a if def_a > def_b else def_b
        if pair_threat > max_defense:
            max_defense = pair_threat
    return (offense_total / len(opp_team.members)) - max_defense


class MetaThreatPairCoverageSelectionPolicy(MetaThreatAwareSelectionPolicy):
    """Meta-weighted pair-coverage offense minus worst-case pair threat.

    Composes the three strongest single-axis selection improvements from
    the rest of this module into one pair scorer:

    - From ``PairCoverageSelectionPolicy``: pair-level joint scoring of
      every ``C(n, 2)`` candidate lead pair instead of independent
      singletons. Captures the doubles reality that two leads scored
      identically on the type chart can compose into very different
      pairs depending on how their coverage and weaknesses overlap.
    - From ``MetaWeightedSelectionPolicy``: usage-weighted opponent
      aggregation for the offense term, so a 50%-usage staple drives
      the score more than a 5%-usage curiosity at the same team slot.
    - From ``MetaThreatAwareSelectionPolicy`` (the current default's
      selection layer): worst-case (max) threat defense -- a single 2x
      super-effective threat that one-shots whichever of our leads is
      more vulnerable removes that lead in doubles, regardless of the
      threat's usage rate. The outer max stays uniform across opp
      members on purpose; damage doesn't get diluted by usage
      probability once the species is on the field.

    Decision shape mirrors ``PairCoverageSelectionPolicy``:

    - Argmax over all ``C(n, 2)`` candidate pairs by
      ``_meta_threat_pair_coverage_score`` (uniform-mean fallback when
      the meta is absent / epoch 0 / all-zero usage).
    - Within the chosen pair, the higher-singleton-score member takes
      the very first slot. Singleton scorer for the tiebreak is the
      parent's meta-threat-aware singleton score
      (``_meta_threat_aware_selection_score`` when meta is usable, the
      ``_threat_aware_uniform_score`` fallback otherwise) so the
      ordering decision uses the same signal as the pair selection
      step. Stable on ties: keep the original ``(i, j)`` order.
    - Remaining slots fill by the same singleton score so the reserve
      order also reflects the meta-threat-aware signal -- important
      because reserves may be brought in later, and a non-meta-aware
      ranker would scramble the strict generalization claim.

    Falls back to ``MetaThreatAwareSelectionPolicy.decision`` when the
    team is smaller than 2 or ``max_size < 2`` (singles regime). The
    parent itself falls back to (uniform_mean_offense - max_defense)
    singleton ranking when the meta is empty, so the worst case at the
    degenerate singles case is parity with the current default.

    Theoretical leverage over each parent in turn:

    - vs ``PairCoverageSelectionPolicy``: uses usage data to bias offense
      toward counters of actually-played species; uses max-threat
      defense per opp (still over a pair) instead of mean-of-pair
      averaged over opp. Both refinements correct under the same
      doubles physics that motivated the parents individually.
    - vs ``MetaThreatAwareSelectionPolicy`` (the current default's
      selection layer): swaps singleton scoring for joint-pair scoring.
      The pair view fixes the redundancy blindspot the singleton
      ranker has -- two identical 2x-Fire leads no longer rank
      equally against a Fire/Water opp duo, because the pair view
      penalises the missing Water coverage that singletons can't see.
    - vs ``MetaWeightedSelectionPolicy``: adds the pair-coverage AND
      max-threat structural improvements stacked on the same
      usage-prior axis.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        n = len(my_team.members)
        if n < 2 or max_size < 2:
            return super().decision(teams, max_size)
        params: BattleRuleParam = self.params
        weights = _opp_usage_weights(self._meta, opp_team)

        def pair_score(i: int, j: int) -> float:
            if weights is None:
                return _threat_aware_pair_coverage_score(
                    my_team.members[i], my_team.members[j], opp_team, params
                )
            return _meta_threat_pair_coverage_score(
                my_team.members[i], my_team.members[j], opp_team, weights, params
            )

        def singleton_score(k: int) -> float:
            if weights is None:
                return _threat_aware_uniform_score(my_team.members[k], opp_team, params)
            return _meta_threat_aware_selection_score(my_team.members[k], opp_team, weights, params)

        best_score = -float("inf")
        best_pair: tuple[int, int] = (0, 1)
        for i in range(n):
            for j in range(i + 1, n):
                s = pair_score(i, j)
                if s > best_score:
                    best_score = s
                    best_pair = (i, j)
        i, j = best_pair
        if singleton_score(j) > singleton_score(i):
            i, j = j, i
        leads = [i, j]
        used = {i, j}
        rest = sorted(
            (k for k in range(n) if k not in used),
            key=lambda k: (-singleton_score(k), k),
        )
        return (leads + rest)[:max_size]


_BP_NORMALIZER: float = 100.0
"""Divisor for the damage-aware score so it stays on the type-chart scale.

``vgc2.util.generator`` samples ``base_power ~ clip(N(100, 40), 0, 140)``,
so dividing by 100 puts the average damaging move's neutral hit near the
type-chart baseline of 1.0 used by every other scorer in this module.
That keeps relative magnitudes comparable across selection policies and
keeps the damage-aware score from numerically dwarfing future
compositions that might add the type-chart primitive back in."""


def _best_damage_potential(attacker: Pokemon, defender: Pokemon, params: BattleRuleParam) -> float:
    """Best ``base_power * STAB * type_eff / _BP_NORMALIZER`` over damaging moves.

    Returns 0.0 if the attacker has no damaging moves (status-only kit) --
    the natural baseline for "this attacker is a non-threat", lower than
    any actual hit. Skips moves with ``base_power == 0`` (status moves)
    the same way ``_best_offense_multiplier`` does.

    The vgc2 damage formula (see ``damage_calculator.calculate_damage``)
    is linear in ``base_power`` and applies STAB and type effectiveness
    multiplicatively, so this product is a faithful proxy for raw damage
    output up to attacker/defender stats. Selection compounds elsewhere
    in this module use the type-eff multiplier alone, which collapses a
    40-BP super-effective hit and a 110-BP super-effective hit to the
    same score; this primitive separates them.
    """
    best = 0.0
    attacker_types = set(attacker.species.types)
    for move in attacker.moves:
        if move.base_power == 0:
            continue
        stab = 1.5 if move.pkm_type in attacker_types else 1.0
        eff = type_effectiveness_modifier(params, move.pkm_type, defender.species.types)
        dmg = float(move.base_power) * stab * eff
        if dmg > best:
            best = dmg
    return best / _BP_NORMALIZER


def _damage_threat_score(my_pkm: Pokemon, opp_team: Team, params: BattleRuleParam) -> float:
    """Mean damage-aware offense minus worst-case damage-aware threat.

    Composes the damage-aware primitive with the threat-aware defense shape
    used by the current championship default:

    - Offense: ``mean_k _best_damage_potential(my, opp_k)`` -- uniform over
      the opp team, matching the type-chart parent. Mean (not max) because
      we don't know which opp the lead will face first; averaging is the
      conservative substitute for the unknown opponent-selection draw.
    - Defense: ``max_k _best_damage_potential(opp_k, my)`` -- worst-case
      across the opp team. A single opp with a high-BP SE move one-shots
      the lead in doubles, so worst-case survival dominates average
      threat (same argument as ``MetaThreatAwareSelectionPolicy``).

    Returns 0.0 on an empty opp team -- matches every other scorer's
    degenerate case so an empty-opp fallback path lands on a stable
    index-tiebroken sort.
    """
    if not opp_team.members:
        return 0.0
    offense_total = 0.0
    max_defense = 0.0
    for opp in opp_team.members:
        offense_total += _best_damage_potential(my_pkm, opp, params)
        threat = _best_damage_potential(opp, my_pkm, params)
        if threat > max_defense:
            max_defense = threat
    return (offense_total / len(opp_team.members)) - max_defense


class DamageThreatSelectionPolicy(MatchupAwareSelectionPolicy):
    """Order team members by damage-aware offense minus worst-case threat.

    Every other policy in this module ranks leads on the type-effectiveness
    multiplier alone -- ``_best_offense_multiplier`` returns 1x / 2x / 4x /
    0.5x and ignores ``move.base_power`` entirely. The vgc2 damage formula
    (``damage_calculator.calculate_damage``) is linear in ``base_power``,
    so collapsing it loses real signal: a 40-BP super-effective move
    ranks identically to a 110-BP super-effective move in the type-chart
    proxy, even though the high-BP move puts roughly 2.75x more raw
    damage on the table.

    Single-axis change vs ``MatchupAwareSelectionPolicy``:

    - Replaces ``_best_offense_multiplier`` with ``_best_damage_potential``
      = ``max_move (base_power * STAB * type_eff) / _BP_NORMALIZER``.
    - Replaces the symmetric mean-mean aggregation with mean-offense
      (uniform over the unknown opp draw) and max-defense (worst-case
      threat, same shape as ``MetaThreatAwareSelectionPolicy``'s
      threat-aware fallback path).

    Theoretical leverage over the current default
    (``matchup_table+matchup_aware``):

    - A 60-BP STAB 2x-SE hit (score 60 * 1.5 * 2 / 100 = 1.80) outranks a
      30-BP STAB 2x-SE hit (score 0.90) where the type-chart proxy ties
      them at 2.0 each. The team-build's ``_move_priority`` already
      prefers high-BP STAB moves, so leads with stronger move kits get
      promoted under selection too -- coherent across the stack.
    - Worst-case defense: a single 110-BP STAB 2x-SE opp move
      (threat = 3.30) costs the full hit, not its 1/N share of the
      mean -- correct under doubles physics because one threat is
      enough to remove the lead.
    - Ignores ``meta`` on purpose -- this isolates the *damage-aware*
      single-axis improvement from any usage-prior or pair-aggregation
      composition. Future compounds can stack meta-weighting or
      pair-coverage on top of this primitive.

    Falls back to a stable index sort when the opp team is empty
    (``_damage_threat_score`` returns 0.0 for every member), matching
    the degenerate-case behaviour of every other scorer here.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        params: BattleRuleParam = self.params
        scored = [
            (
                -_damage_threat_score(p, opp_team, params),
                i,
            )
            for i, p in enumerate(my_team.members)
        ]
        scored.sort()
        ordered = [i for _, i in scored]
        return ordered[:max_size]


def _meta_damage_threat_score(
    my_pkm: Pokemon,
    opp_team: Team,
    weights: list[float],
    params: BattleRuleParam,
) -> float:
    """Usage-weighted damage-aware offense minus worst-case damage threat.

    Composes two single-axis enhancements over ``DamageThreatSelectionPolicy``:

    - offense term: ``sum_j w_j * _best_damage_potential(my, opp_j)`` -- the
      same usage-weighted offense aggregation that
      ``MetaWeightedSelectionPolicy`` uses, but applied to the damage-aware
      primitive ``base_power * STAB * type_eff / _BP_NORMALIZER`` instead of
      the type-chart multiplier. High-usage opp species drive the offense
      signal more than rare ones, AND a 110-BP STAB SE hit no longer ties a
      40-BP STAB SE hit.
    - defense term: ``max_j _best_damage_potential(opp_j, my)`` -- worst-case
      damage threat across the opp team. The max isn't usage-weighted on
      purpose: a high-BP super-effective threat that one-shots the lead
      removes the lead regardless of how often the species is actually
      played -- damage doesn't get diluted by usage probability once the
      species is on the field.

    ``weights`` must already sum to 1; computed once per ``decision`` call
    by ``_opp_usage_weights``. Returns 0.0 on an empty opp team (matches
    every other scorer's degenerate case so stable index tiebreak applies).
    """
    if not opp_team.members:
        return 0.0
    weighted_offense = 0.0
    max_defense = 0.0
    for opp, w in zip(opp_team.members, weights, strict=True):
        weighted_offense += w * _best_damage_potential(my_pkm, opp, params)
        threat = _best_damage_potential(opp, my_pkm, params)
        if threat > max_defense:
            max_defense = threat
    return weighted_offense - max_defense


class MetaDamageThreatSelectionPolicy(MetaThreatAwareSelectionPolicy):
    """Meta-weighted damage-aware offense minus worst-case damage threat.

    Composes the two strongest single-axis selection improvements that act on
    the per-opp scalar score (no pair aggregation, no speed term) into one
    singleton scorer:

    - From ``MetaWeightedSelectionPolicy``: usage-weighted opponent
      aggregation for the offense term, so a 50%-usage staple drives the
      score more than a 5%-usage curiosity at the same team slot.
    - From ``DamageThreatSelectionPolicy``: replaces the type-chart
      multiplier with the damage-aware primitive
      ``base_power * STAB * type_eff / _BP_NORMALIZER`` so a 40-BP STAB
      2x-SE hit no longer ties a 110-BP STAB 2x-SE hit. Defense is the
      worst-case (max) damage threat -- a single high-BP SE move one-shots
      the lead in doubles, so worst-case survival dominates average
      threat (same argument as ``MetaThreatAwareSelectionPolicy``).

    Strict refinement of ``MetaThreatAwareSelectionPolicy`` (the current
    default's selection layer): same usage-weighted-offense / max-threat
    shape, just with the damage-aware primitive in place of the type-chart
    multiplier. Both refinements operate on the same axis the team builder
    optimises for (``_move_priority`` already prefers high-BP STAB moves),
    so leads with stronger move kits get promoted under selection too --
    coherent across the stack.

    Falls back to ``_damage_threat_score`` (uniform mean offense - max
    threat, both damage-aware) whenever the meta is absent or has no
    usable data yet (epoch 0, ``ZeroDivisionError`` from
    ``BasicMeta.usage_rate_pokemon``, or all-zero weights). So the worst
    case at epoch 0 is the proven damage-threat baseline, never a
    regression to the type-chart parent.

    Theoretical leverage over each parent in turn:

    - vs ``MetaThreatAwareSelectionPolicy``: the damage primitive
      distinguishes a 40-BP and a 110-BP super-effective move where the
      type-chart proxy ties them at 2.0 each. In doubles, raw damage
      output is the lever that decides whether a hit OHKOs or leaves the
      target alive to retaliate, so promoting the high-BP lead is the
      direct improvement.
    - vs ``DamageThreatSelectionPolicy``: usage-weighting biases offense
      toward counters of actually-played species. The uniform mean
      treats a 90%-usage staple identically to a 10%-usage curiosity
      sharing the same opp team slot; the meta-weighted variant lets
      the offense signal reflect which opp species the opponent is
      actually likely to field.

    Ignores pair-aggregation and speed-tier on purpose -- this isolates
    the meta + damage composition from those other axes, so the bench
    can attribute any win to that composition specifically. Future
    compounds can stack pair-coverage or speed-tier on top of this
    primitive.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        weights = _opp_usage_weights(self._meta, opp_team)
        params: BattleRuleParam = self.params
        if weights is None:
            scored_fb = [
                (
                    -_damage_threat_score(p, opp_team, params),
                    i,
                )
                for i, p in enumerate(my_team.members)
            ]
            scored_fb.sort()
            return [i for _, i in scored_fb][:max_size]
        scored = [
            (
                -_meta_damage_threat_score(p, opp_team, weights, params),
                i,
            )
            for i, p in enumerate(my_team.members)
        ]
        scored.sort()
        ordered = [i for _, i in scored]
        return ordered[:max_size]


def _speed_damage_threat_score(
    my_pkm: Pokemon,
    opp_team: Team,
    params: BattleRuleParam,
    initiative_bonus: float = _INITIATIVE_BONUS,
) -> float:
    """Damage-aware mean offense minus worst-case damage threat, plus per-opp speed initiative.

    Composes two single-axis enhancements over ``DamageThreatSelectionPolicy``:

    - Damage-aware primitive on both offense and defense terms (already in
      ``_damage_threat_score``): ``_best_damage_potential`` replaces the
      type-chart multiplier so a 110-BP STAB 2x-SE hit no longer ties a
      40-BP STAB 2x-SE hit on either side of the score.
    - Per-opp speed-tier initiative (already in ``_speed_tier_score``):
      +``initiative_bonus`` when our lead outspeeds the opp,
      -``initiative_bonus`` when slower, 0.0 on a tie. The initiative
      term is averaged over the opp team (consistent with the mean-offense
      aggregation -- both are uniform over the unknown opponent-selection
      draw) and is NOT added to the max-threat defense term: once an
      out-speeding super-effective opp is on the field, it still strikes
      first regardless of our matchup against the *rest* of the opp team,
      so the worst-case shape stays intact.

    Returns 0.0 on an empty opp team -- matches every other scorer's
    degenerate case so stable index tiebreak applies in the caller.
    """
    if not opp_team.members:
        return 0.0
    my_speed = my_pkm.stats[Stat.SPEED]
    offense_total = 0.0
    initiative_total = 0.0
    max_defense = 0.0
    for opp in opp_team.members:
        offense_total += _best_damage_potential(my_pkm, opp, params)
        threat = _best_damage_potential(opp, my_pkm, params)
        if threat > max_defense:
            max_defense = threat
        opp_speed = opp.stats[Stat.SPEED]
        if my_speed > opp_speed:
            initiative_total += initiative_bonus
        elif my_speed < opp_speed:
            initiative_total -= initiative_bonus
    n = len(opp_team.members)
    return (offense_total + initiative_total) / n - max_defense


class SpeedDamageThreatSelectionPolicy(DamageThreatSelectionPolicy):
    """Damage-aware mean-offense / max-threat scorer plus per-opp speed initiative.

    Composes the two strongest single-axis selection improvements that act on
    the singleton scorer without invoking the meta or pair-aggregation axes:

    - From ``DamageThreatSelectionPolicy``: replaces the type-chart
      multiplier with the damage-aware primitive
      ``base_power * STAB * type_eff / _BP_NORMALIZER`` on both offense and
      threat terms. A 110-BP STAB super-effective hit no longer ties a
      40-BP STAB super-effective hit; a one-shot threat costs the full
      damage value at the worst-case max, not its 1/N share of a mean.
    - From ``SpeedTierAwareSelectionPolicy``: per-opp +/-``_INITIATIVE_BONUS``
      based on ``Pokemon.stats[Stat.SPEED]``. Captures the doubles reality
      that a faster lead may KO a key threat before being hit, mitigating
      an incoming attack the static damage-threat term would otherwise
      treat as inevitable. Stat is the post-EV/IV/nature value so the
      team-build's JOLLY/TIMID + 252 SPE spread choices flow through.

    Strict refinement of ``DamageThreatSelectionPolicy``: with all speeds
    tied the initiative term is zero everywhere and the policy reduces to
    the damage-threat parent. Same per-opp speed bonus shape as
    ``SpeedTierAwareSelectionPolicy``, just on a damage-aware backbone
    instead of the type-chart proxy.

    Theoretical leverage over each parent:

    - vs ``DamageThreatSelectionPolicy``: two leads with identical damage
      potential against an opp duo (same best move, same STAB / type
      coverage) can have very different practical threat exposure
      depending on speed tier. The damage-threat scorer ranks them
      identically; this scorer breaks the tie toward the lead that
      strikes first.
    - vs ``SpeedTierAwareSelectionPolicy`` (the current default's
      selection layer): a 110-BP STAB 2x-SE lead beats a 40-BP STAB 2x-SE
      lead under this scorer where the type-chart proxy ties them at 2.0
      offense. In doubles the high-BP lead is more likely to OHKO before
      being hit, so promoting it captures damage that the type-chart
      scorer collapses to a flat multiplier.
    - vs the current default compound (``minimax+speed_tier_selection``):
      same team builder, same speed-tier signal -- only difference is the
      damage-aware primitive. The team builder's ``_move_priority`` already
      prefers high-BP STAB moves, so the selection layer's preference for
      hard-hitting leads becomes coherent with the build axis.

    Ignores ``meta`` on purpose -- this isolates the damage + speed
    composition from the usage-prior axis. The minimax team builder is
    fully meta-agnostic anyway, so the entire compound stays meta-free
    and benches deterministically across epochs. Future compounds can
    stack meta-weighting on top.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        params: BattleRuleParam = self.params
        scored = [
            (
                -_speed_damage_threat_score(p, opp_team, params),
                i,
            )
            for i, p in enumerate(my_team.members)
        ]
        scored.sort()
        ordered = [i for _, i in scored]
        return ordered[:max_size]


def _best_effective_damage(attacker: Pokemon, defender: Pokemon, params: BattleRuleParam) -> float:
    r"""Best BP * STAB * type_eff * (atk/def) / max_hp over damaging moves.

    Bulk-aware refinement of ``_best_damage_potential``. The vgc2 damage
    formula (see ``damage_calculator.calculate_damage``) is linear in
    ``base_power`` AND in ``attacking_stats[atk] / defending_stats[def]``,
    and the practical question is "what fraction of the defender's HP does
    the hit remove" -- not raw damage in absolute units. Dividing by
    ``defender.stats[Stat.MAX_HP]`` turns the per-move scalar into a
    KO-likelihood proxy directly comparable across defenders with very
    different bulk profiles.

    Picks the attack/defense stat pair from the move's category --
    PHYSICAL uses ``ATTACK / DEFENSE``, SPECIAL uses
    ``SPECIAL_ATTACK / SPECIAL_DEFENSE`` -- mirroring the engine's
    branch in ``calculate_damage``. STAB and type effectiveness
    multiply on top, exactly as in the engine's modifier composition.

    Returns 0.0 if the attacker has no damaging moves (status-only kit)
    or the defender's max HP is non-positive (defensive guard against
    malformed Pokemon -- ``calculate_stats`` always floors HP at
    ``level + 10`` so this branch is unreachable for any legally-
    generated team, but the guard keeps the scorer robust). Skips moves
    with ``move.category == Category.OTHER`` (status-class damaging
    moves return 0 from the engine) and moves with zero defense stat.
    """
    max_hp = defender.stats[Stat.MAX_HP]
    if max_hp <= 0:
        return 0.0
    best = 0.0
    attacker_types = set(attacker.species.types)
    for move in attacker.moves:
        if move.base_power == 0:
            continue
        if move.category == Category.PHYSICAL:
            atk = attacker.stats[Stat.ATTACK]
            def_ = defender.stats[Stat.DEFENSE]
        elif move.category == Category.SPECIAL:
            atk = attacker.stats[Stat.SPECIAL_ATTACK]
            def_ = defender.stats[Stat.SPECIAL_DEFENSE]
        else:
            continue
        if def_ <= 0:
            continue
        stab = 1.5 if move.pkm_type in attacker_types else 1.0
        eff = type_effectiveness_modifier(params, move.pkm_type, defender.species.types)
        dmg = float(move.base_power) * stab * eff * (float(atk) / float(def_))
        if dmg > best:
            best = dmg
    return best / float(max_hp)


def _bulk_aware_damage_threat_score(
    my_pkm: Pokemon, opp_team: Team, params: BattleRuleParam
) -> float:
    """Mean per-opp-HP offense minus worst-case per-my-HP threat.

    Same aggregation shape as ``_damage_threat_score`` (uniform mean
    offense, max defense) -- the structural improvement is in the
    underlying primitive ``_best_effective_damage``, which normalises by
    the defender's HP and incorporates the relevant offensive/defensive
    stats. Offense is uniform mean because we don't know which opp the
    lead will face first; defense is the max because a single one-shot
    threat removes the lead in doubles regardless of how rare it is.

    HP normalisation is asymmetric on purpose: the offense term divides
    by the opp's ``MAX_HP`` (how lethal are we against them), while the
    defense term divides by our own ``MAX_HP`` (how lethal are they
    against us). Both terms live on the same dimensionless "fraction of
    HP" scale, so the subtraction is meaningful.

    Returns 0.0 on an empty opp team -- matches every other scorer's
    degenerate case so stable index tiebreak applies in the caller.
    """
    if not opp_team.members:
        return 0.0
    offense_total = 0.0
    max_defense = 0.0
    for opp in opp_team.members:
        offense_total += _best_effective_damage(my_pkm, opp, params)
        threat = _best_effective_damage(opp, my_pkm, params)
        if threat > max_defense:
            max_defense = threat
    return (offense_total / len(opp_team.members)) - max_defense


class BulkAwareDamageThreatSelectionPolicy(DamageThreatSelectionPolicy):
    """Bulk-aware damage-threat selection: damage as fraction of defender HP.

    Strict refinement of ``DamageThreatSelectionPolicy`` along the bulk
    axis. Every other damage-aware policy in this module ranks leads on
    ``BP * STAB * type_eff`` -- the BP-only proxy. That collapses bulk
    differences entirely: a 110-BP STAB super-effective hit ranks the
    same against a frail support (low HP, low DEF) and against a
    defensive tank (high HP, high DEF). The vgc2 damage formula
    (``damage_calculator.calculate_damage``) is linear in both
    ``base_power`` AND ``attacking_stats[atk] / defending_stats[def_]``,
    so the BP-only proxy is missing two real signals; the actually-
    relevant quantity in doubles is "what fraction of the defender HP
    does the hit remove", which selects the lead most likely to OHKO a
    key threat.

    Single-axis change vs ``DamageThreatSelectionPolicy``:

    - Replaces ``_best_damage_potential`` (``BP * STAB * type_eff /
      _BP_NORMALIZER``) with ``_best_effective_damage`` (``BP * STAB *
      type_eff * (atk / def_) / max_hp``). Stats are picked by move
      category -- PHYSICAL uses ``ATTACK / DEFENSE``, SPECIAL uses
      ``SPECIAL_ATTACK / SPECIAL_DEFENSE`` -- mirroring the engine's
      branch in ``calculate_damage``.
    - Same mean-offense / max-defense aggregation as the parent. Same
      sort-by-score / stable index tiebreak shape.

    Theoretical leverage over the parent and current default:

    - Bulk asymmetry: a 110-BP STAB SE hit against a 250-HP / 70-DEF
      frail target scores much higher than the same hit against a
      350-HP / 170-DEF bulky target at the same atk. The team builder's
      ``MinimaxTeamBuildPolicy`` is bulk-agnostic too, so it can pick a
      build with a frail offensive partner and a bulky defensive anchor;
      this scorer correctly prefers the offensive lead vs frail opps and
      the defensive lead vs bulky opps.
    - Nature/EV awareness: ``Pokemon.stats`` is the post-EV/IV/nature
      value computed at construction time. The team-build's ATK/SPA
      boosting natures (LONELY, ADAMANT, MODEST, NAUGHTY) and 252 EV
      allocations now feed into selection -- two leads with identical
      moves but different stat spreads no longer tie under this scorer.
    - Coherent with ``DamageThreatSelectionPolicy`` and
      ``SpeedDamageThreatSelectionPolicy``: those already promote
      high-BP leads; this extends the promotion to also prefer leads
      whose attacking stat best exploits the opp's weaker defense
      column (e.g. a special attacker against a phys-bulky / spec-frail
      opp).

    Falls back to a stable index sort when the opp team is empty,
    matching the degenerate-case behaviour of every other scorer here.

    Ignores ``meta`` on purpose -- this isolates the bulk-aware
    damage-primitive improvement from any usage-prior, pair-coverage,
    or speed-tier composition. The minimax team builder is fully
    meta-agnostic anyway, so the compound stays deterministic across
    epochs. Future compounds can stack meta-weighting or pair-coverage
    on top of this primitive.
    """

    def decision(self, teams: tuple[Team, Team], max_size: int) -> SelectionCommand:
        my_team, opp_team = teams
        params: BattleRuleParam = self.params
        scored = [
            (
                -_bulk_aware_damage_threat_score(p, opp_team, params),
                i,
            )
            for i, p in enumerate(my_team.members)
        ]
        scored.sort()
        ordered = [i for _, i in scored]
        return ordered[:max_size]


VgcAiSelectionPolicy = MatchupAwareSelectionPolicy

__all__ = [
    "BulkAwareDamageThreatSelectionPolicy",
    "DamageThreatSelectionPolicy",
    "MatchupAwareSelectionPolicy",
    "MetaDamageThreatSelectionPolicy",
    "MetaThreatAwareSelectionPolicy",
    "MetaThreatPairCoverageSelectionPolicy",
    "MetaWeightedSelectionPolicy",
    "PairCoverageSelectionPolicy",
    "SpeedDamageThreatSelectionPolicy",
    "SpeedPairCoverageSelectionPolicy",
    "SpeedTierAwareSelectionPolicy",
    "VgcAiSelectionPolicy",
]
