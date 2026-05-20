"""Team build policy.

Two builders:

- ``MatchupTableTeamBuildPolicy`` — current ``VgcAiTeamBuildPolicy`` default.
  Computes a roster-x-roster 1v1 matchup table (see
  ``vgc_ai.eval.matchup_table``) and picks species via mean-best-matchup
  greedy: first pick is the species with the highest mean win rate against
  the roster; subsequent picks maximize ``mean_j(max_t(M[t][j]))`` — i.e.
  expand the set of roster species the team can collectively beat. Table
  is cached per-roster on the policy instance.

- ``MetaUsageTeamBuildPolicy`` — the prior baseline (PR #12). Pure usage
  / stat-sum rank. Cleared the championship gate vs ``RandomTeamBuildPolicy``
  at +177 ELO. Kept for A/B comparison.

Both builders go through ``_build_team_command``, which applies
**per-species build tuning** to whatever species the builder picks:

- ``_species_role(species)``: physical if ``ATK >= SPA``, else special.
- ``_optimal_evs``: sweeper spread ``(6 HP / 252 ATK or SPA / 252 SPE)``.
- ``_optimal_nature``: ``JOLLY`` / ``TIMID`` if ``SPE >= attacker stat``
  (speed-positive); else ``ADAMANT`` / ``MODEST``. All four move from
  the unused attacker stat — never costing the stat we're using.
- ``_move_priority(species, role)``: ``base_power * STAB * role_match``
  where off-category moves get a 0.7 penalty (not a hard exclusion).

``BasicMeta.usage_rate_pokemon`` raises ``ZeroDivisionError`` when called
before any matches are recorded (epoch 0 of every championship) —
defended in ``_species_priority``.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
from vgc2.agent import TeamBuildCommand, TeamBuildPolicy
from vgc2.balance.meta import Meta, Roster
from vgc2.battle_engine.modifiers import Category, Nature, Stat, Status
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import PokemonSpecies

from vgc_ai.eval.matchup_table import get_or_build_matchup_table

_DEFAULT_IVS: tuple[int, int, int, int, int, int] = (31, 31, 31, 31, 31, 31)
_STAB_MULTIPLIER: float = 1.5
_OFF_CATEGORY_PENALTY: float = 0.7
_MATCHUP_TABLE_N_PER_PAIR: int = 10

Role = Literal["physical", "special"]

# Sweeper EV spreads. Slots match the Stat IntEnum (HP, ATK, DEF, SPA, SPD, SPE).
# 6 + 252 + 252 = 510, which is the cap fix_builds enforces.
_PHYSICAL_EVS: tuple[int, int, int, int, int, int] = (6, 252, 0, 0, 0, 252)
_SPECIAL_EVS: tuple[int, int, int, int, int, int] = (6, 0, 0, 252, 0, 252)


def _species_role(species: PokemonSpecies) -> Role:
    """Physical or special attacker based on base stats."""
    atk = species.base_stats[Stat.ATTACK]
    spa = species.base_stats[Stat.SPECIAL_ATTACK]
    return "physical" if atk >= spa else "special"


def _optimal_nature(species: PokemonSpecies, role: Role) -> Nature:
    """Pick a nature that boosts our offensive stat without hurting it.

    All four candidate natures move from the *unused* attacker stat into
    either the attacker stat or Speed — never costing us the stat we
    depend on. The speed-positive variants (JOLLY / TIMID) win when the
    species is already fast enough that the extra raw offense isn't worth
    the slower movement.
    """
    speed = species.base_stats[Stat.SPEED]
    if role == "physical":
        attacker = species.base_stats[Stat.ATTACK]
        # JOLLY: +SPE -SPA (never costs ATK). ADAMANT: +ATK -SPA.
        return Nature.JOLLY if speed >= attacker else Nature.ADAMANT
    attacker = species.base_stats[Stat.SPECIAL_ATTACK]
    # TIMID: +SPE -ATK. MODEST: +SPA -ATK.
    return Nature.TIMID if speed >= attacker else Nature.MODEST


def _optimal_evs(role: Role) -> tuple[int, int, int, int, int, int]:
    return _PHYSICAL_EVS if role == "physical" else _SPECIAL_EVS


def _species_priority(roster: Roster, meta: Meta | None) -> list[int]:
    """Return roster indices ordered by team-build preference (highest first).

    Uses ``meta.usage_rate_pokemon`` when meta has data; otherwise falls back
    to base-stat sum. Ties broken by roster index (stable).
    """
    if meta is not None:
        # BasicMeta.usage_rate_pokemon divides by len(record); raises
        # ZeroDivisionError at epoch 0 of every championship. Treat that the
        # same as "meta has no usable data yet" and fall back to stat sum.
        try:
            usage = [meta.usage_rate_pokemon(species) for species in roster]
        except ZeroDivisionError:
            usage = []
        if usage and any(u > 0.0 for u in usage):
            return sorted(range(len(roster)), key=lambda i: (-usage[i], i))
    return sorted(range(len(roster)), key=lambda i: (-sum(roster[i].base_stats), i))


def _move_priority(species: PokemonSpecies, role: Role | None = None) -> list[int]:
    """Return species move indices ordered by score (highest first).

    Score = ``base_power * (1.5 if STAB else 1.0) * (1.0 if matches role else 0.7)``.
    When ``role`` is ``None``, the role match is skipped. Ties broken by move
    index.
    """
    species_types = set(species.types)
    target_category: Category | None
    if role == "physical":
        target_category = Category.PHYSICAL
    elif role == "special":
        target_category = Category.SPECIAL
    else:
        target_category = None

    def score(move: Move) -> float:
        stab = _STAB_MULTIPLIER if move.pkm_type in species_types else 1.0
        role_match = 1.0
        if target_category is not None and move.base_power > 0:
            role_match = 1.0 if move.category == target_category else _OFF_CATEGORY_PENALTY
        return float(move.base_power) * stab * role_match

    return sorted(range(len(species.moves)), key=lambda i: (-score(species.moves[i]), i))


def _build_team_command(roster: Roster, picks: list[int], max_pkm_moves: int) -> TeamBuildCommand:
    """Build per-species-tuned commands for the chosen picks.

    Each picked species gets a role-matched EV spread, nature, and
    role-weighted move priority. Used by both ``MetaUsageTeamBuildPolicy``
    and ``MatchupTableTeamBuildPolicy``.
    """
    cmds: TeamBuildCommand = []
    for species_idx in picks:
        species = roster[species_idx]
        role = _species_role(species)
        evs = _optimal_evs(role)
        nature = _optimal_nature(species, role)
        move_idx = _move_priority(species, role)[:max_pkm_moves]
        cmds.append((species_idx, evs, _DEFAULT_IVS, nature, move_idx))
    return cmds


class MetaUsageTeamBuildPolicy(TeamBuildPolicy):  # type: ignore[misc]
    """Rank species by meta usage (or stat sum); per-species-tuned build."""

    def decision(
        self,
        roster: Roster,
        meta: Meta | None,
        max_team_size: int,
        max_pkm_moves: int,
        n_active: int,
    ) -> TeamBuildCommand:
        ranked = _species_priority(roster, meta)[:max_team_size]
        return _build_team_command(roster, ranked, max_pkm_moves)


def _greedy_coverage_picks(table: npt.NDArray[np.float64], max_team_size: int) -> list[int]:
    """Pick indices that maximize ``mean_j(max_t(M[t][j]))`` greedily.

    First pick: species with the highest mean row (best average matchup vs
    the rest of the roster). Each subsequent pick: the candidate whose row
    most increases ``max_t(M[t][j])`` averaged across remaining ``j``.

    Returns up to ``max_team_size`` indices.
    """
    n = table.shape[0]
    if n == 0 or max_team_size <= 0:
        return []
    mean_per_species = table.mean(axis=1)
    first = int(np.argmax(mean_per_species))
    picks: list[int] = [first]
    used: set[int] = {first}
    # Track per-opponent best matchup the team currently provides.
    best_against = table[first].copy()
    while len(picks) < min(max_team_size, n):
        best_score = -np.inf
        best_pick: int | None = None
        for c in range(n):
            if c in used:
                continue
            new_best = np.maximum(best_against, table[c])
            # Score = mean coverage against opponents not on the team.
            mask = np.ones(n, dtype=bool)
            for p in picks:
                mask[p] = False
            mask[c] = False
            score = float(new_best.mean()) if not mask.any() else float(new_best[mask].mean())
            if score > best_score:
                best_score = score
                best_pick = c
        if best_pick is None:
            break
        picks.append(best_pick)
        used.add(best_pick)
        best_against = np.maximum(best_against, table[best_pick])
    return picks


class MatchupTableTeamBuildPolicy(TeamBuildPolicy):  # type: ignore[misc]
    """Greedy team builder over a precomputed roster-x-roster matchup table.

    Falls back to ``MetaUsageTeamBuildPolicy``'s ranking when the roster is
    empty or ``max_team_size`` is zero. Uses the module-level matchup-table
    cache in ``vgc_ai.eval.matchup_table`` so the table is shared with any
    other policy (e.g. ``MatchupAwareSelectionPolicy``) that needs it.
    """

    def __init__(self, n_battles_per_pair: int = _MATCHUP_TABLE_N_PER_PAIR) -> None:
        self._n_battles_per_pair = n_battles_per_pair

    def decision(
        self,
        roster: Roster,
        meta: Meta | None,
        max_team_size: int,
        max_pkm_moves: int,
        n_active: int,
    ) -> TeamBuildCommand:
        if not roster or max_team_size <= 0:
            return []
        table = get_or_build_matchup_table(
            roster,
            n_battles_per_pair=self._n_battles_per_pair,
            max_pkm_moves=max_pkm_moves,
        )
        picks = _greedy_coverage_picks(table, max_team_size)
        return _build_team_command(roster, picks, max_pkm_moves)


def _solve_minimax_policy(table: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Solve the row player's max-min Nash policy over a zero-sum payoff matrix.

    ``table[i][j]`` is row i's win-rate vs column j (our matchup table is in
    this form). Returns the equilibrium mixing distribution ``p`` of length
    ``n``, where ``p[i]`` is the optimal probability of picking row i.

    Algorithm (standard zero-sum LP, identical in shape to Reis's
    ``vgc-agents/teambuilders.py:get_policy``):

        variables x = [v, p_0, ..., p_{n-1}]
        minimize   -v             (i.e. maximize the worst-case payoff)
        subject to v - p^T M[:, j] <= 0   for each column j
                   sum(p) = 1
                   p_i >= 0,  v unbounded

    Falls back to a uniform distribution if scipy reports the LP infeasible
    or the solver fails (defensive — shouldn't happen for a finite,
    well-formed matchup table).
    """
    from scipy.optimize import linprog

    n = table.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    c = np.zeros(n + 1, dtype=np.float64)
    c[0] = -1.0  # minimize -v == maximize v

    # A_ub: row j is [+1, -M[0][j], -M[1][j], ...]; v - sum(p_i * M[i][j]) <= 0.
    a_ub = np.zeros((n, n + 1), dtype=np.float64)
    a_ub[:, 0] = 1.0
    a_ub[:, 1:] = -table.T
    b_ub = np.zeros(n, dtype=np.float64)

    a_eq = np.zeros((1, n + 1), dtype=np.float64)
    a_eq[0, 1:] = 1.0
    b_eq = np.array([1.0])

    bounds: list[tuple[float | None, float | None]] = [(None, None)] + [(0.0, None)] * n

    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds)
    if not result.success:
        return np.full(n, 1.0 / n, dtype=np.float64)
    p: npt.NDArray[np.float64] = np.asarray(result.x[1:], dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0.0:
        return np.full(n, 1.0 / n, dtype=np.float64)
    return p / s


def _minimax_picks(table: npt.NDArray[np.float64], max_team_size: int) -> list[int]:
    """Pick the top ``max_team_size`` species by Nash equilibrium mass.

    Ties broken by descending mean row (raw matchup strength), then by
    roster index. Pure-deterministic — we treat the LP weights as a
    *ranking* over which species to bring, not as a sampling distribution
    for an individual roll. Random sampling would inject per-call variance
    that would hurt later cache lookups and is unnecessary for a team-of-N
    pre-selection problem.
    """
    n = table.shape[0]
    if n == 0 or max_team_size <= 0:
        return []
    p = _solve_minimax_policy(table)
    row_mean = table.mean(axis=1)
    order = sorted(range(n), key=lambda i: (-p[i], -row_mean[i], i))
    return order[:max_team_size]


class MinimaxTeamBuildPolicy(TeamBuildPolicy):  # type: ignore[misc]
    """LP-minimax team builder over the singleton matchup table.

    Solves a zero-sum-game LP for the Nash equilibrium mixing distribution
    over species, then deterministically picks the top-``max_team_size``
    species by equilibrium mass. This is more robust than
    ``MatchupTableTeamBuildPolicy``'s greedy coverage when the roster has
    a non-trivial rock-paper-scissors structure: the LP finds a portfolio
    that's hard to counter, where greedy coverage can over-commit to one
    branch of the matchup graph.

    Uses the same module-level singleton matchup table as
    ``MatchupTableTeamBuildPolicy``, so the precompute cost (~5s for a
    30-species roster) is shared, not doubled.
    """

    def __init__(self, n_battles_per_pair: int = _MATCHUP_TABLE_N_PER_PAIR) -> None:
        self._n_battles_per_pair = n_battles_per_pair

    def decision(
        self,
        roster: Roster,
        meta: Meta | None,
        max_team_size: int,
        max_pkm_moves: int,
        n_active: int,
    ) -> TeamBuildCommand:
        if not roster or max_team_size <= 0:
            return []
        table = get_or_build_matchup_table(
            roster,
            n_battles_per_pair=self._n_battles_per_pair,
            max_pkm_moves=max_pkm_moves,
        )
        picks = _minimax_picks(table, max_team_size)
        return _build_team_command(roster, picks, max_pkm_moves)


def _meta_weights(roster: Roster, meta: Meta | None) -> npt.NDArray[np.float64]:
    """Per-species opponent weights from observed meta usage.

    Falls back to uniform ``(1/n, ..., 1/n)`` when ``meta`` is ``None``, when
    ``usage_rate_pokemon`` raises ``ZeroDivisionError`` (epoch 0 of every
    championship), or when the meta has no signal (sum == 0). Always returns
    a probability vector summing to 1.
    """
    n = len(roster)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    uniform = np.full(n, 1.0 / n, dtype=np.float64)
    if meta is None:
        return uniform
    try:
        raw = np.array([meta.usage_rate_pokemon(sp) for sp in roster], dtype=np.float64)
    except ZeroDivisionError:
        return uniform
    total = float(raw.sum())
    if total <= 0.0:
        return uniform
    return raw / total


def _weighted_greedy_coverage_picks(
    table: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    max_team_size: int,
) -> list[int]:
    """Pick indices that maximize ``sum_j(weights[j] * max_t(M[t][j]))``.

    Strict generalization of ``_greedy_coverage_picks``: with uniform weights
    the two functions return identical picks. When ``weights`` reflects
    meta usage, opponents that actually appear get up-weighted, biasing
    picks toward meta counters. Classical weighted max-cover greedy
    approximation (``1 - 1/e`` of optimal).
    """
    n = table.shape[0]
    if n == 0 or max_team_size <= 0:
        return []
    # First pick: species with the highest weighted mean row.
    weighted_means = table @ weights
    first = int(np.argmax(weighted_means))
    picks: list[int] = [first]
    used: set[int] = {first}
    best_against = table[first].copy()
    while len(picks) < min(max_team_size, n):
        best_score = -np.inf
        best_pick: int | None = None
        for c in range(n):
            if c in used:
                continue
            new_best = np.maximum(best_against, table[c])
            score = float((new_best * weights).sum())
            if score > best_score:
                best_score = score
                best_pick = c
        if best_pick is None:
            break
        picks.append(best_pick)
        used.add(best_pick)
        best_against = np.maximum(best_against, table[best_pick])
    return picks


class MetaCoverageTeamBuildPolicy(TeamBuildPolicy):  # type: ignore[misc]
    """Meta-usage-weighted greedy coverage team builder.

    Generalizes ``MatchupTableTeamBuildPolicy``: weight the coverage
    objective by observed meta usage rates instead of treating opponents
    as uniformly likely. With uniform weights (``meta is None`` / epoch 0
    / no usage signal) the FIRST pick is identical to
    ``MatchupTableTeamBuildPolicy`` (both ``argmax`` the row mean), but
    later picks may differ: ``MatchupTableTeamBuildPolicy`` masks
    already-picked species from the coverage average, while this policy
    leaves the full opponent distribution in the score. The unmasked
    form lines up cleanly with the meta-weighted generalisation
    (``sum_j(w[j] * max_t(M[t][j]))``), where masking would be
    inconsistent with treating the meta as the opponent distribution.
    When the meta has signal, picks shift toward species that beat
    high-usage opponents.

    Strategic insight vs ``MinimaxTeamBuildPolicy``: minimax solves the
    Nash equilibrium against an adversarial opponent who picks worst-case
    for us. In a real competitive field, opponents don't pick adversarially
    — they cluster around the empirical meta. Optimizing against the meta
    distribution dominates worst-case Nash when the field isn't actually
    adversarial.
    """

    def __init__(self, n_battles_per_pair: int = _MATCHUP_TABLE_N_PER_PAIR) -> None:
        self._n_battles_per_pair = n_battles_per_pair

    def decision(
        self,
        roster: Roster,
        meta: Meta | None,
        max_team_size: int,
        max_pkm_moves: int,
        n_active: int,
    ) -> TeamBuildCommand:
        if not roster or max_team_size <= 0:
            return []
        table = get_or_build_matchup_table(
            roster,
            n_battles_per_pair=self._n_battles_per_pair,
            max_pkm_moves=max_pkm_moves,
        )
        weights = _meta_weights(roster, meta)
        picks = _weighted_greedy_coverage_picks(table, weights, max_team_size)
        return _build_team_command(roster, picks, max_pkm_moves)


def _has_speed_control_move(species: PokemonSpecies) -> bool:
    """Does this species carry any move that gives the team speed control?

    Per principle #1: priority > 0, Trick Room toggle, Tailwind toggle, or a
    paralysis-inflicting move all count. Without any of these the team has
    no answer to a faster opponent setup.
    """
    for m in species.moves:
        if m.priority > 0:
            return True
        if m.toggle_trickroom:
            return True
        if m.toggle_tailwind:
            return True
        if int(m.status) == int(Status.PARALYZED):
            return True
    return False


def _has_status_move(species: PokemonSpecies) -> bool:
    """Per principle #12: any move inflicting a non-NONE status."""
    return any(int(m.status) != int(Status.NONE) for m in species.moves)


def _principle_bonus(roster: Roster, picks: list[int]) -> float:
    """Encoded subset of the 15-principle doubles checklist.

    Additive bonus / penalty applied per candidate team during greedy pick.
    Bonus magnitudes are tuned so principle compliance biases close calls
    (matchup-score deltas <~0.1) without overriding genuinely better
    coverage. Principles encoded:

    - #1 speed-control redundancy: 2+ sources reward; 0 sources gets no
      bonus (implicit penalty vs compliant alternatives).
    - #7 phys/spec split: both attacker types on the team.
    - #10 bulk floor: penalty when 2+ picks have combined def+spd < 150
      (glass cannons).
    - #12 status coverage: at least one status-inflicting move on the team.
    - #15 diversify STAB types: bonus per distinct primary type, capped at 3.

    Principles #4/#5/#8/#9/#11 are not encoded because the underlying
    mechanism doesn't exist in vgc2 (no Fake Out, no redirection, no
    abilities, no spread-target flag). Principles #2/#3/#13/#14 are
    deferred to v2 (cheap algorithmic wins are scoped first).
    """
    if not picks:
        return 0.0
    species = [roster[p] for p in picks]
    bonus = 0.0

    # #1 speed-control redundancy
    sc_count = sum(1 for s in species if _has_speed_control_move(s))
    if sc_count >= 2:
        bonus += 0.10
    elif sc_count == 1:
        bonus += 0.03
    # 0 -> no bonus (implicit penalty)

    # #12 status coverage
    if any(_has_status_move(s) for s in species):
        bonus += 0.05

    # #15 diversify STAB types
    distinct_types: set[int] = set()
    for s in species:
        distinct_types.update(int(t) for t in s.types)
    bonus += 0.02 * min(3, len(distinct_types))

    # #7 phys/spec split
    has_phys = any(s.base_stats[Stat.ATTACK] > s.base_stats[Stat.SPECIAL_ATTACK] for s in species)
    has_spec = any(s.base_stats[Stat.SPECIAL_ATTACK] > s.base_stats[Stat.ATTACK] for s in species)
    if has_phys and has_spec:
        bonus += 0.05

    # #10 bulk floor: glass cannon penalty
    glass = sum(
        1 for s in species if s.base_stats[Stat.DEFENSE] + s.base_stats[Stat.SPECIAL_DEFENSE] < 150
    )
    if glass >= 2:
        bonus -= 0.10

    return bonus


def _principled_greedy_picks(
    table: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    roster: Roster,
    max_team_size: int,
) -> list[int]:
    """Meta-weighted greedy coverage with principle bonus added per candidate.

    Same shape as ``_weighted_greedy_coverage_picks`` but each candidate's
    score adds ``_principle_bonus(roster, picks_so_far + [candidate])``.
    Bonuses are small enough (max ~+0.26, min ~-0.10) that they bias ties
    rather than override strong matchup advantages.
    """
    n = table.shape[0]
    if n == 0 or max_team_size <= 0:
        return []
    weighted_means = table @ weights
    best_first = 0
    best_first_score = -float("inf")
    for c in range(n):
        score = float(weighted_means[c]) + _principle_bonus(roster, [c])
        if score > best_first_score:
            best_first_score = score
            best_first = c
    picks: list[int] = [best_first]
    used: set[int] = {best_first}
    best_against = table[best_first].copy()
    while len(picks) < min(max_team_size, n):
        best_score = -float("inf")
        best_pick: int | None = None
        for c in range(n):
            if c in used:
                continue
            new_best = np.maximum(best_against, table[c])
            coverage = float((new_best * weights).sum())
            principle = _principle_bonus(roster, [*picks, c])
            total = coverage + principle
            if total > best_score:
                best_score = total
                best_pick = c
        if best_pick is None:
            break
        picks.append(best_pick)
        used.add(best_pick)
        best_against = np.maximum(best_against, table[best_pick])
    return picks


class PrincipledCoverageTeamBuildPolicy(TeamBuildPolicy):  # type: ignore[misc]
    """Meta-coverage + encoded subset of the 15-principle doubles checklist.

    Same matchup-table cache + meta-weight machinery as
    ``MetaCoverageTeamBuildPolicy``, but each greedy pick also gets a
    principle bonus (see ``_principle_bonus``). The bonus rewards speed-
    control redundancy, status coverage, phys/spec split, type diversity,
    and penalizes glass-cannon stacking — all from data visible on
    ``Move`` / ``PokemonSpecies`` (no abilities required, no spread-target
    flag, no Fake Out flag — vgc2 doesn't have those concepts).

    Why a sibling class instead of folding the bonus into
    ``MetaCoverageTeamBuildPolicy``: the bonus weights are tuned and could
    be wrong. Keeping them in a separate class lets the bench loop A/B
    both compounds against the current default. If PrincipledCoverage
    beats both Minimax and MetaCoverage at statistical significance, we
    promote it; if it underperforms, MetaCoverage stays clean.
    """

    def __init__(self, n_battles_per_pair: int = _MATCHUP_TABLE_N_PER_PAIR) -> None:
        self._n_battles_per_pair = n_battles_per_pair

    def decision(
        self,
        roster: Roster,
        meta: Meta | None,
        max_team_size: int,
        max_pkm_moves: int,
        n_active: int,
    ) -> TeamBuildCommand:
        if not roster or max_team_size <= 0:
            return []
        table = get_or_build_matchup_table(
            roster,
            n_battles_per_pair=self._n_battles_per_pair,
            max_pkm_moves=max_pkm_moves,
        )
        weights = _meta_weights(roster, meta)
        picks = _principled_greedy_picks(table, weights, roster, max_team_size)
        return _build_team_command(roster, picks, max_pkm_moves)


VgcAiTeamBuildPolicy = MinimaxTeamBuildPolicy

__all__ = [
    "MatchupTableTeamBuildPolicy",
    "MetaCoverageTeamBuildPolicy",
    "MetaUsageTeamBuildPolicy",
    "MinimaxTeamBuildPolicy",
    "PrincipledCoverageTeamBuildPolicy",
    "VgcAiTeamBuildPolicy",
]
