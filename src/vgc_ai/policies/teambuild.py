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
from vgc2.battle_engine.modifiers import Category, Nature, Stat
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


VgcAiTeamBuildPolicy = MinimaxTeamBuildPolicy

__all__ = [
    "MatchupTableTeamBuildPolicy",
    "MetaUsageTeamBuildPolicy",
    "MinimaxTeamBuildPolicy",
    "VgcAiTeamBuildPolicy",
]
