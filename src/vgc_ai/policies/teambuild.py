"""Team build policy.

Two builders:

- ``MatchupTableTeamBuildPolicy`` — current ``VgcAiTeamBuildPolicy`` default.
  Computes a roster-x-roster 1v1 matchup table (see
  ``vgc_ai.eval.matchup_table``) and picks species via mean-best-matchup
  greedy: first pick is the species with the highest mean win rate against
  the roster; subsequent picks maximize ``mean_j(max_t(M[t][j]))`` — i.e.
  expand the set of roster species the team can collectively beat. Table is
  cached per-roster on the policy instance.

- ``MetaUsageTeamBuildPolicy`` — the prior baseline (PR #12). Pure usage /
  stat-sum rank. Cleared the championship gate vs ``RandomTeamBuildPolicy``
  at +177 ELO. Kept for A/B comparison.

For both: move pick is ``top max_pkm_moves by base_power * STAB``. EVs flat
``(85,)*6``, IVs ``(31,)*6``, Nature.SERIOUS. ``BasicMeta.usage_rate_pokemon``
raises ZeroDivisionError when called before any matches — defended in
``_species_priority`` for the few code paths that still consult meta.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from vgc2.agent import TeamBuildCommand, TeamBuildPolicy
from vgc2.balance.meta import Meta, Roster
from vgc2.battle_engine.modifiers import Nature
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import PokemonSpecies

from vgc_ai.eval.matchup_table import build_matchup_table, roster_cache_key

_DEFAULT_EVS: tuple[int, int, int, int, int, int] = (85, 85, 85, 85, 85, 85)
_DEFAULT_IVS: tuple[int, int, int, int, int, int] = (31, 31, 31, 31, 31, 31)
_DEFAULT_NATURE: Nature = Nature.SERIOUS
_STAB_MULTIPLIER: float = 1.5
_MATCHUP_TABLE_N_PER_PAIR: int = 10


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


def _move_priority(species: PokemonSpecies) -> list[int]:
    """Return species move indices ordered by score (highest first).

    Score = ``base_power * (1.5 if STAB else 1.0)``. Ties broken by move index.
    """
    species_types = set(species.types)

    def score(move: Move) -> float:
        stab = _STAB_MULTIPLIER if move.pkm_type in species_types else 1.0
        return float(move.base_power) * stab

    return sorted(range(len(species.moves)), key=lambda i: (-score(species.moves[i]), i))


def _build_team_command(roster: Roster, picks: list[int], max_pkm_moves: int) -> TeamBuildCommand:
    cmds: TeamBuildCommand = []
    for species_idx in picks:
        species = roster[species_idx]
        move_idx = _move_priority(species)[:max_pkm_moves]
        cmds.append((species_idx, _DEFAULT_EVS, _DEFAULT_IVS, _DEFAULT_NATURE, move_idx))
    return cmds


class MetaUsageTeamBuildPolicy(TeamBuildPolicy):  # type: ignore[misc]
    """Rank species by meta usage (or stat sum), pick best moves by base_power * STAB."""

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
    empty or ``max_team_size`` is zero. Per-instance cache keyed by
    ``roster_cache_key`` so each fresh championship pays the table-build cost
    only once.
    """

    def __init__(self, n_battles_per_pair: int = _MATCHUP_TABLE_N_PER_PAIR) -> None:
        self._n_battles_per_pair = n_battles_per_pair
        self._cache: dict[tuple[int, ...], npt.NDArray[np.float64]] = {}

    def _get_table(self, roster: Roster, max_pkm_moves: int) -> npt.NDArray[np.float64]:
        key = (*roster_cache_key(roster), max_pkm_moves)
        if key not in self._cache:
            self._cache[key] = build_matchup_table(
                roster,
                n_battles_per_pair=self._n_battles_per_pair,
                max_pkm_moves=max_pkm_moves,
            )
        return self._cache[key]

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
        table = self._get_table(roster, max_pkm_moves)
        picks = _greedy_coverage_picks(table, max_team_size)
        return _build_team_command(roster, picks, max_pkm_moves)


VgcAiTeamBuildPolicy = MatchupTableTeamBuildPolicy

__all__ = [
    "MatchupTableTeamBuildPolicy",
    "MetaUsageTeamBuildPolicy",
    "VgcAiTeamBuildPolicy",
]
