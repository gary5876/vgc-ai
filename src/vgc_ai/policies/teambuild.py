"""Team build policy.

Two builders:

- ``MetaUsageTeamBuildPolicy`` — current ``VgcAiTeamBuildPolicy`` default.
  Pure usage / stat-sum rank. Cleared the championship gate vs.
  ``RandomTeamBuildPolicy`` at +177 ELO over 5 epochs (PR #12).

- ``CoverageMetaTeamBuildPolicy`` — alternative. Extends MetaUsage with
  type-coverage greedy: picks the highest-priority species first; for each
  remaining slot picks the candidate that maximizes
  ``primary_score + COVERAGE_WEIGHT * coverage_gain`` where:

  - ``primary_score`` is the species's normalized usage (or stat-sum) rank
    in [0, 1].
  - ``coverage_gain`` is the marginal change in the team's offensive type
    coverage minus its defensive weakness, weighted by roster type
    prevalence (more weight to types that show up often, on the bet that
    opponents draw from the same pool).

  A/B testing at COVERAGE_WEIGHT=0.15 (5 seeds x 10 epochs) gave a mean
  ELO delta of -4 vs MetaUsage with a range of [-110, +98] — i.e. no
  detectable improvement under the bench's signal-to-noise. Kept for
  future tuning (different weight, deeper roster knowledge, or a real
  matchup table); not the default.

Per-species move pick, EVs/IVs/Nature defaults, and meta epoch-0
``ZeroDivisionError`` defense are all shared via the helpers below.
"""

from __future__ import annotations

from vgc2.agent import TeamBuildCommand, TeamBuildPolicy
from vgc2.balance.meta import Meta, Roster
from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.damage_calculator import type_effectiveness_modifier
from vgc2.battle_engine.modifiers import Nature, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import PokemonSpecies

_DEFAULT_EVS: tuple[int, int, int, int, int, int] = (85, 85, 85, 85, 85, 85)
_DEFAULT_IVS: tuple[int, int, int, int, int, int] = (31, 31, 31, 31, 31, 31)
_DEFAULT_NATURE: Nature = Nature.SERIOUS
_STAB_MULTIPLIER: float = 1.5
_COVERAGE_WEIGHT: float = 0.15
_WEAKNESS_PENALTY_RATIO: float = 0.5
_SUPER_EFFECTIVE_THRESHOLD: float = 2.0


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


def _type_prevalence(roster: Roster) -> dict[Type, float]:
    """Fraction of roster species that have each type."""
    counts: dict[Type, int] = {}
    for species in roster:
        for t in species.types:
            counts[t] = counts.get(t, 0) + 1
    n = max(len(roster), 1)
    return {t: c / n for t, c in counts.items()}


def _offensive_types(species: PokemonSpecies) -> set[Type]:
    """Move types of this species's damaging moves (base_power > 0)."""
    return {move.pkm_type for move in species.moves if move.base_power > 0}


def _coverage_score(
    team_picks: list[int],
    roster: Roster,
    prevalence: dict[Type, float],
    params: BattleRuleParam,
) -> float:
    """Sum over types of: (1 if team can hit super-effectively else 0) * prevalence
    minus a penalty for team weaknesses against the same types.
    """
    if not team_picks:
        return 0.0
    team_offense_types: set[Type] = set()
    for idx in team_picks:
        team_offense_types |= _offensive_types(roster[idx])

    offensive = 0.0
    weakness = 0.0
    for target_type, weight in prevalence.items():
        hits_target = any(
            type_effectiveness_modifier(params, atk, [target_type]) >= _SUPER_EFFECTIVE_THRESHOLD
            for atk in team_offense_types
        )
        if hits_target:
            offensive += weight
        for idx in team_picks:
            if (
                type_effectiveness_modifier(params, target_type, roster[idx].types)
                >= _SUPER_EFFECTIVE_THRESHOLD
            ):
                weakness += weight
    return offensive - _WEAKNESS_PENALTY_RATIO * weakness


class CoverageMetaTeamBuildPolicy(TeamBuildPolicy):  # type: ignore[misc]
    """Greedy build: usage-ranked first pick, subsequent picks maximize coverage."""

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
        params = BattleRuleParam()
        prevalence = _type_prevalence(roster)
        ranked = _species_priority(roster, meta)
        n = len(ranked)
        primary: dict[int, float] = {}
        for rank, idx in enumerate(ranked):
            primary[idx] = 1.0 - rank / max(n - 1, 1)

        picks: list[int] = [ranked[0]]
        used: set[int] = {ranked[0]}
        while len(picks) < min(max_team_size, n):
            best_score = float("-inf")
            best_pick: int | None = None
            base_coverage = _coverage_score(picks, roster, prevalence, params)
            for candidate in ranked:
                if candidate in used:
                    continue
                trial = [*picks, candidate]
                gain = _coverage_score(trial, roster, prevalence, params) - base_coverage
                score = primary[candidate] + _COVERAGE_WEIGHT * gain
                if score > best_score:
                    best_score = score
                    best_pick = candidate
            if best_pick is None:
                break
            picks.append(best_pick)
            used.add(best_pick)

        return _build_team_command(roster, picks, max_pkm_moves)


VgcAiTeamBuildPolicy = MetaUsageTeamBuildPolicy

__all__ = [
    "CoverageMetaTeamBuildPolicy",
    "MetaUsageTeamBuildPolicy",
    "VgcAiTeamBuildPolicy",
]
