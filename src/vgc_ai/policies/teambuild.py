"""Team build policy.

``MetaUsageTeamBuildPolicy`` is the first non-trivial team builder we ship.

Algorithm:

1. Rank species by ``meta.usage_rate_pokemon`` (high → low). When meta has no
   data (epoch 0, all-zero rates, or ``meta is None``), fall back to ranking
   by base-stat sum so the picks are still principled rather than random.
2. Take the top ``max_team_size`` species.
3. For each picked species, rank its movepool by
   ``base_power * (1.5 if STAB else 1.0)`` (highest first) and keep the top
   ``max_pkm_moves`` move indices. STAB = move's ``pkm_type`` is in the
   species' ``types``.
4. EVs flat (85, 85, 85, 85, 85, 85) — sums to 510, the cap enforced by
   ``vgc2.battle_engine.security.fix_builds``.
5. IVs (31,) * 6 — the canonical maximum.
6. Nature ``Nature.SERIOUS`` — the engine's neutral nature (no stat shift in
   ``vgc2.battle_engine.constants.NATURES``).

Targets ``RandomTeamBuildPolicy`` as the bar (which picks 3 species
uniformly at random regardless of ``max_team_size``). The expected edge
comes from (a) filling all 4 team slots, (b) picking strong species, and
(c) picking strong moves. Refinements (matchup-table-driven coverage / LP
minimax / GA counter) are deferred to follow-up tasks.
"""

from __future__ import annotations

from vgc2.agent import TeamBuildCommand, TeamBuildPolicy
from vgc2.balance.meta import Meta, Roster
from vgc2.battle_engine.modifiers import Nature
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import PokemonSpecies

_DEFAULT_EVS: tuple[int, int, int, int, int, int] = (85, 85, 85, 85, 85, 85)
_DEFAULT_IVS: tuple[int, int, int, int, int, int] = (31, 31, 31, 31, 31, 31)
_DEFAULT_NATURE: Nature = Nature.SERIOUS
_STAB_MULTIPLIER: float = 1.5


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
        cmds: TeamBuildCommand = []
        for species_idx in ranked:
            species = roster[species_idx]
            move_idx = _move_priority(species)[:max_pkm_moves]
            cmds.append((species_idx, _DEFAULT_EVS, _DEFAULT_IVS, _DEFAULT_NATURE, move_idx))
        return cmds


VgcAiTeamBuildPolicy = MetaUsageTeamBuildPolicy

__all__ = ["MetaUsageTeamBuildPolicy", "VgcAiTeamBuildPolicy"]
