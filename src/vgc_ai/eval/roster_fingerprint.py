"""Deterministic content-based fingerprint of a vgc2 ``Roster``.

Two rosters with the same species (regardless of order) produce the same
fingerprint. Two rosters differing in any species' stats, types, or move
signatures produce different fingerprints. The fingerprint is the lookup
key for ``LibraryTeamBuildPolicy`` — exact match only in v1; roster-
similarity matching for unseen rosters is a future extension.

The fingerprint hashes only properties the framework treats as stable:
base stats, types, and per-move (type, base_power, accuracy, max_pp,
category, priority). It deliberately ignores move ``name`` and species
``name`` since those can be unset / synthetic and vary across runs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from vgc2.balance.meta import Roster
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import PokemonSpecies


def _move_signature(move: Move) -> tuple[Any, ...]:
    """Stable, content-only signature of a Move."""
    return (
        int(move.pkm_type),
        int(move.base_power),
        round(float(move.accuracy), 3),
        int(move.max_pp),
        int(move.category),
        int(move.priority),
    )


def _species_signature(species: PokemonSpecies) -> tuple[Any, ...]:
    """Stable, content-only signature of a PokemonSpecies.

    Moves are sorted within the species so two species with the same move
    pool in different orders signature identically.
    """
    stats = tuple(int(s) for s in species.base_stats)
    types = tuple(sorted(int(t) for t in species.types))
    moves = tuple(sorted(_move_signature(m) for m in species.moves))
    return (sum(stats), stats, types, moves)


def roster_fingerprint(roster: Roster) -> str:
    """Order-invariant SHA1 hex of a roster's content signatures."""
    sigs = sorted(_species_signature(sp) for sp in roster)
    payload = json.dumps(sigs, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _move_tags(move: Any) -> list[str]:
    """Short tags for move effects relevant to doubles team-building.

    Surfaces the move-class boolean / enum fields a human designer would
    care about: priority, speed-control toggles, screens, hazards, status,
    healing, recoil, switching, protection. Used by ``describe_roster``
    to make the dump principle-checklist-friendly without changing the
    fingerprint payload.
    """
    tags: list[str] = []
    if getattr(move, "priority", 0) > 0:
        tags.append(f"prio+{move.priority}")
    if getattr(move, "protect", False):
        tags.append("PROTECT")
    if getattr(move, "toggle_trickroom", False):
        tags.append("TRICKROOM")
    if getattr(move, "toggle_tailwind", False):
        tags.append("TAILWIND")
    if getattr(move, "toggle_reflect", False):
        tags.append("REFLECT")
    if getattr(move, "toggle_lightscreen", False):
        tags.append("LIGHTSCREEN")
    if getattr(move, "force_switch", False):
        tags.append("FORCESWITCH")
    if getattr(move, "self_switch", False):
        tags.append("SELFSWITCH")
    heal = float(getattr(move, "heal", 0.0))
    if heal > 0.0:
        tags.append(f"heal{heal:.2f}")
    recoil = float(getattr(move, "recoil", 0.0))
    if recoil > 0.0:
        tags.append(f"recoil{recoil:.2f}")
    status = getattr(move, "status", None)
    if status is not None and int(status) != 0:  # Status.NONE == 0
        tags.append(status.name)
    weather = getattr(move, "weather_start", None)
    if weather is not None and int(weather) != 0:  # Weather.CLEAR == 0
        tags.append(weather.name)
    field = getattr(move, "field_start", None)
    if field is not None and int(field) != 0:  # Terrain.NONE == 0
        tags.append(field.name)
    hazard = getattr(move, "hazard", None)
    if hazard is not None and int(hazard) != 0:  # Hazard.NONE == 0
        tags.append(hazard.name)
    if getattr(move, "ignore_evasion", False):
        tags.append("IGN_EVA")
    if getattr(move, "disable", False):
        tags.append("DISABLE")
    return tags


def describe_roster(roster: Roster, max_moves_per_species: int = 8) -> str:
    """Human-readable multi-line summary of a roster.

    Per-move shows type / base_power / category PLUS tags for doubles-
    relevant effects (priority, Trick Room / Tailwind / screens, status,
    healing, recoil, switching, protect, hazards). This is the data the
    library curator reads to apply the 15-principle team-build checklist.
    """
    lines: list[str] = []
    for i, sp in enumerate(roster):
        bs = sp.base_stats
        types = "/".join(t.name for t in sp.types) or "TYPELESS"
        moves = sp.moves[:max_moves_per_species]
        move_strs: list[str] = []
        for m in moves:
            cat = m.category.name[0] if int(m.category) != 0 else "O"
            pow_str = str(m.base_power) if m.base_power > 0 else "-"
            tags = _move_tags(m)
            tag_str = ("[" + ",".join(tags) + "]") if tags else ""
            move_strs.append(f"{m.pkm_type.name}{pow_str}{cat}{tag_str}")
        lines.append(
            f"  [{i}] hp={bs[0]} atk={bs[1]} def={bs[2]} "
            f"spa={bs[3]} spd={bs[4]} spe={bs[5]} "
            f"types=[{types}] moves=[{', '.join(move_strs)}]"
        )
    return "\n".join(lines)


__all__ = ["describe_roster", "roster_fingerprint"]
