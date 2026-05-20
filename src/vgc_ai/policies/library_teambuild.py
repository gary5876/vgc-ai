"""Pre-computed team library lookup with deterministic fallback.

Architecture:

- Offline phase (run BEFORE competition / outside the eval machine):
  a curator (Claude in an interactive session, or any other source)
  designs strong teams for a set of synthetic rosters and writes them
  to ``data/team_library.json``. Each entry is keyed by the roster
  fingerprint from ``vgc_ai.eval.roster_fingerprint`` so two rosters
  with identical content match exactly.

- Runtime phase (this module):
  ``LibraryTeamBuildPolicy.decision`` fingerprints the current roster,
  looks it up, returns the stored picks if present. On miss it
  delegates to ``MetaCoverageTeamBuildPolicy`` so we always return a
  legal team. Pure-deterministic, sub-100 µs lookup, zero network /
  subprocess at runtime. Safe for any submission environment.

The v1 lookup is exact-hash only. Roster-similarity matching for unseen
rosters is a deliberate v2 — exact match validates the mechanism first
and avoids the species-translation problem (library picks are indices
into the LIBRARY's roster, not arbitrary rosters).

Library JSON schema (one fingerprint -> one entry):

    {
      "<sha1-roster-fingerprint>": {
        "roster_summary": "human-readable multi-line dump",
        "picks": [int, ...],
        "notes": "1-2 sentences on the design rationale (optional)",
        "generated_by": "claude-opus-4-7 / hand / ..."
      }
    }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from vgc2.agent import TeamBuildCommand, TeamBuildPolicy
from vgc2.balance.meta import Meta, Roster

from vgc_ai.eval.roster_fingerprint import roster_fingerprint
from vgc_ai.policies.teambuild import (
    MetaCoverageTeamBuildPolicy,
    _build_team_command,
)

DEFAULT_LIBRARY_PATH = Path("data/team_library.json")


def _load_library(path: Path) -> dict[str, dict[str, Any]]:
    """Load the library JSON; return empty dict if missing or malformed.

    A missing file is a normal state — the bench loop benches the policy
    even when the library is empty (it'll just always fall back). Malformed
    JSON degrades the same way, with a stderr warning so the next curator
    pass notices.
    """
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[library-teambuild] could not load {path}: {exc}", file=sys.stderr)
        return {}
    if not isinstance(data, dict):
        print(f"[library-teambuild] {path} is not a JSON object", file=sys.stderr)
        return {}
    return data


def _validate_entry(
    entry: dict[str, Any], roster_size: int, max_team_size: int
) -> list[int] | None:
    """Return the entry's picks if structurally valid, else None.

    Validation: picks present, list of ints, all in range, distinct, exactly
    ``max_team_size`` entries. A failed entry triggers fallback rather than
    an exception so a single bad library row never crashes a tournament.
    """
    picks_raw = entry.get("picks")
    if not isinstance(picks_raw, list):
        return None
    picks: list[int] = []
    for p in picks_raw:
        if not isinstance(p, int) or isinstance(p, bool):
            return None
        if p < 0 or p >= roster_size:
            return None
        if p in picks:
            return None
        picks.append(p)
    if len(picks) != max_team_size:
        return None
    return picks


class LibraryTeamBuildPolicy(TeamBuildPolicy):  # type: ignore[misc]
    """Look up the current roster's fingerprint in a pre-computed team library.

    Falls back to ``MetaCoverageTeamBuildPolicy`` (which itself reduces to
    matchup-table greedy coverage at epoch 0) on any of:
    - Library file missing or malformed
    - Roster fingerprint not present in the library
    - Stored picks fail structural validation (range, distinctness, size)

    The library file is loaded once at ``__init__``; pass a different
    ``library_path`` to swap libraries (used by tests to inject fixtures).
    Pass an explicit ``library`` dict to bypass the file entirely (also
    used by tests).
    """

    def __init__(
        self,
        library_path: Path = DEFAULT_LIBRARY_PATH,
        fallback: TeamBuildPolicy | None = None,
        library: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._library: dict[str, dict[str, Any]] = (
            library if library is not None else _load_library(library_path)
        )
        self._fallback: TeamBuildPolicy = (
            fallback if fallback is not None else MetaCoverageTeamBuildPolicy()
        )

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
        fp = roster_fingerprint(roster)
        entry = self._library.get(fp)
        if entry is None:
            return self._fallback.decision(roster, meta, max_team_size, max_pkm_moves, n_active)
        picks = _validate_entry(entry, roster_size=len(roster), max_team_size=max_team_size)
        if picks is None:
            print(
                f"[library-teambuild] entry for {fp[:12]}... failed validation; fallback",
                file=sys.stderr,
            )
            return self._fallback.decision(roster, meta, max_team_size, max_pkm_moves, n_active)
        return _build_team_command(roster, picks, max_pkm_moves)


__all__ = ["DEFAULT_LIBRARY_PATH", "LibraryTeamBuildPolicy"]
