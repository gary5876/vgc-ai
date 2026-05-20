"""Unit tests for ``LibraryTeamBuildPolicy``.

The policy has two interesting paths to verify:

- HIT: roster fingerprint is in the library → return the stored picks.
- MISS: fingerprint absent or entry malformed → fall back to the fallback
  policy and still return a legal team.

The runtime guarantee is "always returns a legal TeamBuildCommand," so
malformed library entries must not raise.
"""

from __future__ import annotations

from numpy.random import default_rng
from vgc2.util.generator import gen_move_set, gen_pkm_roster

from vgc_ai.eval.roster_fingerprint import roster_fingerprint
from vgc_ai.policies.library_teambuild import LibraryTeamBuildPolicy

MAX_TEAM_SIZE = 4
MAX_PKM_MOVES = 4
N_ACTIVE = 2


def _make_roster(seed: int = 42, n_species: int = 8, n_moves: int = 16):
    rng = default_rng(seed)
    move_set = gen_move_set(n_moves, rng=rng)
    for i, m in enumerate(move_set):
        m.id = i
    roster = gen_pkm_roster(n_species, move_set, rng=rng)
    for i, s in enumerate(roster):
        s.id = i
    return move_set, roster


def test_hit_returns_stored_picks() -> None:
    _, roster = _make_roster(n_species=6)
    fp = roster_fingerprint(roster)
    library = {fp: {"picks": [0, 1, 2, 3], "notes": "fixture"}}
    policy = LibraryTeamBuildPolicy(library=library)
    cmd = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    assert [entry[0] for entry in cmd] == [0, 1, 2, 3]


def test_miss_falls_back_to_meta_coverage() -> None:
    """Empty library → fallback fires → still returns a legal team."""
    _, roster = _make_roster(n_species=8)
    policy = LibraryTeamBuildPolicy(library={})
    cmd = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    assert len(cmd) == MAX_TEAM_SIZE
    ids = [entry[0] for entry in cmd]
    assert len(set(ids)) == MAX_TEAM_SIZE
    assert all(0 <= i < len(roster) for i in ids)


def test_malformed_entry_falls_back() -> None:
    """A library entry with out-of-range picks must trigger fallback, not crash."""
    _, roster = _make_roster(n_species=6)
    fp = roster_fingerprint(roster)
    library = {fp: {"picks": [0, 1, 2, 999]}}  # 999 is out of range
    policy = LibraryTeamBuildPolicy(library=library)
    cmd = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    # Fallback fired — must return a legal team.
    assert len(cmd) == MAX_TEAM_SIZE
    ids = [entry[0] for entry in cmd]
    assert all(0 <= i < len(roster) for i in ids)


def test_entry_with_wrong_count_falls_back() -> None:
    """An entry with too few picks must trigger fallback rather than return
    a short team."""
    _, roster = _make_roster(n_species=6)
    fp = roster_fingerprint(roster)
    library = {fp: {"picks": [0, 1]}}  # short of MAX_TEAM_SIZE=4
    policy = LibraryTeamBuildPolicy(library=library)
    cmd = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    assert len(cmd) == MAX_TEAM_SIZE
