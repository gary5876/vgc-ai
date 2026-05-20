"""Unit tests for ``vgc_ai.eval.roster_fingerprint``.

Three properties matter for the LibraryTeamBuildPolicy lookup:

1. Determinism — same roster, same fingerprint, every time.
2. Order-invariance — shuffling the roster's species order does not change
   the fingerprint (the library's stored picks are indices, but the
   fingerprint key has to be roster-content only).
3. Content-sensitivity — distinct rosters fingerprint distinctly.
"""

from __future__ import annotations

from numpy.random import default_rng
from vgc2.util.generator import gen_move_set, gen_pkm_roster

from vgc_ai.eval.roster_fingerprint import roster_fingerprint


def _make_roster(seed: int = 42, n_species: int = 8, n_moves: int = 16):
    rng = default_rng(seed)
    move_set = gen_move_set(n_moves, rng=rng)
    for i, m in enumerate(move_set):
        m.id = i
    roster = gen_pkm_roster(n_species, move_set, rng=rng)
    for i, s in enumerate(roster):
        s.id = i
    return move_set, roster


def test_fingerprint_is_deterministic() -> None:
    _, roster = _make_roster()
    assert roster_fingerprint(roster) == roster_fingerprint(roster)


def test_fingerprint_is_order_invariant() -> None:
    """Reversing the roster order must produce the same fingerprint."""
    _, roster = _make_roster()
    reversed_roster = list(reversed(roster))
    assert roster_fingerprint(roster) == roster_fingerprint(reversed_roster)


def test_different_rosters_fingerprint_differently() -> None:
    """Two rosters generated from different seeds must fingerprint distinctly."""
    _, roster_a = _make_roster(seed=0)
    _, roster_b = _make_roster(seed=1)
    assert roster_fingerprint(roster_a) != roster_fingerprint(roster_b)
