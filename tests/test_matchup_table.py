"""Unit tests for the matchup-table generator."""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from vgc2.util.generator import gen_move_set, gen_pkm_roster

from vgc_ai.eval.matchup_table import (
    build_doubles_matchup_table,
    build_matchup_table,
    roster_cache_key,
)


def _make_roster(seed: int = 42, n_species: int = 6, n_moves: int = 12):
    rng = default_rng(seed)
    move_set = gen_move_set(n_moves, rng=rng)
    roster = gen_pkm_roster(n_species, move_set, rng=rng)
    return move_set, roster


def test_table_shape_matches_roster() -> None:
    _, roster = _make_roster(n_species=5)
    table = build_matchup_table(roster, n_battles_per_pair=2)
    assert table.shape == (5, 5)


def test_table_diagonal_is_half() -> None:
    _, roster = _make_roster(n_species=5)
    table = build_matchup_table(roster, n_battles_per_pair=2)
    for i in range(5):
        assert table[i][i] == 0.5


def test_table_is_skew_symmetric() -> None:
    _, roster = _make_roster(n_species=5)
    table = build_matchup_table(roster, n_battles_per_pair=2)
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            assert np.isclose(table[i][j] + table[j][i], 1.0)


def test_table_values_in_unit_interval() -> None:
    _, roster = _make_roster(n_species=6)
    table = build_matchup_table(roster, n_battles_per_pair=3)
    assert (table >= 0).all()
    assert (table <= 1).all()


def test_roster_cache_key_stable_on_same_roster() -> None:
    _, roster = _make_roster()
    assert roster_cache_key(roster) == roster_cache_key(roster)


def test_roster_cache_key_differs_on_distinct_objects() -> None:
    _, roster_a = _make_roster(seed=1)
    _, roster_b = _make_roster(seed=2)
    assert roster_cache_key(roster_a) != roster_cache_key(roster_b)


def test_doubles_table_shape_matches_roster() -> None:
    _, roster = _make_roster(n_species=5)
    table = build_doubles_matchup_table(roster, n_battles_per_pair=2)
    assert table.shape == (5, 5)


def test_doubles_table_diagonal_is_half() -> None:
    _, roster = _make_roster(n_species=5)
    table = build_doubles_matchup_table(roster, n_battles_per_pair=2)
    for i in range(5):
        assert table[i][i] == 0.5


def test_doubles_table_is_skew_symmetric() -> None:
    _, roster = _make_roster(n_species=5)
    table = build_doubles_matchup_table(roster, n_battles_per_pair=2)
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            assert np.isclose(table[i][j] + table[j][i], 1.0)


def test_doubles_table_values_in_unit_interval() -> None:
    _, roster = _make_roster(n_species=6)
    table = build_doubles_matchup_table(roster, n_battles_per_pair=2)
    assert (table >= 0).all()
    assert (table <= 1).all()
