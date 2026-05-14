"""Unit tests for ``MetaUsageTeamBuildPolicy`` and ``CoverageMetaTeamBuildPolicy``.

Uses ``vgc2.util.generator.gen_pkm_roster`` + ``gen_move_set`` to fabricate a
deterministic roster, and ``BasicMeta`` to provide / suppress usage data.
"""

from __future__ import annotations

from numpy.random import default_rng
from vgc2.balance.meta import BasicMeta
from vgc2.battle_engine.modifiers import Nature
from vgc2.util.generator import gen_move_set, gen_pkm_roster

from vgc_ai.policies.teambuild import (
    CoverageMetaTeamBuildPolicy,
    MetaUsageTeamBuildPolicy,
    VgcAiTeamBuildPolicy,
    _move_priority,
    _species_priority,
)

MAX_TEAM_SIZE = 4
MAX_PKM_MOVES = 4
N_ACTIVE = 2


def _make_roster(seed: int = 42, n_species: int = 12, n_moves: int = 20):
    rng = default_rng(seed)
    move_set = gen_move_set(n_moves, rng=rng)
    for i, m in enumerate(move_set):
        m.id = i
    roster = gen_pkm_roster(n_species, move_set, rng=rng)
    for i, s in enumerate(roster):
        s.id = i
    return move_set, roster


def test_alias_remains_metausage() -> None:
    assert VgcAiTeamBuildPolicy is MetaUsageTeamBuildPolicy


def test_coverage_decision_returns_max_team_size_entries() -> None:
    _, roster = _make_roster()
    policy = CoverageMetaTeamBuildPolicy()
    cmd = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    assert len(cmd) == MAX_TEAM_SIZE


def test_coverage_species_indices_unique_and_in_range() -> None:
    _, roster = _make_roster()
    policy = CoverageMetaTeamBuildPolicy()
    cmd = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    ids = [entry[0] for entry in cmd]
    assert len(set(ids)) == len(ids)
    assert all(0 <= i < len(roster) for i in ids)


def test_coverage_first_pick_matches_priority() -> None:
    _, roster = _make_roster()
    policy = CoverageMetaTeamBuildPolicy()
    cmd = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    expected_first = _species_priority(roster, None)[0]
    assert cmd[0][0] == expected_first


def test_coverage_deterministic_across_calls() -> None:
    _, roster = _make_roster()
    policy = CoverageMetaTeamBuildPolicy()
    a = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    b = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    assert a == b


def test_coverage_handles_empty_roster() -> None:
    policy = CoverageMetaTeamBuildPolicy()
    assert policy.decision([], None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE) == []


def test_coverage_handles_zero_team_size() -> None:
    _, roster = _make_roster()
    policy = CoverageMetaTeamBuildPolicy()
    assert policy.decision(roster, None, 0, MAX_PKM_MOVES, N_ACTIVE) == []


def test_decision_returns_max_team_size_entries() -> None:
    _, roster = _make_roster()
    policy = MetaUsageTeamBuildPolicy()
    cmd = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    assert len(cmd) == MAX_TEAM_SIZE


def test_species_indices_unique_and_in_range() -> None:
    _, roster = _make_roster()
    policy = MetaUsageTeamBuildPolicy()
    cmd = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    ids = [entry[0] for entry in cmd]
    assert len(set(ids)) == len(ids)
    assert all(0 <= i < len(roster) for i in ids)


def test_move_indices_unique_and_capped() -> None:
    _, roster = _make_roster()
    policy = MetaUsageTeamBuildPolicy()
    cmd = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    for species_idx, _, _, _, moves in cmd:
        assert len(set(moves)) == len(moves)
        assert len(moves) <= MAX_PKM_MOVES
        assert all(0 <= i < len(roster[species_idx].moves) for i in moves)


def test_ev_sum_within_cap() -> None:
    _, roster = _make_roster()
    policy = MetaUsageTeamBuildPolicy()
    cmd = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    for _, evs, ivs, nature, _ in cmd:
        assert sum(evs) <= 510
        assert all(0 <= ev <= 255 for ev in evs)
        assert all(0 <= iv <= 31 for iv in ivs)
        assert isinstance(nature, Nature)


def test_falls_back_to_stat_sum_when_meta_is_none() -> None:
    _, roster = _make_roster()
    priority = _species_priority(roster, None)
    # The first picked species must have the largest base-stat sum.
    expected_top = max(range(len(roster)), key=lambda i: sum(roster[i].base_stats))
    assert priority[0] == expected_top


def test_falls_back_to_stat_sum_when_meta_is_empty() -> None:
    move_set, roster = _make_roster()
    meta = BasicMeta(move_set, roster)  # no matches added → all usage rates are 0
    priority_meta = _species_priority(roster, meta)
    priority_none = _species_priority(roster, None)
    assert priority_meta == priority_none


def test_prefers_high_usage_species_when_meta_has_data() -> None:
    move_set, roster = _make_roster()
    meta = BasicMeta(move_set, roster)
    # Hand-inject usage so species 3 has highest pokemon_usage; bypass add_match
    # because that requires a full Team object — we only need usage_rate_pokemon
    # to return a sensible ranking.
    meta.pokemon_usage[3] = 100
    meta.pokemon_usage[1] = 50
    meta.record.append((None, 0, (1500, 1500)))  # type: ignore[arg-type]  # need non-empty record
    priority = _species_priority(roster, meta)
    assert priority[0] == 3
    assert priority[1] == 1


def test_move_priority_orders_by_power_and_stab() -> None:
    _, roster = _make_roster()
    species = roster[0]
    priority = _move_priority(species)
    # Score must be non-increasing along the returned order.
    species_types = set(species.types)
    scored = [
        species.moves[i].base_power * (1.5 if species.moves[i].pkm_type in species_types else 1.0)
        for i in priority
    ]
    assert scored == sorted(scored, reverse=True)


def test_deterministic_across_calls() -> None:
    _, roster = _make_roster()
    policy = MetaUsageTeamBuildPolicy()
    a = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    b = policy.decision(roster, None, MAX_TEAM_SIZE, MAX_PKM_MOVES, N_ACTIVE)
    assert a == b
