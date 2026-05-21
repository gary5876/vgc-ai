"""Unit tests for ``StabAwareSelectionPolicy``."""

from __future__ import annotations

import math

from numpy.random import default_rng
from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.modifiers import Category, Nature, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies
from vgc2.battle_engine.team import Team
from vgc2.util.generator import gen_team

from vgc_ai.policies.selection import (
    MatchupAwareSelectionPolicy,
    StabAwareSelectionPolicy,
    _best_stab_offense_multiplier,
    _stab_aware_selection_score,
)

PARAMS = BattleRuleParam()


def _mk_move(pkm_type: Type, base_power: int = 80) -> Move:
    return Move(
        pkm_type=pkm_type,
        base_power=base_power,
        accuracy=1.0,
        max_pp=10,
        category=Category.PHYSICAL,
    )


def _mk_pkm(types: list[Type], move_types: list[Type]) -> Pokemon:
    moves = [_mk_move(t) for t in move_types]
    species = PokemonSpecies(
        base_stats=(100, 100, 100, 100, 100, 100),
        types=types,
        moves=moves,
    )
    return Pokemon(
        species=species,
        move_indexes=list(range(len(moves))),
        evs=(85, 85, 85, 85, 85, 85),
        ivs=(31, 31, 31, 31, 31, 31),
        nature=Nature.SERIOUS,
    )


def test_subclasses_matchup_aware() -> None:
    assert issubclass(StabAwareSelectionPolicy, MatchupAwareSelectionPolicy)


def test_stab_multiplier_applies_to_same_type_move() -> None:
    fire = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    normal = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    m = _best_stab_offense_multiplier(fire, normal, PARAMS)
    assert math.isclose(m, 1.5, rel_tol=1e-9)


def test_no_stab_when_move_type_differs() -> None:
    fire = _mk_pkm(types=[Type.FIRE], move_types=[Type.NORMAL])
    normal = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    m = _best_stab_offense_multiplier(fire, normal, PARAMS)
    assert math.isclose(m, 1.0, rel_tol=1e-9)


def test_stab_compounds_with_type_effectiveness() -> None:
    fire = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    grass = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    m = _best_stab_offense_multiplier(fire, grass, PARAMS)
    assert math.isclose(m, 3.0, rel_tol=1e-9)


def test_no_damaging_moves_returns_neutral_baseline() -> None:
    moves = [
        Move(
            pkm_type=Type.NORMAL,
            base_power=0,
            accuracy=1.0,
            max_pp=10,
            category=Category.OTHER,
        )
    ]
    species = PokemonSpecies(
        base_stats=(100, 100, 100, 100, 100, 100),
        types=[Type.NORMAL],
        moves=moves,
    )
    me = Pokemon(
        species=species,
        move_indexes=[0],
        evs=(85, 85, 85, 85, 85, 85),
        ivs=(31, 31, 31, 31, 31, 31),
        nature=Nature.SERIOUS,
    )
    opp = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert math.isclose(_best_stab_offense_multiplier(me, opp, PARAMS), 1.0, rel_tol=1e-9)


def test_stab_aware_score_empty_opp_team_is_zero() -> None:
    me = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _stab_aware_selection_score(me, Team([]), PARAMS) == 0.0


def test_stab_breaks_tie_between_same_type_chart_matchup() -> None:
    stab_fire = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    nonstab_fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    grass = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    my_team = Team([nonstab_fire, stab_fire])
    opp_team = Team([grass])
    new_cmd = StabAwareSelectionPolicy().decision((my_team, opp_team), 2)
    parent_cmd = MatchupAwareSelectionPolicy().decision((my_team, opp_team), 2)
    assert parent_cmd == [0, 1]
    assert new_cmd == [1, 0]


def test_opp_stab_threat_penalises_defense() -> None:
    grass_lead = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    water_lead = _mk_pkm(types=[Type.WATER], move_types=[Type.NORMAL])
    fire_opp = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    my_team = Team([grass_lead, water_lead])
    opp_team = Team([fire_opp])
    cmd = StabAwareSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == [1, 0]


def test_stab_aware_does_not_override_type_advantage() -> None:
    nonstab_fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    stab_water = _mk_pkm(types=[Type.WATER], move_types=[Type.WATER])
    grass = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    my_team = Team([stab_water, nonstab_fire])
    opp_team = Team([grass])
    cmd = StabAwareSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == [1, 0]


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(3141)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = StabAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(3142)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = StabAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(3143)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = StabAwareSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_meta_is_ignored() -> None:
    from vgc2.balance.meta import BasicMeta

    rng = default_rng(3144)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy_a = StabAwareSelectionPolicy()
    policy_b = StabAwareSelectionPolicy()
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99, 1, 1]
    policy_b.set_meta(meta)
    assert policy_a.decision((my_team, opp_team), 4) == policy_b.decision((my_team, opp_team), 4)


def test_empty_opp_team_returns_stable_order() -> None:
    rng = default_rng(3145)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = StabAwareSelectionPolicy()
    cmd = policy.decision((my_team, empty), 4)
    assert cmd == [0, 1, 2, 3]


def test_stab_aware_matches_parent_when_no_one_has_stab() -> None:
    a = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    b = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    c = _mk_pkm(types=[Type.NORMAL], move_types=[Type.GRASS])
    opp1 = _mk_pkm(types=[Type.FIRE], move_types=[Type.WATER])
    opp2 = _mk_pkm(types=[Type.GRASS], move_types=[Type.FIRE])
    my_team = Team([a, b, c])
    opp_team = Team([opp1, opp2])
    new_cmd = StabAwareSelectionPolicy().decision((my_team, opp_team), 3)
    parent_cmd = MatchupAwareSelectionPolicy().decision((my_team, opp_team), 3)
    assert new_cmd == parent_cmd
