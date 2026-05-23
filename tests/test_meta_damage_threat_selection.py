"""Unit tests for ``MetaDamageThreatSelectionPolicy``.

The policy composes two single-axis improvements over its parents:
- usage-weighted opponent aggregation (from ``MetaWeightedSelectionPolicy``)
- damage-aware primitive ``base_power * STAB * type_eff / _BP_NORMALIZER``
  and max-threat defense (from ``DamageThreatSelectionPolicy``)
"""

from __future__ import annotations

import math

from numpy.random import default_rng
from vgc2.balance.meta import BasicMeta
from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.modifiers import Category, Nature, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies
from vgc2.battle_engine.team import Team
from vgc2.util.generator import gen_team

from vgc_ai.policies.selection import (
    _BP_NORMALIZER,
    DamageThreatSelectionPolicy,
    MatchupAwareSelectionPolicy,
    MetaDamageThreatSelectionPolicy,
    MetaThreatAwareSelectionPolicy,
    _damage_threat_score,
    _meta_damage_threat_score,
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


def _mk_pkm(types: list[Type], moves: list[Move]) -> Pokemon:
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


def test_subclasses_meta_threat_aware() -> None:
    assert issubclass(MetaDamageThreatSelectionPolicy, MetaThreatAwareSelectionPolicy)


def test_subclass_of_matchup_aware_chain() -> None:
    assert issubclass(MetaDamageThreatSelectionPolicy, MatchupAwareSelectionPolicy)


def test_meta_damage_threat_score_empty_opp_team_is_zero() -> None:
    me = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL)])
    assert _meta_damage_threat_score(me, Team([]), [], PARAMS) == 0.0


def test_meta_damage_threat_score_pure_offense() -> None:
    me = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=80)])
    opp = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=0)])
    score = _meta_damage_threat_score(me, Team([opp]), [1.0], PARAMS)
    assert math.isclose(score, 80.0 * 1.5 / _BP_NORMALIZER, rel_tol=1e-9)


def test_meta_damage_threat_score_uses_max_defense_uniform() -> None:
    me = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=0)])
    weak_opp = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=40)])
    strong_opp = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.FIRE, base_power=100)])
    score = _meta_damage_threat_score(me, Team([weak_opp, strong_opp]), [0.99, 0.01], PARAMS)
    expected_defense = 100.0 * 1.5 * 2.0 / _BP_NORMALIZER
    assert math.isclose(score, -expected_defense, rel_tol=1e-9)


def test_meta_damage_threat_score_weighted_offense_shifts_with_usage() -> None:
    me = _mk_pkm(types=[Type.WATER], moves=[_mk_move(Type.WATER, base_power=100)])
    grass_opp = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=0)])
    fire_opp = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.NORMAL, base_power=0)])
    low_fire = _meta_damage_threat_score(me, Team([grass_opp, fire_opp]), [0.9, 0.1], PARAMS)
    high_fire = _meta_damage_threat_score(me, Team([grass_opp, fire_opp]), [0.1, 0.9], PARAMS)
    expected_low = 0.9 * 0.75 + 0.1 * 3.0
    expected_high = 0.1 * 0.75 + 0.9 * 3.0
    assert math.isclose(low_fire, expected_low, rel_tol=1e-9)
    assert math.isclose(high_fire, expected_high, rel_tol=1e-9)
    assert high_fire > low_fire


def test_falls_back_to_damage_threat_when_meta_none() -> None:
    rng = default_rng(701)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    no_meta = MetaDamageThreatSelectionPolicy()
    parent = DamageThreatSelectionPolicy()
    cmd = no_meta.decision((my_team, opp_team), 4)
    expected = parent.decision((my_team, opp_team), 4)
    assert cmd == expected


def test_falls_back_to_damage_threat_at_epoch_zero() -> None:
    rng = default_rng(703)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    policy = MetaDamageThreatSelectionPolicy()
    policy.set_meta(meta)
    cmd_meta_empty = policy.decision((my_team, opp_team), 4)
    expected = DamageThreatSelectionPolicy().decision((my_team, opp_team), 4)
    assert cmd_meta_empty == expected


def test_high_bp_lead_beats_low_bp_lead() -> None:
    low = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=40)])
    high = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=110)])
    grass_opp = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=0)])
    my_team = Team([low, high])
    opp_team = Team([grass_opp])
    cmd = MetaDamageThreatSelectionPolicy().decision((my_team, opp_team), 1)
    assert cmd == [1]


def test_meta_shifts_lead_to_high_usage_counter() -> None:
    counter_normal = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIGHT, base_power=80)])
    counter_fire = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.WATER, base_power=80)])
    counter_water = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.GRASS, base_power=80)])
    opp_normal = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=0)])
    opp_fire = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.NORMAL, base_power=0)])
    my_team = Team([counter_normal, counter_fire, counter_water])
    opp_team = Team([opp_normal, opp_fire])
    opp_normal.species.id = 0
    opp_fire.species.id = 1
    meta = BasicMeta(move_set=[], roster=[opp_normal.species, opp_fire.species])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99]
    policy = MetaDamageThreatSelectionPolicy()
    policy.set_meta(meta)
    cmd = policy.decision((my_team, opp_team), 3)
    assert cmd[0] == 1


def test_distinct_from_meta_threat_aware_on_bp_disambiguation() -> None:
    low = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.WATER, base_power=40)])
    high = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.WATER, base_power=110)])
    fire_opp = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.NORMAL, base_power=0)])
    my_team = Team([low, high])
    opp_team = Team([fire_opp])
    fire_opp.species.id = 0
    meta = BasicMeta(move_set=[], roster=[fire_opp.species])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [100]
    parent_policy = MetaThreatAwareSelectionPolicy()
    parent_policy.set_meta(meta)
    new_policy = MetaDamageThreatSelectionPolicy()
    new_policy.set_meta(meta)
    parent_cmd = parent_policy.decision((my_team, opp_team), 1)
    new_cmd = new_policy.decision((my_team, opp_team), 1)
    assert parent_cmd == [0]
    assert new_cmd == [1]


def test_distinct_from_damage_threat_when_meta_present() -> None:
    counter_a = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=80)])
    counter_b = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.WATER, base_power=80)])
    opp_grass = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=0)])
    opp_fire = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.NORMAL, base_power=0)])
    my_team = Team([counter_a, counter_b])
    opp_team = Team([opp_grass, opp_fire])
    opp_grass.species.id = 0
    opp_fire.species.id = 1
    meta = BasicMeta(move_set=[], roster=[opp_grass.species, opp_fire.species])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99]
    new_policy = MetaDamageThreatSelectionPolicy()
    new_policy.set_meta(meta)
    new_cmd = new_policy.decision((my_team, opp_team), 1)
    damage_cmd = DamageThreatSelectionPolicy().decision((my_team, opp_team), 1)
    assert damage_cmd == [0]
    assert new_cmd == [1]


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(705)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = MetaDamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(707)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = MetaDamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(709)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = MetaDamageThreatSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_empty_opp_team_returns_stable_index_order() -> None:
    rng = default_rng(711)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = MetaDamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, empty), 4)
    assert cmd == [0, 1, 2, 3]


def test_damage_threat_score_imported_smoke() -> None:
    rng = default_rng(713)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    for p in my_team.members:
        s = _damage_threat_score(p, opp_team, PARAMS)
        assert isinstance(s, float)
