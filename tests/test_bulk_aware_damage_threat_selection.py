"""Unit tests for BulkAwareDamageThreatSelectionPolicy.

Bulk-aware refinement of DamageThreatSelectionPolicy: the primitive is
BP * STAB * type_eff * (atk / def_) / max_hp instead of BP * STAB *
type_eff / 100, so the scorer picks up bulk asymmetries (frail vs tank)
and stat-spread differences (ATK vs SPA, EV allocation) that every other
damage-aware scorer in the module collapses.
"""

from __future__ import annotations

import math

from numpy.random import default_rng
from vgc2.balance.meta import BasicMeta
from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.modifiers import Category, Nature, Stat, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies
from vgc2.battle_engine.team import Team
from vgc2.util.generator import gen_team

from vgc_ai.policies.selection import (
    BulkAwareDamageThreatSelectionPolicy,
    DamageThreatSelectionPolicy,
    MatchupAwareSelectionPolicy,
    _best_effective_damage,
    _bulk_aware_damage_threat_score,
)

PARAMS = BattleRuleParam()


def _mk_move(pkm_type: Type, base_power: int = 80, category: Category = Category.PHYSICAL) -> Move:
    return Move(
        pkm_type=pkm_type,
        base_power=base_power,
        accuracy=1.0,
        max_pp=10,
        category=category,
    )


def _mk_pkm(
    types: list[Type],
    moves: list[Move],
    base_stats: tuple[int, int, int, int, int, int] = (100, 100, 100, 100, 100, 100),
) -> Pokemon:
    species = PokemonSpecies(base_stats=base_stats, types=types, moves=moves)
    return Pokemon(
        species=species,
        move_indexes=list(range(len(moves))),
        evs=(85, 85, 85, 85, 85, 85),
        ivs=(31, 31, 31, 31, 31, 31),
        nature=Nature.SERIOUS,
    )


def test_subclasses_damage_threat() -> None:
    assert issubclass(BulkAwareDamageThreatSelectionPolicy, DamageThreatSelectionPolicy)


def test_subclasses_matchup_aware_chain() -> None:
    assert issubclass(BulkAwareDamageThreatSelectionPolicy, MatchupAwareSelectionPolicy)


def test_effective_damage_zero_for_status_only_kit() -> None:
    status_only = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=0)])
    target = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL)])
    assert _best_effective_damage(status_only, target, PARAMS) == 0.0


def test_effective_damage_scales_with_base_power() -> None:
    weak = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=40)])
    strong = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=120)])
    grass = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL)])
    weak_dmg = _best_effective_damage(weak, grass, PARAMS)
    strong_dmg = _best_effective_damage(strong, grass, PARAMS)
    max_hp = grass.stats[Stat.MAX_HP]
    assert math.isclose(weak_dmg, 40 * 2 / max_hp, rel_tol=1e-9)
    assert math.isclose(strong_dmg, 120 * 2 / max_hp, rel_tol=1e-9)
    assert strong_dmg > weak_dmg


def test_effective_damage_applies_stab() -> None:
    stab = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.FIRE, base_power=80)])
    non_stab = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=80)])
    grass = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL)])
    max_hp = grass.stats[Stat.MAX_HP]
    assert math.isclose(
        _best_effective_damage(stab, grass, PARAMS),
        80 * 1.5 * 2 / max_hp,
        rel_tol=1e-9,
    )
    assert math.isclose(
        _best_effective_damage(non_stab, grass, PARAMS),
        80 * 1.0 * 2 / max_hp,
        rel_tol=1e-9,
    )


def test_effective_damage_picks_physical_atk_def_pair() -> None:
    attacker = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.NORMAL, base_power=100, category=Category.PHYSICAL)],
        base_stats=(100, 200, 50, 50, 200, 100),
    )
    defender = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(100, 50, 50, 200, 200, 100),
    )
    dmg = _best_effective_damage(attacker, defender, PARAMS)
    atk = attacker.stats[Stat.ATTACK]
    def_ = defender.stats[Stat.DEFENSE]
    max_hp = defender.stats[Stat.MAX_HP]
    expected = 100 * 1.5 * 1.0 * (atk / def_) / max_hp
    assert math.isclose(dmg, expected, rel_tol=1e-9)


def test_effective_damage_picks_special_spa_spd_pair() -> None:
    attacker = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.NORMAL, base_power=100, category=Category.SPECIAL)],
        base_stats=(100, 50, 200, 200, 50, 100),
    )
    defender = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(100, 200, 200, 50, 50, 100),
    )
    dmg = _best_effective_damage(attacker, defender, PARAMS)
    spa = attacker.stats[Stat.SPECIAL_ATTACK]
    spd = defender.stats[Stat.SPECIAL_DEFENSE]
    max_hp = defender.stats[Stat.MAX_HP]
    expected = 100 * 1.5 * 1.0 * (spa / spd) / max_hp
    assert math.isclose(dmg, expected, rel_tol=1e-9)


def test_effective_damage_drops_with_defender_bulk() -> None:
    attacker = _mk_pkm(
        types=[Type.FIRE],
        moves=[_mk_move(Type.FIRE, base_power=100)],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    frail = _mk_pkm(
        types=[Type.GRASS],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(50, 100, 50, 100, 100, 100),
    )
    bulky = _mk_pkm(
        types=[Type.GRASS],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(200, 100, 200, 100, 100, 100),
    )
    frail_dmg = _best_effective_damage(attacker, frail, PARAMS)
    bulky_dmg = _best_effective_damage(attacker, bulky, PARAMS)
    assert frail_dmg > bulky_dmg


def test_score_zero_for_empty_opp_team() -> None:
    me = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL)])
    assert _bulk_aware_damage_threat_score(me, Team([]), PARAMS) == 0.0


def test_score_uses_max_for_defense() -> None:
    me = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=0)])
    weak_opp = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=30)])
    strong_opp = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.FIRE, base_power=110)])
    score = _bulk_aware_damage_threat_score(me, Team([weak_opp, strong_opp]), PARAMS)
    my_hp = me.stats[Stat.MAX_HP]
    expected = -(110 * 1.5 * 2 * 1.0 / my_hp)
    assert math.isclose(score, expected, rel_tol=1e-9)


def test_higher_bp_attacker_ranks_above_lower_bp_at_same_matchup() -> None:
    low_bp = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=40)])
    high_bp = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=110)])
    grass_opp = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=0)])
    my_team = Team([low_bp, high_bp])
    opp_team = Team([grass_opp])
    cmd = BulkAwareDamageThreatSelectionPolicy().decision((my_team, opp_team), 2)
    parent_cmd = MatchupAwareSelectionPolicy().decision((my_team, opp_team), 2)
    assert parent_cmd == [0, 1]
    assert cmd == [1, 0]


def test_higher_atk_lead_ranks_above_lower_atk_at_same_move() -> None:
    high_atk = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.FIRE, base_power=100, category=Category.PHYSICAL)],
        base_stats=(100, 200, 100, 100, 100, 100),
    )
    low_atk = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.FIRE, base_power=100, category=Category.PHYSICAL)],
        base_stats=(100, 50, 100, 100, 100, 100),
    )
    grass_opp = _mk_pkm(
        types=[Type.GRASS],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    my_team = Team([low_atk, high_atk])
    opp_team = Team([grass_opp])
    cmd = BulkAwareDamageThreatSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == [1, 0]


def test_physical_lead_outranks_special_against_phys_frail_opp() -> None:
    phys = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.NORMAL, base_power=100, category=Category.PHYSICAL)],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    spec = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.NORMAL, base_power=100, category=Category.SPECIAL)],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    defender = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(100, 100, 50, 100, 200, 100),
    )
    my_team = Team([spec, phys])
    opp_team = Team([defender])
    cmd = BulkAwareDamageThreatSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == [1, 0]


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(9001)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = BulkAwareDamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(9002)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = BulkAwareDamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(9003)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = BulkAwareDamageThreatSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_empty_opp_team_returns_stable_order() -> None:
    rng = default_rng(9005)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = BulkAwareDamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, empty), 4)
    assert cmd == [0, 1, 2, 3]


def test_meta_is_ignored() -> None:
    rng = default_rng(9007)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy_a = BulkAwareDamageThreatSelectionPolicy()
    policy_b = BulkAwareDamageThreatSelectionPolicy()
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99, 1, 1]
    policy_b.set_meta(meta)
    assert policy_a.decision((my_team, opp_team), 4) == policy_b.decision((my_team, opp_team), 4)
