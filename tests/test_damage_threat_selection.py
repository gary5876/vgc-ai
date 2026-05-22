"""Unit tests for ``DamageThreatSelectionPolicy``.

The policy refines ``MatchupAwareSelectionPolicy`` along the move-power
axis: replaces the type-effectiveness multiplier with
``(base_power * STAB * type_eff) / _BP_NORMALIZER`` so high-BP moves
out-rank low-BP moves at the same type matchup, and pairs that with
worst-case (max) threat defense (the current default's defense shape).
"""

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
    _BP_NORMALIZER,
    DamageThreatSelectionPolicy,
    MatchupAwareSelectionPolicy,
    _best_damage_potential,
    _damage_threat_score,
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


def test_subclasses_matchup_aware() -> None:
    assert issubclass(DamageThreatSelectionPolicy, MatchupAwareSelectionPolicy)


def test_damage_potential_zero_for_status_only_kit() -> None:
    status_only = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=0)])
    target = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL)])
    assert _best_damage_potential(status_only, target, PARAMS) == 0.0


def test_damage_potential_scales_with_base_power() -> None:
    weak = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=40)])
    strong = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=120)])
    grass = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL)])
    weak_dmg = _best_damage_potential(weak, grass, PARAMS)
    strong_dmg = _best_damage_potential(strong, grass, PARAMS)
    assert math.isclose(weak_dmg, 40 * 2 / _BP_NORMALIZER, rel_tol=1e-9)
    assert math.isclose(strong_dmg, 120 * 2 / _BP_NORMALIZER, rel_tol=1e-9)
    assert strong_dmg > weak_dmg


def test_damage_potential_applies_stab() -> None:
    stab = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.FIRE, base_power=80)])
    non_stab = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=80)])
    grass = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL)])
    assert math.isclose(
        _best_damage_potential(stab, grass, PARAMS),
        80 * 1.5 * 2 / _BP_NORMALIZER,
        rel_tol=1e-9,
    )
    assert math.isclose(
        _best_damage_potential(non_stab, grass, PARAMS),
        80 * 1.0 * 2 / _BP_NORMALIZER,
        rel_tol=1e-9,
    )


def test_damage_potential_picks_best_move() -> None:
    attacker = _mk_pkm(
        types=[Type.NORMAL],
        moves=[
            _mk_move(Type.NORMAL, base_power=100),
            _mk_move(Type.WATER, base_power=80),
        ],
    )
    fire = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.NORMAL)])
    assert math.isclose(
        _best_damage_potential(attacker, fire, PARAMS), 160 / _BP_NORMALIZER, rel_tol=1e-9
    )


def test_score_zero_for_empty_opp_team() -> None:
    me = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL)])
    assert _damage_threat_score(me, Team([]), PARAMS) == 0.0


def test_score_uses_max_for_defense() -> None:
    me = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=100)])
    threat = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=110)])
    weak = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=30)])
    score = _damage_threat_score(me, Team([threat, weak]), PARAMS)
    assert math.isclose(score, 1.0 - 2.2, rel_tol=1e-9)


def test_score_uses_mean_for_offense() -> None:
    me = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=100)])
    grass = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=0)])
    water = _mk_pkm(types=[Type.WATER], moves=[_mk_move(Type.NORMAL, base_power=0)])
    score = _damage_threat_score(me, Team([grass, water]), PARAMS)
    assert math.isclose(score, 1.25, rel_tol=1e-9)


def test_higher_bp_attacker_ranks_above_lower_bp_at_same_matchup() -> None:
    low_bp = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=40)])
    high_bp = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=110)])
    grass_opp = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=0)])
    my_team = Team([low_bp, high_bp])
    opp_team = Team([grass_opp])
    damage_cmd = DamageThreatSelectionPolicy().decision((my_team, opp_team), 2)
    parent_cmd = MatchupAwareSelectionPolicy().decision((my_team, opp_team), 2)
    assert parent_cmd == [0, 1]
    assert damage_cmd == [1, 0]


def test_low_bp_se_loses_to_high_bp_neutral_when_bp_dominates() -> None:
    weak_se = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=30)])
    strong_neutral = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=120)])
    grass_opp = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=0)])
    my_team = Team([weak_se, strong_neutral])
    opp_team = Team([grass_opp])
    damage_cmd = DamageThreatSelectionPolicy().decision((my_team, opp_team), 2)
    parent_cmd = MatchupAwareSelectionPolicy().decision((my_team, opp_team), 2)
    assert parent_cmd == [0, 1]
    assert damage_cmd == [1, 0]


def test_max_threat_defense_dominates_average() -> None:
    a_weak = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=80)])
    b_neutral = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=80)])
    fire_threat = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=100)])
    filler = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=80)])
    my_team = Team([a_weak, b_neutral])
    opp_team = Team([fire_threat, filler])
    cmd = DamageThreatSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == [1, 0]


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(3001)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = DamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(3002)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = DamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(3003)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = DamageThreatSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_empty_opp_team_returns_stable_order() -> None:
    rng = default_rng(3005)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = DamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, empty), 4)
    assert cmd == [0, 1, 2, 3]


def test_meta_is_ignored() -> None:
    from vgc2.balance.meta import BasicMeta

    rng = default_rng(3007)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy_a = DamageThreatSelectionPolicy()
    policy_b = DamageThreatSelectionPolicy()
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99, 1, 1]
    policy_b.set_meta(meta)
    assert policy_a.decision((my_team, opp_team), 4) == policy_b.decision((my_team, opp_team), 4)
