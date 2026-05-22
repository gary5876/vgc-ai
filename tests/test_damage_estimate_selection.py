"""Unit tests for ``DamageEstimateSelectionPolicy``."""

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
    DamageEstimateSelectionPolicy,
    MatchupAwareSelectionPolicy,
    _best_damage_estimate_multiplier,
    _damage_estimate_selection_score,
)

PARAMS = BattleRuleParam()


def _mk_move(
    pkm_type: Type,
    base_power: int = 100,
    accuracy: float = 1.0,
    category: Category = Category.PHYSICAL,
) -> Move:
    return Move(
        pkm_type=pkm_type,
        base_power=base_power,
        accuracy=accuracy,
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
    assert issubclass(DamageEstimateSelectionPolicy, MatchupAwareSelectionPolicy)


def test_reference_move_matches_type_chart_baseline() -> None:
    # 100 BP, 100% acc, non-STAB, neutral vs NORMAL: multiplier should be 1.0.
    me = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.GRASS, base_power=100)])
    opp = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL)])
    m = _best_damage_estimate_multiplier(me, opp, PARAMS)
    assert math.isclose(m, 1.0, rel_tol=1e-9)


def test_base_power_scales_multiplier() -> None:
    fire_60 = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.FIRE, base_power=60)])
    fire_130 = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.FIRE, base_power=130)])
    grass = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL)])
    low = _best_damage_estimate_multiplier(fire_60, grass, PARAMS)
    high = _best_damage_estimate_multiplier(fire_130, grass, PARAMS)
    assert math.isclose(low, 0.60 * 1.5 * 2.0, rel_tol=1e-9)
    assert math.isclose(high, 1.30 * 1.5 * 2.0, rel_tol=1e-9)
    assert high > low


def test_accuracy_scales_multiplier() -> None:
    perfect = _mk_pkm(
        types=[Type.FIRE],
        moves=[_mk_move(Type.FIRE, base_power=100, accuracy=1.0)],
    )
    shaky = _mk_pkm(
        types=[Type.FIRE],
        moves=[_mk_move(Type.FIRE, base_power=100, accuracy=0.5)],
    )
    grass = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL)])
    full = _best_damage_estimate_multiplier(perfect, grass, PARAMS)
    half = _best_damage_estimate_multiplier(shaky, grass, PARAMS)
    assert math.isclose(full, 1.0 * 1.0 * 1.5 * 2.0, rel_tol=1e-9)
    assert math.isclose(half, 1.0 * 0.5 * 1.5 * 2.0, rel_tol=1e-9)
    assert full > half


def test_stab_folds_into_multiplier() -> None:
    fire = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.FIRE, base_power=100)])
    normal = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL)])
    m = _best_damage_estimate_multiplier(fire, normal, PARAMS)
    assert math.isclose(m, 1.5, rel_tol=1e-9)


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
    me = _mk_pkm(types=[Type.NORMAL], moves=moves)
    opp = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL)])
    assert math.isclose(_best_damage_estimate_multiplier(me, opp, PARAMS), 1.0, rel_tol=1e-9)


def test_damage_estimate_score_empty_opp_team_is_zero() -> None:
    me = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL)])
    assert _damage_estimate_selection_score(me, Team([]), PARAMS) == 0.0


def test_base_power_breaks_tie_between_same_type_matchup() -> None:
    weak_fire = _mk_pkm(
        types=[Type.FIRE],
        moves=[_mk_move(Type.FIRE, base_power=60)],
    )
    strong_fire = _mk_pkm(
        types=[Type.FIRE],
        moves=[_mk_move(Type.FIRE, base_power=130)],
    )
    grass = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=100)])
    my_team = Team([weak_fire, strong_fire])
    opp_team = Team([grass])
    parent_cmd = MatchupAwareSelectionPolicy().decision((my_team, opp_team), 2)
    assert parent_cmd == [0, 1]
    cmd = DamageEstimateSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == [1, 0]


def test_low_accuracy_high_bp_loses_to_perfect_lower_bp() -> None:
    shaky = _mk_pkm(
        types=[Type.FIRE],
        moves=[_mk_move(Type.FIRE, base_power=130, accuracy=0.5)],
    )
    reliable = _mk_pkm(
        types=[Type.FIRE],
        moves=[_mk_move(Type.FIRE, base_power=100, accuracy=1.0)],
    )
    grass = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=100)])
    my_team = Team([shaky, reliable])
    opp_team = Team([grass])
    cmd = DamageEstimateSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == [1, 0]


def test_opp_high_bp_stab_penalises_defense() -> None:
    grass_lead = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL)])
    water_lead = _mk_pkm(types=[Type.WATER], moves=[_mk_move(Type.NORMAL)])
    fire_opp = _mk_pkm(
        types=[Type.FIRE],
        moves=[_mk_move(Type.FIRE, base_power=130)],
    )
    my_team = Team([grass_lead, water_lead])
    opp_team = Team([fire_opp])
    cmd = DamageEstimateSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == [1, 0]


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(4242)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = DamageEstimateSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(4243)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = DamageEstimateSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(4244)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = DamageEstimateSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_meta_is_ignored() -> None:
    from vgc2.balance.meta import BasicMeta

    rng = default_rng(4245)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy_a = DamageEstimateSelectionPolicy()
    policy_b = DamageEstimateSelectionPolicy()
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99, 1, 1]
    policy_b.set_meta(meta)
    assert policy_a.decision((my_team, opp_team), 4) == policy_b.decision((my_team, opp_team), 4)


def test_empty_opp_team_returns_stable_order() -> None:
    rng = default_rng(4246)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = DamageEstimateSelectionPolicy()
    cmd = policy.decision((my_team, empty), 4)
    assert cmd == [0, 1, 2, 3]


def test_collapses_to_parent_at_reference_scalars() -> None:
    a = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.FIRE, base_power=100, accuracy=1.0)],
    )
    b = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.WATER, base_power=100, accuracy=1.0)],
    )
    c = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.GRASS, base_power=100, accuracy=1.0)],
    )
    opp1 = _mk_pkm(
        types=[Type.FIRE],
        moves=[_mk_move(Type.WATER, base_power=100, accuracy=1.0)],
    )
    opp2 = _mk_pkm(
        types=[Type.GRASS],
        moves=[_mk_move(Type.FIRE, base_power=100, accuracy=1.0)],
    )
    my_team = Team([a, b, c])
    opp_team = Team([opp1, opp2])
    new_cmd = DamageEstimateSelectionPolicy().decision((my_team, opp_team), 3)
    parent_cmd = MatchupAwareSelectionPolicy().decision((my_team, opp_team), 3)
    assert new_cmd == parent_cmd
