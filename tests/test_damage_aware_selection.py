"""Unit tests for DamageAwareSelectionPolicy."""

from __future__ import annotations

from numpy.random import default_rng
from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.modifiers import Category, Nature, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies
from vgc2.battle_engine.team import Team
from vgc2.util.generator import gen_team

from vgc_ai.policies.selection import (
    DamageAwareSelectionPolicy,
    MatchupAwareSelectionPolicy,
    _damage_aware_score,
    _damage_fraction,
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
    move_types: list[Type],
    base_stats: tuple[int, int, int, int, int, int] = (100, 100, 100, 100, 100, 100),
    move_category: Category = Category.PHYSICAL,
) -> Pokemon:
    moves = [_mk_move(t, category=move_category) for t in move_types]
    species = PokemonSpecies(
        base_stats=base_stats,
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
    assert issubclass(DamageAwareSelectionPolicy, MatchupAwareSelectionPolicy)


def test_damage_fraction_zero_for_status_only_attacker() -> None:
    attacker = _mk_pkm(types=[Type.NORMAL], move_types=[])
    attacker.moves = [
        Move(
            pkm_type=Type.WATER,
            base_power=0,
            accuracy=1.0,
            max_pp=10,
            category=Category.OTHER,
        )
    ]
    defender = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _damage_fraction(attacker, defender, PARAMS) == 0.0


def test_damage_fraction_skips_other_category_moves() -> None:
    attacker = _mk_pkm(types=[Type.NORMAL], move_types=[])
    attacker.moves = [
        Move(
            pkm_type=Type.NORMAL,
            base_power=80,
            accuracy=1.0,
            max_pp=10,
            category=Category.OTHER,
        )
    ]
    defender = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _damage_fraction(attacker, defender, PARAMS) == 0.0


def test_damage_fraction_super_effective_beats_neutral_at_equal_stats() -> None:
    fire_attacker = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    normal_attacker = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    grass_defender = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    fire_df = _damage_fraction(fire_attacker, grass_defender, PARAMS)
    normal_df = _damage_fraction(normal_attacker, grass_defender, PARAMS)
    assert fire_df > normal_df


def test_damage_fraction_bulky_defender_takes_less_than_frail() -> None:
    attacker = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    frail = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(40, 100, 40, 100, 40, 100),
    )
    bulky = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(200, 100, 200, 100, 200, 100),
    )
    df_frail = _damage_fraction(attacker, frail, PARAMS)
    df_bulky = _damage_fraction(attacker, bulky, PARAMS)
    assert df_frail > df_bulky


def test_damage_fraction_high_attack_beats_low_attack() -> None:
    high_atk = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 200, 100, 100, 100, 100),
    )
    low_atk = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 40, 100, 100, 100, 100),
    )
    defender = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    df_high = _damage_fraction(high_atk, defender, PARAMS)
    df_low = _damage_fraction(low_atk, defender, PARAMS)
    assert df_high > df_low


def test_damage_fraction_picks_best_move() -> None:
    attacker = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL, Type.FIRE])
    grass_defender = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    df = _damage_fraction(attacker, grass_defender, PARAMS)
    only_normal = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    df_normal = _damage_fraction(only_normal, grass_defender, PARAMS)
    assert df >= df_normal


def test_score_zero_for_empty_opp_team() -> None:
    pkm = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _damage_aware_score(pkm, Team([]), PARAMS) == 0.0


def test_super_effective_attacker_ranks_above_neutral() -> None:
    fire_attacker = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    normal_attacker = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    grass_target = _mk_pkm(types=[Type.GRASS], move_types=[Type.GRASS])
    my_team = Team([normal_attacker, fire_attacker])
    opp_team = Team([grass_target])
    cmd = DamageAwareSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd[0] == 1, "fire (super-effective vs grass) should lead"


def test_high_atk_beats_super_effective_low_atk_at_extreme_stats() -> None:
    # Load-bearing test for the policy's novel angle: at extreme stat
    # differences, raw damage potential flips the ranking the type-chart
    # proxy would assign.
    weak_super = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.FIRE],
        base_stats=(50, 30, 50, 30, 50, 50),
    )
    strong_neutral = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(150, 250, 150, 250, 150, 150),
    )
    grass_defender = _mk_pkm(
        types=[Type.GRASS],
        move_types=[Type.NORMAL],
        base_stats=(200, 100, 200, 100, 200, 100),
    )
    df_weak = _damage_fraction(weak_super, grass_defender, PARAMS)
    df_strong = _damage_fraction(strong_neutral, grass_defender, PARAMS)
    assert df_strong > df_weak


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(901)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = DamageAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(903)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = DamageAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_empty_opp_team_returns_stable_order() -> None:
    rng = default_rng(905)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = DamageAwareSelectionPolicy()
    cmd = policy.decision((my_team, empty), 4)
    assert cmd == [0, 1, 2, 3]


def test_deterministic_across_calls() -> None:
    rng = default_rng(907)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = DamageAwareSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_meta_is_ignored() -> None:
    from vgc2.balance.meta import BasicMeta

    rng = default_rng(909)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy_a = DamageAwareSelectionPolicy()
    policy_b = DamageAwareSelectionPolicy()
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99, 1, 1]
    policy_b.set_meta(meta)
    assert policy_a.decision((my_team, opp_team), 4) == policy_b.decision((my_team, opp_team), 4)
