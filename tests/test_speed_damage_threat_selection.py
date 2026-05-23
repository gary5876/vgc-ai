"""Unit tests for SpeedDamageThreatSelectionPolicy.

Composes:
- damage-aware primitive (base_power * STAB * type_eff / _BP_NORMALIZER)
  and max-threat defense (from DamageThreatSelectionPolicy)
- per-opp speed-tier initiative bonus (from SpeedTierAwareSelectionPolicy)
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
    _BP_NORMALIZER,
    _INITIATIVE_BONUS,
    DamageThreatSelectionPolicy,
    MatchupAwareSelectionPolicy,
    SpeedDamageThreatSelectionPolicy,
    SpeedTierAwareSelectionPolicy,
    _speed_damage_threat_score,
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
    assert issubclass(SpeedDamageThreatSelectionPolicy, DamageThreatSelectionPolicy)


def test_subclasses_matchup_aware_chain() -> None:
    assert issubclass(SpeedDamageThreatSelectionPolicy, MatchupAwareSelectionPolicy)


def test_score_zero_for_empty_opp_team() -> None:
    me = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL)])
    assert _speed_damage_threat_score(me, Team([]), PARAMS) == 0.0


def test_score_pure_damage_when_speeds_tie() -> None:
    me = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=100)])
    opp = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=0)])
    score = _speed_damage_threat_score(me, Team([opp]), PARAMS)
    assert math.isclose(score, 100.0 * 1.5 / _BP_NORMALIZER, rel_tol=1e-9)


def test_outspeed_adds_initiative_bonus_to_score() -> None:
    fast = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(100, 100, 100, 100, 100, 200),
    )
    opp = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    score = _speed_damage_threat_score(fast, Team([opp]), PARAMS)
    assert math.isclose(score, _INITIATIVE_BONUS, rel_tol=1e-9)


def test_outsped_subtracts_initiative_bonus() -> None:
    slow = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(100, 100, 100, 100, 100, 50),
    )
    opp = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(100, 100, 100, 100, 100, 150),
    )
    score = _speed_damage_threat_score(slow, Team([opp]), PARAMS)
    assert math.isclose(score, -_INITIATIVE_BONUS, rel_tol=1e-9)


def test_score_uses_max_for_defense() -> None:
    me = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=0)])
    weak_opp = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.NORMAL, base_power=30)])
    strong_opp = _mk_pkm(types=[Type.FIRE], moves=[_mk_move(Type.FIRE, base_power=110)])
    score = _speed_damage_threat_score(me, Team([weak_opp, strong_opp]), PARAMS)
    expected_defense = 110.0 * 1.5 * 2.0 / _BP_NORMALIZER
    assert math.isclose(score, -expected_defense, rel_tol=1e-9)


def test_speed_breaks_tie_between_equal_damage_leads() -> None:
    fast = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.FIRE, base_power=80)],
        base_stats=(100, 100, 100, 100, 100, 200),
    )
    slow = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.FIRE, base_power=80)],
        base_stats=(100, 100, 100, 100, 100, 50),
    )
    grass = _mk_pkm(
        types=[Type.GRASS],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    my_team = Team([slow, fast])
    opp_team = Team([grass])
    damage_cmd = DamageThreatSelectionPolicy().decision((my_team, opp_team), 2)
    speed_dmg_cmd = SpeedDamageThreatSelectionPolicy().decision((my_team, opp_team), 2)
    assert damage_cmd == [0, 1]
    assert speed_dmg_cmd == [1, 0]


def test_high_bp_lead_beats_low_bp_lead_under_speed_damage() -> None:
    low = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=40)])
    high = _mk_pkm(types=[Type.NORMAL], moves=[_mk_move(Type.FIRE, base_power=110)])
    grass = _mk_pkm(types=[Type.GRASS], moves=[_mk_move(Type.NORMAL, base_power=0)])
    my_team = Team([low, high])
    opp_team = Team([grass])
    speed_cmd = SpeedTierAwareSelectionPolicy().decision((my_team, opp_team), 1)
    speed_dmg_cmd = SpeedDamageThreatSelectionPolicy().decision((my_team, opp_team), 1)
    assert speed_cmd == [0]
    assert speed_dmg_cmd == [1]


def test_initiative_bonus_does_not_override_type_advantage() -> None:
    slow_fire = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.FIRE, base_power=80)],
        base_stats=(100, 100, 100, 100, 100, 50),
    )
    fast_water = _mk_pkm(
        types=[Type.NORMAL],
        moves=[_mk_move(Type.WATER, base_power=80)],
        base_stats=(100, 100, 100, 100, 100, 200),
    )
    grass = _mk_pkm(
        types=[Type.GRASS],
        moves=[_mk_move(Type.NORMAL, base_power=0)],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    my_team = Team([slow_fire, fast_water])
    opp_team = Team([grass])
    cmd = SpeedDamageThreatSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == [0, 1]


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(8001)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = SpeedDamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(8002)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = SpeedDamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(8003)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = SpeedDamageThreatSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_empty_opp_team_returns_stable_index_order() -> None:
    rng = default_rng(8005)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = SpeedDamageThreatSelectionPolicy()
    cmd = policy.decision((my_team, empty), 4)
    assert cmd == [0, 1, 2, 3]


def test_meta_is_ignored() -> None:
    rng = default_rng(8007)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy_a = SpeedDamageThreatSelectionPolicy()
    policy_b = SpeedDamageThreatSelectionPolicy()
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99, 1, 1]
    policy_b.set_meta(meta)
    assert policy_a.decision((my_team, opp_team), 4) == policy_b.decision((my_team, opp_team), 4)


def test_recovers_damage_threat_when_all_speeds_equal() -> None:
    rng = default_rng(8011)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    for p in list(my_team.members) + list(opp_team.members):
        stats = list(p.stats)
        stats[Stat.SPEED] = 100
        p.stats = type(p.stats)(stats)
    speed_dmg = SpeedDamageThreatSelectionPolicy().decision((my_team, opp_team), 4)
    dmg = DamageThreatSelectionPolicy().decision((my_team, opp_team), 4)
    assert speed_dmg == dmg
