"""Unit tests for ``MetaThreatPairCoverageSelectionPolicy``.

The policy composes three single-axis improvements:
- pair-level joint scoring (vs singleton baseline)
- meta-weighted offense (vs uniform)
- worst-case max-of-pair-then-max-over-opp threat defense (vs mean)
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
    MatchupAwareSelectionPolicy,
    MetaThreatAwareSelectionPolicy,
    MetaThreatPairCoverageSelectionPolicy,
    PairCoverageSelectionPolicy,
    _meta_threat_pair_coverage_score,
    _threat_aware_pair_coverage_score,
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


def test_subclasses_meta_threat_aware() -> None:
    assert issubclass(MetaThreatPairCoverageSelectionPolicy, MetaThreatAwareSelectionPolicy)


def test_pair_scores_zero_for_empty_opp_team() -> None:
    a = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    b = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _meta_threat_pair_coverage_score(a, b, Team([]), [], PARAMS) == 0.0
    assert _threat_aware_pair_coverage_score(a, b, Team([]), PARAMS) == 0.0


def test_uniform_pair_score_uses_max_defense() -> None:
    a = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    b = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    fire_opp = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    score = _threat_aware_pair_coverage_score(a, b, Team([fire_opp]), PARAMS)
    assert math.isclose(score, -1.0, rel_tol=1e-9)


def test_uniform_pair_score_takes_max_of_pair_offense() -> None:
    a = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    b = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    grass_opp = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    score = _threat_aware_pair_coverage_score(a, b, Team([grass_opp]), PARAMS)
    assert math.isclose(score, 1.0, rel_tol=1e-9)


def test_meta_weighted_pair_score_uses_usage_for_offense() -> None:
    a = _mk_pkm(types=[Type.FIGHT], move_types=[Type.FIGHT])
    b = _mk_pkm(types=[Type.WATER], move_types=[Type.WATER])
    opp_normal = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    opp_fire = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    weights = [0.1, 0.9]
    score = _meta_threat_pair_coverage_score(a, b, Team([opp_normal, opp_fire]), weights, PARAMS)
    assert math.isclose(score, 1.0, rel_tol=1e-9)


def test_meta_weighted_pair_defense_is_max_not_weighted() -> None:
    grass = _mk_pkm(types=[Type.GRASS], move_types=[Type.GROUND])
    water = _mk_pkm(types=[Type.WATER], move_types=[Type.GROUND])
    fire_opp = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    normal_opp = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    weights_low_fire = [0.01, 0.99]
    score = _meta_threat_pair_coverage_score(
        grass, water, Team([fire_opp, normal_opp]), weights_low_fire, PARAMS
    )
    assert math.isclose(score, 1.01 - 2.0, rel_tol=1e-9)


def test_falls_back_to_pair_uniform_when_meta_none() -> None:
    rng = default_rng(901)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = MetaThreatPairCoverageSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    n = len(my_team.members)
    best_score = -float("inf")
    best_pair: tuple[int, int] = (0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            s = _threat_aware_pair_coverage_score(
                my_team.members[i], my_team.members[j], opp_team, PARAMS
            )
            if s > best_score:
                best_score = s
                best_pair = (i, j)
    assert set(cmd[:2]) == set(best_pair)


def test_falls_back_to_pair_uniform_at_epoch_zero() -> None:
    rng = default_rng(903)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    policy = MetaThreatPairCoverageSelectionPolicy()
    policy.set_meta(meta)
    cmd_meta = policy.decision((my_team, opp_team), 4)
    no_meta = MetaThreatPairCoverageSelectionPolicy()
    cmd_no_meta = no_meta.decision((my_team, opp_team), 4)
    assert set(cmd_meta[:2]) == set(cmd_no_meta[:2])


def test_singles_fallback_matches_meta_threat_aware_parent() -> None:
    fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    water = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    grass_opp = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    my_team = Team([water, fire])
    opp_team = Team([grass_opp])
    cmd = MetaThreatPairCoverageSelectionPolicy().decision((my_team, opp_team), 1)
    parent = MetaThreatAwareSelectionPolicy().decision((my_team, opp_team), 1)
    assert cmd == parent


def test_team_smaller_than_pair_falls_back() -> None:
    fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    grass_opp = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    my_team = Team([fire])
    opp_team = Team([grass_opp])
    cmd = MetaThreatPairCoverageSelectionPolicy().decision((my_team, opp_team), 2)
    parent = MetaThreatAwareSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == parent


def test_complementary_pair_leads_over_redundant_no_meta() -> None:
    fire_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    redundant_fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    water_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    opp_grass = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    opp_ground = _mk_pkm(types=[Type.GROUND], move_types=[Type.NORMAL])
    my_team = Team([fire_cover, redundant_fire, water_cover])
    opp_team = Team([opp_grass, opp_ground])
    cmd = MetaThreatPairCoverageSelectionPolicy().decision((my_team, opp_team), 3)
    assert set(cmd[:2]) == {0, 2}


def test_meta_shifts_lead_pair_to_high_usage_counter() -> None:
    a_counter_normal = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIGHT])
    b_counter_fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    c_counter_water = _mk_pkm(types=[Type.NORMAL], move_types=[Type.GRASS])
    opp_normal = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    opp_fire = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    my_team = Team([a_counter_normal, b_counter_fire, c_counter_water])
    opp_team = Team([opp_normal, opp_fire])
    opp_normal.species.id = 0
    opp_fire.species.id = 1
    meta = BasicMeta(move_set=[], roster=[opp_normal.species, opp_fire.species])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99]
    policy = MetaThreatPairCoverageSelectionPolicy()
    policy.set_meta(meta)
    cmd = policy.decision((my_team, opp_team), 3)
    assert 1 in cmd[:2], (
        "the fire-counter must be in the lead pair when fire is the dominant meta threat"
    )


def test_distinct_from_meta_threat_aware_singleton_on_redundancy() -> None:
    fire_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    redundant_fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    water_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    opp_grass = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    opp_ground = _mk_pkm(types=[Type.GROUND], move_types=[Type.NORMAL])
    my_team = Team([fire_cover, redundant_fire, water_cover])
    opp_team = Team([opp_grass, opp_ground])
    pair_cmd = MetaThreatPairCoverageSelectionPolicy().decision((my_team, opp_team), 3)
    singleton_cmd = MetaThreatAwareSelectionPolicy().decision((my_team, opp_team), 3)
    assert set(pair_cmd[:2]) == {0, 2}
    assert set(pair_cmd[:2]) != set(singleton_cmd[:2])


def test_distinct_from_pair_coverage_when_meta_present() -> None:
    a_counter_normal = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIGHT])
    b_counter_fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    c_counter_water = _mk_pkm(types=[Type.NORMAL], move_types=[Type.GRASS])
    opp_normal = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    opp_fire = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    my_team = Team([a_counter_normal, b_counter_fire, c_counter_water])
    opp_team = Team([opp_normal, opp_fire])
    opp_normal.species.id = 0
    opp_fire.species.id = 1
    meta = BasicMeta(move_set=[], roster=[opp_normal.species, opp_fire.species])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99]
    meta_policy = MetaThreatPairCoverageSelectionPolicy()
    meta_policy.set_meta(meta)
    pair_policy = PairCoverageSelectionPolicy()
    meta_cmd = meta_policy.decision((my_team, opp_team), 3)
    pair_cmd = pair_policy.decision((my_team, opp_team), 3)
    assert set(meta_cmd[:2]) != set(pair_cmd[:2]) or meta_cmd[0] != pair_cmd[0]


def test_falls_back_on_empty_opp_team() -> None:
    rng = default_rng(905)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = MetaThreatPairCoverageSelectionPolicy()
    policy.set_meta(BasicMeta(move_set=[], roster=[]))
    cmd = policy.decision((my_team, empty), 4)
    assert cmd[:2] == [0, 1]


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(907)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = MetaThreatPairCoverageSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(909)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = MetaThreatPairCoverageSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(911)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = MetaThreatPairCoverageSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_subclass_of_matchup_aware_chain() -> None:
    assert issubclass(MetaThreatPairCoverageSelectionPolicy, MatchupAwareSelectionPolicy)
