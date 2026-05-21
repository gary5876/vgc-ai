"""Unit tests for PairCoverageSelectionPolicy."""

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
    PairCoverageSelectionPolicy,
    _pair_coverage_score,
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
) -> Pokemon:
    moves = [_mk_move(t) for t in move_types]
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
    assert issubclass(PairCoverageSelectionPolicy, MatchupAwareSelectionPolicy)


def test_pair_coverage_score_empty_opp_team_is_zero() -> None:
    a = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    b = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _pair_coverage_score(a, b, Team([]), PARAMS) == 0.0


def test_pair_coverage_score_takes_max_of_pair_offense() -> None:
    # Fire move on member A (2x vs Grass), Water move on member B (0.5x vs Grass).
    # Best-of-pair offense vs a Grass opp comes from A.
    a = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    b = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    grass_opp = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    score = _pair_coverage_score(a, b, Team([grass_opp]), PARAMS)
    # offense max = 2.0 (Fire>Grass), defense max = 1.0 (Normal>Normal both sides)
    assert math.isclose(score, 1.0, rel_tol=1e-9)


def test_pair_coverage_score_uses_worst_case_defense() -> None:
    # Member A is 2x weak to Fire (Grass type); member B is 1x to Fire.
    # The pair pays the worst-case threat -- even though B alone is safe.
    a = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    b = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    fire_opp = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    score = _pair_coverage_score(a, b, Team([fire_opp]), PARAMS)
    # offense max = 1.0 (Normal>Normal), defense max = 2.0 (Fire>Grass)
    assert math.isclose(score, -1.0, rel_tol=1e-9)


def test_pair_coverage_prefers_complementary_pair() -> None:
    # Two single-type covers vs a redundant duplicate against a Fire/Water opp.
    fire_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    water_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    redundant_fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    opp_grass = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    opp_ground = _mk_pkm(types=[Type.GROUND], move_types=[Type.NORMAL])
    opp_team = Team([opp_grass, opp_ground])
    pair_complement = _pair_coverage_score(fire_cover, water_cover, opp_team, PARAMS)
    pair_redundant = _pair_coverage_score(fire_cover, redundant_fire, opp_team, PARAMS)
    assert pair_complement > pair_redundant


def test_complementary_pair_leads_over_redundant() -> None:
    # A team with two same-type covers and one complementary cover. The
    # complementary pair should be picked as leads.
    fire_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    redundant_fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    water_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    opp_grass = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    opp_ground = _mk_pkm(types=[Type.GROUND], move_types=[Type.NORMAL])
    opp_team = Team([opp_grass, opp_ground])
    my_team = Team([fire_cover, redundant_fire, water_cover])
    cmd = PairCoverageSelectionPolicy().decision((my_team, opp_team), 3)
    assert set(cmd[:2]) == {0, 2}


def test_n_active_fallback_to_singleton_ranker() -> None:
    # max_size < 2 (singles regime) delegates to the parent ranker.
    fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    water = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    grass_opp = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    my_team = Team([water, fire])
    opp_team = Team([grass_opp])
    cmd = PairCoverageSelectionPolicy().decision((my_team, opp_team), 1)
    parent = MatchupAwareSelectionPolicy().decision((my_team, opp_team), 1)
    assert cmd == parent


def test_team_smaller_than_pair_falls_back() -> None:
    # n < 2 cannot form a pair, falls back to parent ranker.
    fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    grass_opp = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    my_team = Team([fire])
    opp_team = Team([grass_opp])
    cmd = PairCoverageSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == [0]


def test_meta_is_ignored() -> None:
    from vgc2.balance.meta import BasicMeta

    rng = default_rng(2026)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy_a = PairCoverageSelectionPolicy()
    policy_b = PairCoverageSelectionPolicy()
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99, 1, 1]
    policy_b.set_meta(meta)
    assert policy_a.decision((my_team, opp_team), 4) == policy_b.decision((my_team, opp_team), 4)


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(2027)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = PairCoverageSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(2028)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = PairCoverageSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_empty_opp_team_returns_stable_order() -> None:
    rng = default_rng(2029)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = PairCoverageSelectionPolicy()
    cmd = policy.decision((my_team, empty), 4)
    # All pair / singleton scores tie at 0.0 -> stable index tiebreak.
    assert cmd[:2] == [0, 1]


def test_deterministic_across_calls() -> None:
    rng = default_rng(2030)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = PairCoverageSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_distinct_from_matchup_aware_on_complementary_team() -> None:
    # The existence of any test team where pair-coverage diverges from the
    # singleton parent proves the policy carries a different signal.
    fire_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    redundant_fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    water_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    opp_grass = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    opp_ground = _mk_pkm(types=[Type.GROUND], move_types=[Type.NORMAL])
    my_team = Team([fire_cover, redundant_fire, water_cover])
    opp_team = Team([opp_grass, opp_ground])
    pair_cmd = PairCoverageSelectionPolicy().decision((my_team, opp_team), 3)
    parent_cmd = MatchupAwareSelectionPolicy().decision((my_team, opp_team), 3)
    assert pair_cmd[:2] != parent_cmd[:2]
    assert set(pair_cmd[:2]) == {0, 2}
