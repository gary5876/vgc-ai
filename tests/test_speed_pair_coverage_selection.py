"""Unit tests for SpeedPairCoverageSelectionPolicy."""

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
    _INITIATIVE_BONUS,
    MatchupAwareSelectionPolicy,
    PairCoverageSelectionPolicy,
    SpeedPairCoverageSelectionPolicy,
    SpeedTierAwareSelectionPolicy,
    _pair_coverage_score,
    _speed_pair_coverage_score,
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
    assert issubclass(SpeedPairCoverageSelectionPolicy, MatchupAwareSelectionPolicy)


def test_score_empty_opp_team_is_zero() -> None:
    a = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    b = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _speed_pair_coverage_score(a, b, Team([]), PARAMS) == 0.0


def test_speed_tie_reduces_to_pair_coverage() -> None:
    # All speeds equal => initiative term is 0 per opp => score equals
    # the pair-coverage parent exactly.
    a = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    b = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    grass_opp = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    ground_opp = _mk_pkm(types=[Type.GROUND], move_types=[Type.NORMAL])
    opp_team = Team([grass_opp, ground_opp])
    pair = _pair_coverage_score(a, b, opp_team, PARAMS)
    combined = _speed_pair_coverage_score(a, b, opp_team, PARAMS)
    assert math.isclose(pair, combined, rel_tol=1e-9)


def test_both_leads_outspeed_doubles_initiative_bonus() -> None:
    # Both leads have neutral type matchup vs opp; both outspeed => each
    # contributes +_INITIATIVE_BONUS. Averaged over a single opp => 2 * bonus.
    a = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 200),
    )
    b = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 200),
    )
    opp = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    score = _speed_pair_coverage_score(a, b, Team([opp]), PARAMS)
    # offense - defense = 0 (neutral both ways), initiative = 2 * bonus.
    assert math.isclose(score, 2.0 * _INITIATIVE_BONUS, rel_tol=1e-9)


def test_mixed_speed_pair_nets_zero_initiative() -> None:
    # One faster + one slower vs the same opp -> bonuses cancel.
    fast = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 200),
    )
    slow = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 50),
    )
    opp = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    score = _speed_pair_coverage_score(fast, slow, Team([opp]), PARAMS)
    assert math.isclose(score, 0.0, abs_tol=1e-9)


def test_prefers_complementary_pair_like_pair_coverage() -> None:
    # All speeds tie so the initiative term is zero -- the pair argmax
    # must still prefer the complementary (Fire+Water) pair over the
    # redundant (Fire+Fire) pair against a Fire/Water-weak opp duo.
    fire_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    redundant_fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    water_cover = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    grass_opp = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    ground_opp = _mk_pkm(types=[Type.GROUND], move_types=[Type.NORMAL])
    my_team = Team([fire_cover, redundant_fire, water_cover])
    opp_team = Team([grass_opp, ground_opp])
    cmd = SpeedPairCoverageSelectionPolicy().decision((my_team, opp_team), 3)
    assert set(cmd[:2]) == {0, 2}


def test_speed_breaks_tie_within_chosen_pair() -> None:
    # Two leads with identical type matchups but different speed. The
    # faster lead takes the very first slot (within-pair ordering uses
    # _speed_tier_score, which prefers the outspeeder).
    fast = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 200),
    )
    slow = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 50),
    )
    opp = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    my_team = Team([slow, fast])
    opp_team = Team([opp])
    speed_cmd = SpeedPairCoverageSelectionPolicy().decision((my_team, opp_team), 2)
    pair_cmd = PairCoverageSelectionPolicy().decision((my_team, opp_team), 2)
    # Pair-coverage parent: index tiebreak -> [0, 1] = [slow, fast].
    # Speed-pair-coverage: fast takes first slot -> [1, 0].
    assert pair_cmd == [0, 1]
    assert speed_cmd == [1, 0]


def test_n_active_fallback_uses_speed_tier_singleton() -> None:
    # max_size < 2 (singles regime) should rank by speed-tier singleton,
    # not by the pair-coverage parent's matchup-aware singleton.
    fast = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 200),
    )
    slow = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 50),
    )
    opp = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    my_team = Team([slow, fast])
    opp_team = Team([opp])
    sp_cmd = SpeedPairCoverageSelectionPolicy().decision((my_team, opp_team), 1)
    st_cmd = SpeedTierAwareSelectionPolicy().decision((my_team, opp_team), 1)
    parent_cmd = MatchupAwareSelectionPolicy().decision((my_team, opp_team), 1)
    # Speed-tier prefers fast; matchup-aware parent ties at 0 and breaks by index.
    assert st_cmd == [1]
    assert parent_cmd == [0]
    assert sp_cmd == [1]


def test_team_smaller_than_pair_falls_back() -> None:
    # n < 2 cannot form a pair, falls back to singleton ranker.
    fire = _mk_pkm(types=[Type.NORMAL], move_types=[Type.FIRE])
    grass_opp = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    my_team = Team([fire])
    opp_team = Team([grass_opp])
    cmd = SpeedPairCoverageSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd == [0]


def test_meta_is_ignored() -> None:
    from vgc2.balance.meta import BasicMeta

    rng = default_rng(4096)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy_a = SpeedPairCoverageSelectionPolicy()
    policy_b = SpeedPairCoverageSelectionPolicy()
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99, 1, 1]
    policy_b.set_meta(meta)
    assert policy_a.decision((my_team, opp_team), 4) == policy_b.decision((my_team, opp_team), 4)


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(4097)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = SpeedPairCoverageSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(4098)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = SpeedPairCoverageSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(4099)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = SpeedPairCoverageSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_empty_opp_team_returns_stable_order() -> None:
    rng = default_rng(4100)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = SpeedPairCoverageSelectionPolicy()
    cmd = policy.decision((my_team, empty), 4)
    # All pair / singleton scores tie at 0.0 -> stable index tiebreak.
    assert cmd[:2] == [0, 1]
