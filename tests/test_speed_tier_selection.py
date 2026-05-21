"""Unit tests for SpeedTierAwareSelectionPolicy."""

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
    SpeedTierAwareSelectionPolicy,
    _speed_tier_score,
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
    assert issubclass(SpeedTierAwareSelectionPolicy, MatchupAwareSelectionPolicy)


def test_speed_tier_score_empty_opp_team_is_zero() -> None:
    me = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _speed_tier_score(me, Team([]), PARAMS) == 0.0


def test_outspeed_adds_initiative_bonus() -> None:
    # Identical type matchup (neutral both ways) -- the only differentiator
    # between fast and slow is the speed-tier initiative term.
    fast = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 200),
    )
    opp = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    score = _speed_tier_score(fast, Team([opp]), PARAMS)
    # offense - defense = 0 (neutral both ways), initiative = +_INITIATIVE_BONUS.
    assert math.isclose(score, _INITIATIVE_BONUS, rel_tol=1e-9)


def test_outsped_subtracts_initiative_bonus() -> None:
    slow = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 50),
    )
    opp = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 150),
    )
    score = _speed_tier_score(slow, Team([opp]), PARAMS)
    assert math.isclose(score, -_INITIATIVE_BONUS, rel_tol=1e-9)


def test_speed_tie_is_neutral() -> None:
    me = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    opp = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    score = _speed_tier_score(me, Team([opp]), PARAMS)
    assert math.isclose(score, 0.0, abs_tol=1e-9)


def test_initiative_bonus_does_not_override_type_advantage() -> None:
    # Slow Fire-mover (-bonus initiative) vs Grass opp should still rank
    # higher than fast Water-mover (+bonus initiative) because the type
    # multiplier dominates the initiative term.
    slow_fire = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.FIRE],
        base_stats=(100, 100, 100, 100, 100, 50),
    )
    fast_water = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.WATER],
        base_stats=(100, 100, 100, 100, 100, 200),
    )
    grass_opp = _mk_pkm(
        types=[Type.GRASS],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    opp_team = Team([grass_opp])
    s_fire = _speed_tier_score(slow_fire, opp_team, PARAMS)
    s_water = _speed_tier_score(fast_water, opp_team, PARAMS)
    # Fire: offense=2.0, defense=1.0, initiative=-bonus -> 1.0 - bonus
    # Water: offense=0.5, defense=1.0, initiative=+bonus -> -0.5 + bonus
    assert s_fire > s_water


def test_outspeed_breaks_tie_between_equal_type_matchups() -> None:
    # Two team members with identical type matchups vs the opp; the faster
    # one wins the tiebreak when SpeedTier is used, where MatchupAware
    # falls back to original index.
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
    my_team = Team([slow, fast])  # slow first so index tiebreak would put it ahead
    opp_team = Team([opp])
    speed_cmd = SpeedTierAwareSelectionPolicy().decision((my_team, opp_team), 2)
    parent_cmd = MatchupAwareSelectionPolicy().decision((my_team, opp_team), 2)
    # Parent: index tiebreak -> [0, 1] = [slow, fast].
    # Speed:  initiative bonus -> fast ranks above slow -> [1, 0].
    assert parent_cmd == [0, 1]
    assert speed_cmd == [1, 0]


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(2031)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = SpeedTierAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(2032)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = SpeedTierAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(2033)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = SpeedTierAwareSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_meta_is_ignored() -> None:
    from vgc2.balance.meta import BasicMeta

    rng = default_rng(2034)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy_a = SpeedTierAwareSelectionPolicy()
    policy_b = SpeedTierAwareSelectionPolicy()
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99, 1, 1]
    policy_b.set_meta(meta)
    assert policy_a.decision((my_team, opp_team), 4) == policy_b.decision((my_team, opp_team), 4)


def test_empty_opp_team_returns_stable_order() -> None:
    rng = default_rng(2035)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = SpeedTierAwareSelectionPolicy()
    cmd = policy.decision((my_team, empty), 4)
    # All scores tie at 0.0 -> stable index tiebreak.
    assert cmd == [0, 1, 2, 3]


def test_score_averages_over_multiple_opps() -> None:
    # One outspeed + one outsped should net to zero initiative.
    me = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 100),
    )
    faster = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 200),
    )
    slower = _mk_pkm(
        types=[Type.NORMAL],
        move_types=[Type.NORMAL],
        base_stats=(100, 100, 100, 100, 100, 50),
    )
    score = _speed_tier_score(me, Team([faster, slower]), PARAMS)
    assert math.isclose(score, 0.0, abs_tol=1e-9)
