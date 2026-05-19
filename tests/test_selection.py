"""Unit tests for ``MatchupAwareSelectionPolicy``.

Mix of synthetic teams (for deterministic ranking checks) and
``gen_team``-driven teams (for legality / contract).
"""

from __future__ import annotations

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
    VgcAiSelectionPolicy,
    _best_offense_multiplier,
    _selection_score,
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


def test_alias_points_to_concrete_policy() -> None:
    assert VgcAiSelectionPolicy is MatchupAwareSelectionPolicy


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(7)
    my_team = gen_team(4, 4, rng=rng)
    opp_team = gen_team(4, 4, rng=rng)
    policy = MatchupAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_empty_opp_team_returns_stable_order() -> None:
    rng = default_rng(11)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    policy = MatchupAwareSelectionPolicy()
    cmd = policy.decision((my_team, empty), 4)
    assert cmd == [0, 1, 2, 3]


def test_super_effective_attacker_ranks_above_neutral() -> None:
    fire_attacker_vs_grass = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    normal_attacker = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    grass_target = _mk_pkm(types=[Type.GRASS], move_types=[Type.GRASS])
    my_team = Team([normal_attacker, fire_attacker_vs_grass])
    opp_team = Team([grass_target])

    policy = MatchupAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert cmd[0] == 1, "fire attacker (super-effective vs grass) should lead"


def test_defensive_weakness_demotes_member() -> None:
    grass_weak_to_fire = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    fire_resistant = _mk_pkm(types=[Type.WATER], move_types=[Type.NORMAL])
    fire_attacker = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    my_team = Team([grass_weak_to_fire, fire_resistant])
    opp_team = Team([fire_attacker])

    policy = MatchupAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert cmd[0] == 1, "fire-resistant member should outrank fire-weak member"


def test_max_size_caps_output() -> None:
    rng = default_rng(13)
    my_team = gen_team(4, 4, rng=rng)
    opp_team = gen_team(4, 4, rng=rng)
    policy = MatchupAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(17)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = MatchupAwareSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_offense_multiplier_skips_status_moves() -> None:
    status_only = _mk_pkm(types=[Type.NORMAL], move_types=[Type.WATER])
    # Replace the move with a status (base_power=0)
    status_only.moves[0] = Move(
        pkm_type=Type.WATER,
        base_power=0,
        accuracy=1.0,
        max_pp=10,
        category=Category.OTHER,
    )
    target = _mk_pkm(types=[Type.FIRE], move_types=[Type.NORMAL])
    # No damaging moves → multiplier stays at neutral 1.0
    assert _best_offense_multiplier(status_only, target, PARAMS) == 1.0


def test_selection_score_zero_against_empty_opp() -> None:
    pkm = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _selection_score(pkm, Team([]), PARAMS) == 0.0


def test_set_meta_stores_meta_on_instance() -> None:
    policy = MatchupAwareSelectionPolicy()
    assert policy._meta is None
    meta = BasicMeta(move_set=[], roster=[])
    policy.set_meta(meta)
    assert policy._meta is meta
