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
    MetaWeightedSelectionPolicy,
    VgcAiSelectionPolicy,
    _best_offense_multiplier,
    _meta_weighted_selection_score,
    _opp_usage_weights,
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


# ----- MetaWeightedSelectionPolicy ----------------------------------------


def _pop_meta_for_opp_team(opp_team: Team) -> BasicMeta:
    """Hand-construct a BasicMeta with synthetic per-species usage.

    BasicMeta.usage_rate_pokemon = pokemon_usage[species.id] / (len(record) * 2),
    so we only need to populate ``record`` (any non-empty sentinel) and
    ``pokemon_usage`` indexed by ``species.id``. Each opp member gets a fresh
    id so the weights can differ per slot.
    """
    roster = []
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
        roster.append(opp.species)
    meta = BasicMeta(move_set=[], roster=roster)
    # 10 records -> divisor = 20; usage counts below produce 0.05, 0.45.
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 9][: len(opp_team.members)] + [0] * max(0, len(opp_team.members) - 2)
    return meta


def test_meta_weighted_alias_subclasses_matchup_aware() -> None:
    assert issubclass(MetaWeightedSelectionPolicy, MatchupAwareSelectionPolicy)


def test_meta_weighted_falls_back_when_meta_none() -> None:
    rng = default_rng(101)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    base = MatchupAwareSelectionPolicy()
    new = MetaWeightedSelectionPolicy()
    assert base.decision((my_team, opp_team), 4) == new.decision((my_team, opp_team), 4)


def test_meta_weighted_falls_back_at_epoch_zero() -> None:
    # BasicMeta with empty record => usage_rate_pokemon raises ZeroDivisionError.
    # Policy should silently fall back to MatchupAware parent behavior.
    rng = default_rng(103)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    new = MetaWeightedSelectionPolicy()
    new.set_meta(meta)
    base = MatchupAwareSelectionPolicy()
    assert new.decision((my_team, opp_team), 4) == base.decision((my_team, opp_team), 4)


def test_meta_weighted_falls_back_on_empty_opp_team() -> None:
    rng = default_rng(105)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    new = MetaWeightedSelectionPolicy()
    new.set_meta(BasicMeta(move_set=[], roster=[]))
    assert new.decision((my_team, empty), 4) == [0, 1, 2, 3]


def test_opp_usage_weights_normalise_to_unit_sum() -> None:
    rng = default_rng(107)
    opp_team = gen_team(4, 4, rng=rng)
    meta = _pop_meta_for_opp_team(opp_team)
    weights = _opp_usage_weights(meta, opp_team)
    assert weights is not None
    assert len(weights) == len(opp_team.members)
    assert abs(sum(weights) - 1.0) < 1e-9
    # The two non-zero opp ids get all the mass; the rest are zero per
    # _pop_meta_for_opp_team's pokemon_usage = [1, 9, 0, 0].
    assert weights[0] > 0.0 and weights[1] > 0.0
    assert weights[1] > weights[0], "id=1 (count=9) should outweigh id=0 (count=1)"


def test_meta_weighted_uses_usage_when_populated() -> None:
    # Two opp members; species 1 has 9x the usage weight of species 0.
    # Our team has two members: A counters species 0, B counters species 1.
    # Type-chart parent treats both opps equally (mean) -> A and B score the
    # same up to defensive ties. With meta weighting, B wins because the high-
    # usage opp drives the score.
    counter_to_normal = _mk_pkm(types=[Type.FIGHT], move_types=[Type.FIGHT])
    counter_to_fire = _mk_pkm(types=[Type.WATER], move_types=[Type.WATER])
    opp_normal = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    opp_fire = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    my_team = Team([counter_to_normal, counter_to_fire])
    opp_team = Team([opp_normal, opp_fire])

    # Skewed usage: species 1 (opp_fire) gets 9x mass of species 0 (opp_normal).
    opp_normal.species.id = 0
    opp_fire.species.id = 1
    meta = BasicMeta(move_set=[], roster=[opp_normal.species, opp_fire.species])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 9]

    policy = MetaWeightedSelectionPolicy()
    policy.set_meta(meta)
    cmd = policy.decision((my_team, opp_team), 2)
    assert cmd[0] == 1, "counter_to_fire should lead when the meta heavily weights opp_fire (id=1)"


def test_meta_weighted_uses_typechart_when_meta_set_but_zero_usage() -> None:
    # set_meta with a BasicMeta that has record but all-zero pokemon_usage.
    # Should still fall back (total weight = 0).
    rng = default_rng(109)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 4  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [0] * len(opp_team.members)
    assert _opp_usage_weights(meta, opp_team) is None
    new = MetaWeightedSelectionPolicy()
    new.set_meta(meta)
    base = MatchupAwareSelectionPolicy()
    assert new.decision((my_team, opp_team), 4) == base.decision((my_team, opp_team), 4)


def test_meta_weighted_score_is_zero_for_empty_opp_team() -> None:
    pkm = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _meta_weighted_selection_score(pkm, Team([]), [], PARAMS) == 0.0
