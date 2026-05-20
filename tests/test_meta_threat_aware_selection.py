"""Unit tests for ``MetaThreatAwareSelectionPolicy``.

Mirrors the structure of the parent tests in ``test_selection.py``: synthetic
teams for deterministic ranking checks, ``gen_team``-driven teams for legality
and contract. The policy composes meta-weighted offense with worst-case
threat defense; the fallback path uses (uniform_mean_offense - max_defense)
when the championship meta is empty.
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
    MetaThreatAwareSelectionPolicy,
    _meta_threat_aware_selection_score,
    _threat_aware_uniform_score,
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


def test_subclasses_matchup_aware() -> None:
    assert issubclass(MetaThreatAwareSelectionPolicy, MatchupAwareSelectionPolicy)


def test_score_zero_for_empty_opp_team() -> None:
    pkm = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _meta_threat_aware_selection_score(pkm, Team([]), [], PARAMS) == 0.0


def test_threat_aware_uniform_score_zero_for_empty_opp_team() -> None:
    pkm = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    assert _threat_aware_uniform_score(pkm, Team([]), PARAMS) == 0.0


def test_falls_back_to_uniform_threat_when_meta_none() -> None:
    # No set_meta -> _meta is None -> uniform (mean_offense - max_defense)
    # ranking. We assert the chosen order matches _threat_aware_uniform_score
    # scoring directly.
    rng = default_rng(801)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    new = MetaThreatAwareSelectionPolicy()
    cmd = new.decision((my_team, opp_team), 4)
    expected = sorted(
        range(len(my_team.members)),
        key=lambda i: (-_threat_aware_uniform_score(my_team.members[i], opp_team, PARAMS), i),
    )
    assert cmd == expected


def test_falls_back_to_uniform_threat_at_epoch_zero() -> None:
    # BasicMeta with empty record -> usage_rate_pokemon raises ZeroDivisionError.
    # Policy should silently fall back to the uniform threat-aware baseline.
    rng = default_rng(803)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    new = MetaThreatAwareSelectionPolicy()
    new.set_meta(meta)
    cmd = new.decision((my_team, opp_team), 4)
    expected = sorted(
        range(len(my_team.members)),
        key=lambda i: (-_threat_aware_uniform_score(my_team.members[i], opp_team, PARAMS), i),
    )
    assert cmd == expected


def test_falls_back_to_uniform_threat_when_zero_usage() -> None:
    rng = default_rng(805)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 4  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [0] * len(opp_team.members)
    new = MetaThreatAwareSelectionPolicy()
    new.set_meta(meta)
    cmd = new.decision((my_team, opp_team), 4)
    expected = sorted(
        range(len(my_team.members)),
        key=lambda i: (-_threat_aware_uniform_score(my_team.members[i], opp_team, PARAMS), i),
    )
    assert cmd == expected


def test_falls_back_on_empty_opp_team() -> None:
    rng = default_rng(807)
    my_team = gen_team(4, 4, rng=rng)
    empty = Team([])
    new = MetaThreatAwareSelectionPolicy()
    new.set_meta(BasicMeta(move_set=[], roster=[]))
    assert new.decision((my_team, empty), 4) == [0, 1, 2, 3]


def test_uses_usage_for_offense_when_populated() -> None:
    # Two opp members; species 1 has 9x the usage weight of species 0.
    # Our team has two members: A counters species 0, B counters species 1.
    # The uniform fallback treats both opps\' offense equally (mean) -> A and B
    # score the same up to defense ties. With meta-threat weighting, B wins
    # because the high-usage opp drives the offense score.
    counter_to_normal = _mk_pkm(types=[Type.FIGHT], move_types=[Type.FIGHT])
    counter_to_fire = _mk_pkm(types=[Type.WATER], move_types=[Type.WATER])
    opp_normal = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    opp_fire = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    my_team = Team([counter_to_normal, counter_to_fire])
    opp_team = Team([opp_normal, opp_fire])

    opp_normal.species.id = 0
    opp_fire.species.id = 1
    meta = BasicMeta(move_set=[], roster=[opp_normal.species, opp_fire.species])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 9]

    policy = MetaThreatAwareSelectionPolicy()
    policy.set_meta(meta)
    cmd = policy.decision((my_team, opp_team), 2)
    assert cmd[0] == 1, "counter_to_fire should lead when the meta heavily weights opp_fire (id=1)"


def test_defense_term_is_max_not_weighted_mean() -> None:
    # Construct a case where a usage-weighted defense (as in MetaWeighted)
    # and the max defense disagree, then assert MetaThreatAware uses the max.
    #
    # Our member A: GRASS (2x weak to FIRE; neutral vs NORMAL).
    # Our member B: WATER (resists FIRE -> 0.5x; neutral vs NORMAL).
    # Opps: a FIRE attacker (rare in the meta) and a NORMAL attacker (common).
    # Both members carry GROUND moves (neutral on both opps) so offense ties.
    #
    # Under heavy normal weighting [1, 99], a usage-weighted defense would
    # almost ignore the fire threat -- the GRASS member\'s spike vanishes
    # into the average. Max defense ignores the weight and still penalises
    # GRASS the full 2.0 for the FIRE matchup, so WATER strictly outranks
    # GRASS under the max view.
    grass = _mk_pkm(types=[Type.GRASS], move_types=[Type.GROUND])
    water = _mk_pkm(types=[Type.WATER], move_types=[Type.GROUND])
    fire_opp = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    normal_opp = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    my_team = Team([grass, water])
    opp_team = Team([fire_opp, normal_opp])

    fire_opp.species.id = 0
    normal_opp.species.id = 1
    meta = BasicMeta(move_set=[], roster=[fire_opp.species, normal_opp.species])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99]

    policy = MetaThreatAwareSelectionPolicy()
    policy.set_meta(meta)
    cmd = policy.decision((my_team, opp_team), 2)
    assert cmd[0] == 1, (
        "water (no spike weakness) must outrank grass under max-defense scoring "
        "even when the meta strongly weights the non-threatening opp"
    )


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(809)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = MetaThreatAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(811)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = MetaThreatAwareSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(813)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = MetaThreatAwareSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b
