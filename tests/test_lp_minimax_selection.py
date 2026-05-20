"""Unit tests for ``LpMinimaxSelectionPolicy``.

The policy orders our team members by the row-player\'s LP-minimax
mixing distribution over the (my x opp) type-chart matchup matrix.
Tests use synthetic teams to drive deterministic ranking expectations,
plus ``gen_team``-driven teams for legality.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng
from vgc2.battle_engine import BattleRuleParam
from vgc2.battle_engine.modifiers import Category, Nature, Type
from vgc2.battle_engine.move import Move
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies
from vgc2.battle_engine.team import Team
from vgc2.util.generator import gen_team

from vgc_ai.eval.minimax import solve_row_minimax_policy
from vgc_ai.policies.selection import (
    LpMinimaxSelectionPolicy,
    MatchupAwareSelectionPolicy,
    _matchup_payoff_matrix,
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
    assert issubclass(LpMinimaxSelectionPolicy, MatchupAwareSelectionPolicy)


def test_falls_back_to_uniform_on_empty_opp_team() -> None:
    rng = default_rng(101)
    my_team = gen_team(4, 4, rng=rng)
    policy = LpMinimaxSelectionPolicy()
    parent = MatchupAwareSelectionPolicy()
    assert policy.decision((my_team, Team([])), 4) == parent.decision((my_team, Team([])), 4)


def test_falls_back_to_uniform_on_single_member_team() -> None:
    # Single-member my team -> only one row in the LP matrix, so the
    # output is trivially [0]. We assert it matches the parent\'s
    # behaviour exactly so the wrapper adds no surprise in the
    # degenerate case.
    rng = default_rng(103)
    opp_team = gen_team(4, 4, rng=rng)
    single = Team([_mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])])
    policy = LpMinimaxSelectionPolicy()
    parent = MatchupAwareSelectionPolicy()
    assert policy.decision((single, opp_team), 1) == parent.decision((single, opp_team), 1)


def test_matchup_payoff_matrix_shape() -> None:
    rng = default_rng(105)
    my_team = gen_team(3, 4, rng=rng)
    opp_team = gen_team(2, 4, rng=rng)
    matrix = _matchup_payoff_matrix(my_team, opp_team, PARAMS)
    assert matrix.shape == (3, 2)


def test_matchup_payoff_is_antisymmetric_when_teams_swap() -> None:
    rng = default_rng(107)
    a = gen_team(3, 4, rng=rng)
    b = gen_team(3, 4, rng=rng)
    matrix_ab = _matchup_payoff_matrix(a, b, PARAMS)
    matrix_ba = _matchup_payoff_matrix(b, a, PARAMS)
    np.testing.assert_allclose(matrix_ab, -matrix_ba.T)


def test_dominant_lead_ranks_first() -> None:
    # A FIRE attacker that hits both opps super-effectively and has
    # neutral defensive matchup dominates the other lead candidate. The
    # LP-minimax must put it first regardless of opp lead choice.
    fire_attacker = _mk_pkm(types=[Type.FIRE], move_types=[Type.FIRE])
    weak_normal = _mk_pkm(types=[Type.NORMAL], move_types=[Type.NORMAL])
    grass_opp = _mk_pkm(types=[Type.GRASS], move_types=[Type.NORMAL])
    bug_opp = _mk_pkm(types=[Type.BUG], move_types=[Type.NORMAL])
    my_team = Team([weak_normal, fire_attacker])
    opp_team = Team([grass_opp, bug_opp])

    cmd = LpMinimaxSelectionPolicy().decision((my_team, opp_team), 2)
    assert cmd[0] == 1, "fire attacker dominates -- LP must put it first"


def test_returns_unique_in_range_capped_to_max_size() -> None:
    rng = default_rng(109)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = LpMinimaxSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    assert len(cmd) <= 4
    assert len(set(cmd)) == len(cmd)
    assert all(0 <= i < len(my_team.members) for i in cmd)


def test_max_size_caps_output() -> None:
    rng = default_rng(111)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = LpMinimaxSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 2)
    assert len(cmd) == 2


def test_deterministic_across_calls() -> None:
    rng = default_rng(113)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = LpMinimaxSelectionPolicy()
    a = policy.decision((my_team, opp_team), 4)
    b = policy.decision((my_team, opp_team), 4)
    assert a == b


def test_order_consistent_with_lp_mass() -> None:
    # The chosen order must be sorted by descending LP equilibrium mass
    # (with uniform_score tiebreaks). This is the central correctness
    # contract -- if it ever breaks, the policy stops solving the
    # advertised problem.
    rng = default_rng(117)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy = LpMinimaxSelectionPolicy()
    cmd = policy.decision((my_team, opp_team), 4)
    matrix = _matchup_payoff_matrix(my_team, opp_team, PARAMS)
    p = solve_row_minimax_policy(matrix)
    # Check sorted property: for adjacent entries i, j in cmd, p[i] >= p[j].
    from itertools import pairwise

    for left, right in pairwise(cmd):
        assert p[left] + 1e-9 >= p[right]


def test_meta_is_ignored() -> None:
    # The policy explicitly ignores meta. Setting one should not change
    # the decision.
    from vgc2.balance.meta import BasicMeta

    rng = default_rng(119)
    my_team, opp_team = gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng)
    policy_a = LpMinimaxSelectionPolicy()
    policy_b = LpMinimaxSelectionPolicy()
    for i, opp in enumerate(opp_team.members):
        opp.species.id = i
    meta = BasicMeta(move_set=[], roster=[opp.species for opp in opp_team.members])
    meta.record = [(([], []), 0, (0, 0))] * 10  # type: ignore[arg-type,list-item]
    meta.pokemon_usage = [1, 99, 1, 1]
    policy_b.set_meta(meta)
    assert policy_a.decision((my_team, opp_team), 4) == policy_b.decision((my_team, opp_team), 4)
