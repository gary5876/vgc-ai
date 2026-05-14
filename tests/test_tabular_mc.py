"""Tests for the tabular MC battle policy.

Covers the encoder invariants (deterministic, fixed length), canonical action
keys (stable across visits even when the underlying ``battling_moves`` order
shifts), the first-visit MC update math, JSON save/load round-tripping with
nested action keys, and an end-to-end smoke duel against ``RandomBattlePolicy``
to make sure the policy still plays full battles without crashing.
"""

from __future__ import annotations

import json

import numpy as np
from vgc2.agent.battle import RandomBattlePolicy, get_actions
from vgc2.battle_engine import State
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.util.generator import gen_team

from vgc_ai.eval.duel import duel
from vgc_ai.policies.tabular_mc import (
    TabularMCBattlePolicy,
    canonicalize_action,
    encode_state,
)


def _make_state(seed: int) -> State:
    rng = np.random.default_rng(seed)
    team = (gen_team(4, 4, rng=rng), gen_team(4, 4, rng=rng))
    return State(get_battle_teams(team, 2))


def test_encoder_is_deterministic_for_same_state() -> None:
    state = _make_state(seed=123)
    a = encode_state(state)
    b = encode_state(state)
    assert a == b


def test_encoder_length_constant_across_20_random_states() -> None:
    lengths = {len(encode_state(_make_state(seed=i))) for i in range(20)}
    assert len(lengths) == 1
    (length,) = lengths
    # 11 dims for doubles n_active=2: see encode_state docstring.
    assert length == 11


def test_untrained_policy_returns_legal_command_per_active() -> None:
    state = _make_state(seed=7)
    policy = TabularMCBattlePolicy(rng_seed=0)
    cmds = policy.decision(state)
    assert len(cmds) == len(state.sides[0].team.active)
    for action, target in cmds:
        assert isinstance(action, int)
        assert isinstance(target, int)


def test_smoke_bench_runs_end_to_end_against_random() -> None:
    """The whole point of the skeleton: an untrained instance plays
    full battles without crashing. No win required."""
    result = duel(
        TabularMCBattlePolicy,
        RandomBattlePolicy,
        n_battles=2,
        fixed_team_seed=11,
    )
    assert result.n_battles == 2
    assert result.wins_a + result.wins_b + result.ties == 2


def test_canonical_action_key_is_deterministic() -> None:
    state = _make_state(seed=42)
    team_pair = (state.sides[0].team, state.sides[1].team)
    joint = get_actions(team_pair)[0]
    a = canonicalize_action(joint, team_pair)
    b = canonicalize_action(joint, team_pair)
    assert a == b


def test_canonical_action_key_distinguishes_distinct_actions() -> None:
    """Two semantically different per-slot commands must canonicalize to
    different keys, otherwise MC updates will conflate them.
    """
    state = _make_state(seed=42)
    team_pair = (state.sides[0].team, state.sides[1].team)
    joints = get_actions(team_pair)
    keys = {canonicalize_action(j, team_pair) for j in joints}
    # At minimum, any two distinct joint actions should produce distinct keys
    # for randomly generated teams (where moves and species are distinct).
    assert len(keys) > 1


def test_canonical_action_key_stable_across_move_reorder() -> None:
    """Reordering an attacker's ``battling_moves`` list (which actually
    happens turn-to-turn as PP decreases / disabling kicks in) must not
    change the canonical key for the same semantic action.
    """
    state = _make_state(seed=99)
    team_pair = (state.sides[0].team, state.sides[1].team)
    joint = get_actions(team_pair)[0]
    before = canonicalize_action(joint, team_pair)
    # Simulate engine reorder: rotate moves list of the first attacker.
    attacker = team_pair[0].active[0]
    moves = list(attacker.battling_moves)
    if len(moves) > 1:
        # Find the move at index 0; after rotation it sits at len-1.
        original_move = moves[0]
        attacker.battling_moves = [*moves[1:], original_move]
        # Rebuild a joint that selects the same semantic move via its new index.
        new_idx = len(moves) - 1
        # Replace slot 0's move_idx with new_idx, keep target/other slots.
        new_slot0 = (new_idx, joint[0][1])
        new_joint = (new_slot0, *tuple(joint[1:]))
        after = canonicalize_action(new_joint, team_pair)
        assert before == after, "canonical key changed despite same semantic move"


def test_learn_records_first_visit_only() -> None:
    policy = TabularMCBattlePolicy(rng_seed=0)
    s = (1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 0)
    a0 = ((0, 11, 80, 100, 1),)
    a1 = ((0, 12, 80, 100, 1),)
    # Two visits of (s, a0) in one episode → first-visit MC counts it once.
    policy.learn([[(s, a0, 1.0), (s, a0, 0.0), (s, a1, 0.0)]])
    assert policy._n[s][a0] == 1
    assert policy._n[s][a1] == 1
    # Episode return = 1.0 → mean for both first-visited (s,a) is 1.0.
    assert policy._q[s][a0] == 1.0
    assert policy._q[s][a1] == 1.0


def test_learn_incremental_mean_matches_handcomputed_value() -> None:
    policy = TabularMCBattlePolicy(rng_seed=0)
    s = (0,) * 11
    a = ((0, 11, 80, 100, 1),)
    # Two episodes both first-visit (s, a). Returns 2.0 and 4.0. Mean = 3.0.
    policy.learn([[(s, a, 2.0)]])
    policy.learn([[(s, a, 4.0)]])
    assert policy._n[s][a] == 2
    assert policy._q[s][a] == 3.0


def test_save_load_roundtrips_table(tmp_path) -> None:  # type: ignore[no-untyped-def]
    policy = TabularMCBattlePolicy(rng_seed=0)
    s = (0, 1, 2, 3, 0, 0, 0, 0, 4, 4, 0)
    a = ((0, 11, 80, 100, 1),)
    policy.learn([[(s, a, 1.5)], [(s, a, 2.5)]])

    path = tmp_path / "table.json"
    policy.save(path)
    raw = json.loads(path.read_text())
    assert "q" in raw and "n" in raw

    restored = TabularMCBattlePolicy(rng_seed=0)
    restored.load(path)
    assert restored._q == policy._q
    assert restored._n == policy._n


def test_model_path_autoload(tmp_path) -> None:  # type: ignore[no-untyped-def]
    policy = TabularMCBattlePolicy(rng_seed=0)
    s = (0,) * 11
    a = ((0, 11, 80, 100, 1),)
    policy.learn([[(s, a, 1.0)]])
    path = tmp_path / "model.json"
    policy.save(path)

    restored = TabularMCBattlePolicy(rng_seed=0, model_path=path)
    assert restored._q == policy._q
    assert restored._n == policy._n


def test_model_path_missing_file_is_silent(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """If the file isn't there yet (e.g. fresh checkout, no training run),
    the constructor must NOT raise — a zero-knowledge policy is still useful
    for unit tests and the smoke bench."""
    missing = tmp_path / "does-not-exist.json"
    policy = TabularMCBattlePolicy(rng_seed=0, model_path=missing)
    assert policy._q == {}
    assert policy._n == {}
