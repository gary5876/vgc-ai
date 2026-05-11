"""Tests for the tabular MC battle policy skeleton.

The acceptance criteria from `policy-tabular-mc-skeleton` in TASKS.md:
(a) the state encoder is deterministic across two encodings of the same State,
(b) the encoded tuple has the same length across 20 random states.

Plus a smoke test that an untrained instance can run a battle end-to-end via
the duel harness without crashing, and that the JSON save/load round-trips
the table.
"""

from __future__ import annotations

import json

import numpy as np
from vgc2.agent.battle import RandomBattlePolicy
from vgc2.battle_engine import State
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.util.generator import gen_team

from vgc_ai.eval.duel import duel
from vgc_ai.policies.tabular_mc import (
    TabularMCBattlePolicy,
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


def test_learn_records_first_visit_only() -> None:
    policy = TabularMCBattlePolicy(rng_seed=0)
    s = (1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 0)
    # Two visits of (s, 0) in one episode → first-visit MC counts it once.
    policy.learn([[(s, 0, 1.0), (s, 0, 0.0), (s, 1, 0.0)]])
    assert policy._n[s][0] == 1
    assert policy._n[s][1] == 1
    # Episode return = 1.0 → mean for both first-visited (s,a) is 1.0.
    assert policy._q[s][0] == 1.0
    assert policy._q[s][1] == 1.0


def test_learn_incremental_mean_matches_handcomputed_value() -> None:
    policy = TabularMCBattlePolicy(rng_seed=0)
    s = (0,) * 11
    # Two episodes both first-visit (s, 0). Returns 2.0 and 4.0. Mean = 3.0.
    policy.learn([[(s, 0, 2.0)]])
    policy.learn([[(s, 0, 4.0)]])
    assert policy._n[s][0] == 2
    assert policy._q[s][0] == 3.0


def test_save_load_roundtrips_table(tmp_path) -> None:  # type: ignore[no-untyped-def]
    policy = TabularMCBattlePolicy(rng_seed=0)
    s = (0, 1, 2, 3, 0, 0, 0, 0, 4, 4, 0)
    policy.learn([[(s, 2, 1.5)], [(s, 2, 2.5)]])

    path = tmp_path / "table.json"
    policy.save(path)
    # Sanity check the on-disk format is valid JSON.
    raw = json.loads(path.read_text())
    assert "q" in raw and "n" in raw

    restored = TabularMCBattlePolicy(rng_seed=0)
    restored.load(path)
    assert restored._q == policy._q
    assert restored._n == policy._n
