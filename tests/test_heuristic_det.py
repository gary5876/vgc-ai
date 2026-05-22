"""Smoke tests for the deterministic-rollout heuristic policy.

Sanity-only: the policy is registered, builds, plays a legal battle, and the
1-ply rollout is actually deterministic (same input state → same chosen
action across repeated decisions).
"""

from __future__ import annotations

from typing import Any, ClassVar

from vgc2.agent.battle import RandomBattlePolicy

from vgc_ai.cli import POLICIES
from vgc_ai.eval.duel import duel
from vgc_ai.policies.heuristic_det import HeuristicDetBattlePolicy


def test_heuristic_det_is_registered() -> None:
    assert POLICIES["heuristic_det"] is HeuristicDetBattlePolicy


def test_heuristic_det_runs_a_battle() -> None:
    result = duel(HeuristicDetBattlePolicy, RandomBattlePolicy, n_battles=2, fixed_team_seed=42)
    assert result.n_battles == 2
    assert result.wins_a + result.wins_b + result.ties == 2


def test_heuristic_det_is_deterministic_with_fixed_team_seed() -> None:
    r1 = duel(HeuristicDetBattlePolicy, RandomBattlePolicy, n_battles=4, fixed_team_seed=7)
    r2 = duel(HeuristicDetBattlePolicy, RandomBattlePolicy, n_battles=4, fixed_team_seed=7)
    assert (r1.wins_a, r1.wins_b, r1.ties) == (r2.wins_a, r2.wins_b, r2.ties)


def test_set_meta_with_empty_record_keeps_uniform_eval() -> None:
    # Epoch 0 of every championship: meta exists but no matches recorded.
    # The cache must stay None so evaluate() takes the bit-identical
    # uniform path (worst-case-parity guarantee).
    policy = HeuristicDetBattlePolicy()

    class _EmptyMeta:
        record: ClassVar[list[Any]] = []

    policy.set_meta(_EmptyMeta())  # type: ignore[arg-type]
    assert policy._usage_weights is None


def test_set_meta_with_records_caches_usage_rates() -> None:
    # Two matches recorded; species 0 appeared on every team (4/4 → 1.0),
    # species 1 appeared on one team (1/4 → 0.25). Within-team duplicates
    # must collapse, matching BasicMeta._update_usage's counted_species logic.
    policy = HeuristicDetBattlePolicy()

    class _Species:
        def __init__(self, sid: int) -> None:
            self.id = sid

    class _Member:
        def __init__(self, sid: int) -> None:
            self.species = _Species(sid)

    class _Team:
        def __init__(self, sids: list[int]) -> None:
            self.members = [_Member(s) for s in sids]

    class _Meta:
        def __init__(self) -> None:
            self.record = [
                ((_Team([0, 0, 1]), _Team([0])), 0, (1200, 1200)),  # species 0 dup
                ((_Team([0]), _Team([0])), 1, (1200, 1200)),
            ]

    policy.set_meta(_Meta())  # type: ignore[arg-type]
    assert policy._usage_weights == {0: 1.0, 1: 0.25}
