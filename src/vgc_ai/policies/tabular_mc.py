"""Tabular first-visit Monte Carlo battle policy.

Follows the structure of AurelianTactics 2024 (3rd place): a collapsed
~11-dim integer state encoding plus a state→action-values table backed by a
plain ``dict``, learned via first-visit MC over completed episodes.

State encoding (11 dims for doubles, ``n_active=2``):

    0,1   side 0 active HP bucket (0..3) for slots 0,1     (-1 if missing)
    2,3   side 1 active HP bucket (0..3) for slots 0,1     (-1 if missing)
    4,5   side 0 active status (Status enum int) for slots 0,1   (-1 if missing)
    6,7   side 1 active status                                   (-1 if missing)
    8     side 0 non-fainted Pokemon count (0..team_size)
    9     side 1 non-fainted Pokemon count (0..team_size)
    10    Weather enum int (0..4)

Action keys are **canonical** rather than positional. The legacy positional
``action_idx`` (index into ``get_actions``' cartesian product) was unstable
across turns: PP-decay, disabling, and reserve fainting all reorder
``battling_moves`` and ``reserve``, so the same ``action_idx`` denotes
different commands across visits to the same ``state_key`` — which conflates
incompatible returns under first-visit MC.

The canonical per-slot command is ``(kind, payload)`` where:

* ``kind = 0`` → MOVE; payload is the move's stable signature
  ``(pkm_type, base_power, accuracy_pct, category)``
  (vgc2 sets ``move.constants.id == -1`` for procedurally-generated
  moves, so the spec's ``id`` path is never taken — the derived signature
  is the canonical id in practice).
* ``kind = 1`` → SWITCH; payload is the destination Pokemon's stable
  signature ``(types_tuple, base_stats_tuple)`` derived from its species
  (``species.id == -1`` likewise).

The joint action key is the tuple of per-slot canonical commands.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vgc2.agent import BattlePolicy
from vgc2.agent.battle import get_actions
from vgc2.battle_engine import BattleCommand, State
from vgc2.battle_engine.team import BattlingTeam
from vgc2.battle_engine.view import TeamView

StateKey = tuple[int, ...]
"""Encoded state — a fixed-length integer tuple, hashable for dict use."""

ActionKey = tuple[Any, ...]
"""Canonical joint action — a tuple of per-slot ``(kind, payload)`` tuples."""

_HP_BUCKETS = 4
_MAX_ACTIVE_SLOTS = 2
_ENCODING_LEN = 4 * _MAX_ACTIVE_SLOTS + 3
"""4 features per active slot (hp,hp,status,status across both sides)
+ alive_count_a + alive_count_b + weather."""

_KIND_MOVE = 0
_KIND_SWITCH = 1


def _hp_bucket(hp: int, max_hp: int) -> int:
    """Bucket the current HP into ``_HP_BUCKETS`` quartiles.

    Fainted (hp == 0) → 0. Full HP → ``_HP_BUCKETS - 1``. The clamp guards
    against the rare engine state where ``hp`` momentarily exceeds ``max_hp``
    (e.g. after a heal mid-resolution).
    """
    if max_hp <= 0:
        return 0
    bucket = int(hp * _HP_BUCKETS / max_hp)
    if bucket < 0:
        return 0
    if bucket >= _HP_BUCKETS:
        return _HP_BUCKETS - 1
    return bucket


def _alive_count(side: Any) -> int:
    return sum(1 for p in list(side.team.active) + list(side.team.reserve) if p.hp > 0)


def encode_state(state: State) -> StateKey:
    """Project a vgc2 ``State`` into the integer tuple used as a Q-table key."""
    side0, side1 = state.sides[0], state.sides[1]
    feats: list[int] = []
    for side in (side0, side1):
        active = list(side.team.active)
        for slot in range(_MAX_ACTIVE_SLOTS):
            if slot < len(active):
                pkm = active[slot]
                feats.append(_hp_bucket(pkm.hp, pkm.constants.stats[0]))
            else:
                feats.append(-1)
    for side in (side0, side1):
        active = list(side.team.active)
        for slot in range(_MAX_ACTIVE_SLOTS):
            if slot < len(active):
                feats.append(int(active[slot].status))
            else:
                feats.append(-1)
    feats.append(_alive_count(side0))
    feats.append(_alive_count(side1))
    feats.append(int(state.weather))
    return tuple(feats)


def _move_signature(move: Any) -> tuple[int, ...]:
    """Stable id for a move — ``move.constants.id`` if set, else derived signature."""
    mc = move.constants
    move_id = getattr(mc, "id", -1)
    if move_id is not None and move_id != -1:
        return (int(move_id),)
    pkm_type = mc.pkm_type
    category = mc.category
    return (
        int(getattr(pkm_type, "value", pkm_type)),
        int(mc.base_power),
        round(mc.accuracy * 100),
        int(getattr(category, "value", category)),
    )


def _pkm_signature(pkm: Any) -> tuple[int, ...]:
    """Stable id for a Pokemon — ``species.id`` if set, else derived signature.

    The vgc2 generator emits fictional Pokemon with ``species.id == -1``, so
    in practice the fallback path is always taken. We hash on ``(types,
    base_stats)``: both invariant across the battle, and together they
    uniquely identify the species in any reasonable randomly-generated team.
    """
    pc = pkm.constants
    species = pc.species
    species_id = getattr(species, "id", -1)
    if species_id is not None and species_id != -1:
        return (int(species_id),)
    types_tuple = tuple(int(getattr(t, "value", t)) for t in species.types)
    base_stats = tuple(int(s) for s in species.base_stats)
    return types_tuple + base_stats


def canonicalize_action(
    joint: Iterable[BattleCommand],
    team_pair: tuple[BattlingTeam, BattlingTeam],
) -> ActionKey:
    """Map a per-slot list of ``(move_idx_or_-1, target_idx)`` commands to a
    canonical key whose components depend on move/Pokemon identity, not on
    positional indices into volatile lists.
    """
    attackers = list(team_pair[0].active)
    reserve = list(team_pair[0].reserve)
    canon: list[tuple[Any, ...]] = []
    for slot, cmd in enumerate(joint):
        move_idx, target_idx = cmd
        if move_idx == -1:
            if 0 <= target_idx < len(reserve):
                canon.append((_KIND_SWITCH, _pkm_signature(reserve[target_idx])))
            else:
                canon.append((_KIND_SWITCH, ()))
        else:
            if slot < len(attackers):
                moves = attackers[slot].battling_moves
                if 0 <= move_idx < len(moves):
                    canon.append((_KIND_MOVE, _move_signature(moves[move_idx])))
                else:
                    canon.append((_KIND_MOVE, ()))
            else:
                canon.append((_KIND_MOVE, ()))
    return tuple(canon)


def _action_key_to_json(key: ActionKey) -> str:
    """Serialise an ActionKey to a JSON-string dict key."""
    return json.dumps(key)


def _action_key_from_json(s: str) -> ActionKey:
    """Inverse of :func:`_action_key_to_json` — produces a hashable nested tuple."""

    def _to_tuple(obj: Any) -> Any:
        if isinstance(obj, list):
            return tuple(_to_tuple(x) for x in obj)
        return obj

    parsed = json.loads(s)
    result: ActionKey = _to_tuple(parsed)
    return result


class TabularMCBattlePolicy(BattlePolicy):  # type: ignore[misc]  # vgc2 is untyped; BattlePolicy resolves as Any under --strict
    """First-visit Monte Carlo over a collapsed integer state encoding.

    The Q-table is empty until :meth:`learn` is called or :meth:`load`
    deserialises a checkpoint. Until then, :meth:`decision` returns a
    uniformly-random legal joint action from :func:`get_actions`. After
    training, the policy picks the joint action whose canonical key has the
    highest visited mean return for the encoded state, falling back to random
    when the state is unseen or no legal action's canonical key has been
    visited from this state.
    """

    def __init__(self, rng_seed: int | None = None, model_path: str | Path | None = None) -> None:
        self._q: dict[StateKey, dict[ActionKey, float]] = {}
        self._n: dict[StateKey, dict[ActionKey, int]] = {}
        self._rng = random.Random(rng_seed)
        if model_path is not None:
            path = Path(model_path)
            if path.exists():
                self.load(path)

    def decision(
        self,
        state: State,
        opp_view: TeamView | None = None,
    ) -> list[BattleCommand]:
        team_pair = (state.sides[0].team, state.sides[1].team)
        joint_actions = get_actions(team_pair)
        if not joint_actions:
            return [(0, 0)] * len(state.sides[0].team.active)

        s_key = encode_state(state)
        q_row = self._q.get(s_key)
        n_row = self._n.get(s_key)
        best_idx = -1
        best_val = float("-inf")
        if q_row is not None and n_row is not None:
            for idx, joint in enumerate(joint_actions):
                a_key = canonicalize_action(joint, team_pair)
                visits = n_row.get(a_key, 0)
                if visits > 0:
                    val = q_row.get(a_key, 0.0)
                    if val > best_val:
                        best_val = val
                        best_idx = idx
        if best_idx < 0:
            best_idx = self._rng.randrange(len(joint_actions))
        return list(joint_actions[best_idx])

    def learn(self, trajectories: list[list[tuple[StateKey, ActionKey, float]]]) -> None:
        """First-visit MC update.

        ``trajectories`` is a list of episodes; each episode is a list of
        ``(state_key, action_key, step_reward)`` tuples in temporal order.
        For each unique ``(state_key, action_key)`` first-visited in an
        episode, the table mean is updated incrementally with the
        episode's terminal return (sum of step rewards).
        """
        for episode in trajectories:
            if not episode:
                continue
            g = sum(step[2] for step in episode)
            seen: set[tuple[StateKey, ActionKey]] = set()
            for s_key, a_key, _r in episode:
                if (s_key, a_key) in seen:
                    continue
                seen.add((s_key, a_key))
                self._update(s_key, a_key, g)

    def _update(self, s_key: StateKey, a_key: ActionKey, g: float) -> None:
        q_row = self._q.get(s_key)
        n_row = self._n.get(s_key)
        if q_row is None or n_row is None:
            q_row = {}
            n_row = {}
            self._q[s_key] = q_row
            self._n[s_key] = n_row
        new_n = n_row.get(a_key, 0) + 1
        n_row[a_key] = new_n
        prev_q = q_row.get(a_key, 0.0)
        # incremental mean: q_new = q_old + (g - q_old) / n
        q_row[a_key] = prev_q + (g - prev_q) / new_n

    def num_state_action_keys(self) -> int:
        """Total number of distinct (state_key, action_key) pairs in the table.

        Used by training/diagnostic scripts to track table growth and to
        verify that canonicalization actually reduces key explosion versus
        the legacy positional-index baseline.
        """
        return sum(len(row) for row in self._n.values())

    def save(self, path: str | Path) -> None:
        """Serialise the table to JSON.

        State keys (int tuples) become comma-separated strings; action keys
        (nested tuples) become JSON-encoded strings — both reversed by
        :meth:`load`.
        """
        payload = {
            "q": {
                ",".join(str(i) for i in s_key): {
                    _action_key_to_json(a_key): v for a_key, v in row.items()
                }
                for s_key, row in self._q.items()
            },
            "n": {
                ",".join(str(i) for i in s_key): {
                    _action_key_to_json(a_key): v for a_key, v in row.items()
                }
                for s_key, row in self._n.items()
            },
        }
        Path(path).write_text(json.dumps(payload))

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text())
        self._q = {
            tuple(int(x) for x in s_key.split(",")): {
                _action_key_from_json(a_key): float(v) for a_key, v in row.items()
            }
            for s_key, row in payload.get("q", {}).items()
        }
        self._n = {
            tuple(int(x) for x in s_key.split(",")): {
                _action_key_from_json(a_key): int(v) for a_key, v in row.items()
            }
            for s_key, row in payload.get("n", {}).items()
        }


__all__ = [
    "ActionKey",
    "StateKey",
    "TabularMCBattlePolicy",
    "canonicalize_action",
    "encode_state",
]
