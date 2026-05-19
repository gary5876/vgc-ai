"""Tabular first-visit Monte Carlo battle policy.

Follows the structure of AurelianTactics 2024 (3rd place): a collapsed
~11-dim integer state encoding, a state→action-values table backed by a plain
``dict``, and a first-visit MC ``learn`` step that averages observed returns.

Inference falls back to ``GreedyBattlePolicy`` whenever the encoded state is
unseen or no candidate action has cleared the ``min_visits`` threshold —
matching the writeup's "if found in dictionary, take the action; if not, use
the baseline agent." This is the load-bearing detail: an earlier random
fallback meant most turns picked a random legal action (round-robin row
``tabular_mc vs random = 0.500``), masking the Q-table entirely.

State encoding (11 dims for doubles, ``n_active=2``):

    0,1   side 0 active HP bucket (0..3) for slots 0,1     (-1 if missing)
    2,3   side 1 active HP bucket (0..3) for slots 0,1     (-1 if missing)
    4,5   side 0 active status (Status enum int) for slots 0,1   (-1 if missing)
    6,7   side 1 active status                                   (-1 if missing)
    8     side 0 non-fainted Pokemon count (0..team_size)
    9     side 1 non-fainted Pokemon count (0..team_size)
    10    Weather enum int (0..4)

The encoding is a deliberately lossy projection — many actual battle states
collapse to the same key. That is the point of tabular MC: the table generalises
across positions that share gross features (HP totals, status, board count).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from vgc2.agent import BattlePolicy
from vgc2.agent.battle import GreedyBattlePolicy, get_actions
from vgc2.battle_engine import BattleCommand, State
from vgc2.battle_engine.view import TeamView

_DEFAULT_MIN_VISITS = 100
"""Per AurelianTactics 2024: ignore Q-values backed by fewer than 100 visits;
fall through to the baseline. Typical well-sampled states had 1000-20000 visits."""

StateKey = tuple[int, ...]
"""Encoded state — a fixed-length integer tuple, hashable for dict use."""

_HP_BUCKETS = 4
_MAX_ACTIVE_SLOTS = 2
_ENCODING_LEN = 4 * _MAX_ACTIVE_SLOTS + 3
"""4 features per active slot (hp,hp,status,status across both sides)
+ alive_count_a + alive_count_b + weather."""


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
    """Project a vgc2 ``State`` into the integer tuple used as a Q-table key.

    Always emits ``_ENCODING_LEN`` ints. Missing active slots (1v0 or 0v0
    edge configurations) are encoded as -1 so they hash to a distinct bucket
    rather than colliding with "full HP" or "no status".
    """
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


class TabularMCBattlePolicy(BattlePolicy):  # type: ignore[misc]  # vgc2 is untyped; BattlePolicy resolves as Any under --strict
    """First-visit Monte Carlo over a collapsed integer state encoding.

    When the encoded state is unseen, or no candidate joint action has reached
    ``min_visits`` samples, :meth:`decision` delegates to
    :class:`GreedyBattlePolicy`. This means an *empty* table behaves exactly
    like greedy — a strong floor — and the Q-table is pure upside above it.
    Once a state-action pair has enough samples, the policy picks the
    highest-mean-return joint action.
    """

    def __init__(
        self,
        rng_seed: int | None = None,
        *,
        min_visits: int = _DEFAULT_MIN_VISITS,
        model_path: str | Path | None = None,
    ) -> None:
        self._q: dict[StateKey, list[float]] = {}
        self._n: dict[StateKey, list[int]] = {}
        self._rng = random.Random(rng_seed)
        self._min_visits = min_visits
        self._baseline = GreedyBattlePolicy()
        if model_path is not None:
            path = Path(model_path)
            if path.is_file():
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

        key = encode_state(state)
        q_row = self._q.get(key)
        n_row = self._n.get(key)
        best_idx = -1
        best_val = float("-inf")
        if q_row is not None and n_row is not None:
            limit = min(len(joint_actions), len(q_row))
            for idx in range(limit):
                if n_row[idx] >= self._min_visits and q_row[idx] > best_val:
                    best_val = q_row[idx]
                    best_idx = idx
        if best_idx < 0:
            fallback: list[BattleCommand] = self._baseline.decision(state, opp_view)
            return fallback
        return list(joint_actions[best_idx])

    def learn(self, trajectories: list[list[tuple[StateKey, int, float]]]) -> None:
        """First-visit MC update.

        ``trajectories`` is a list of episodes; each episode is a list of
        ``(state_key, action_idx, step_reward)`` tuples in temporal order.
        For each unique ``(state_key, action_idx)`` first-visited in an
        episode, the table mean is updated incrementally with the
        episode's terminal return (sum of step rewards).
        """
        for episode in trajectories:
            if not episode:
                continue
            g = sum(step[2] for step in episode)
            seen: set[tuple[StateKey, int]] = set()
            for s_key, a_idx, _r in episode:
                if (s_key, a_idx) in seen:
                    continue
                seen.add((s_key, a_idx))
                self._update(s_key, a_idx, g)

    def _update(self, s_key: StateKey, a_idx: int, g: float) -> None:
        if a_idx < 0:
            return
        q_row = self._q.get(s_key)
        n_row = self._n.get(s_key)
        if q_row is None or n_row is None:
            q_row = []
            n_row = []
            self._q[s_key] = q_row
            self._n[s_key] = n_row
        while len(q_row) <= a_idx:
            q_row.append(0.0)
            n_row.append(0)
        n_row[a_idx] += 1
        # incremental mean: q_new = q_old + (g - q_old) / n
        q_row[a_idx] += (g - q_row[a_idx]) / n_row[a_idx]

    def save(self, path: str | Path) -> None:
        """Serialise the table to JSON.

        Tuple keys are encoded as comma-separated strings because JSON only
        supports string keys at the object level.
        """
        payload = {
            "q": {",".join(str(i) for i in k): v for k, v in self._q.items()},
            "n": {",".join(str(i) for i in k): v for k, v in self._n.items()},
        }
        Path(path).write_text(json.dumps(payload))

    def load(self, path: str | Path) -> None:
        payload = json.loads(Path(path).read_text())
        self._q = {
            tuple(int(x) for x in k.split(",")): [float(v) for v in row]
            for k, row in payload.get("q", {}).items()
        }
        self._n = {
            tuple(int(x) for x in k.split(",")): [int(v) for v in row]
            for k, row in payload.get("n", {}).items()
        }


__all__ = ["StateKey", "TabularMCBattlePolicy", "encode_state"]
