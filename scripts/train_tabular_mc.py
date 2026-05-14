"""Train ``TabularMCBattlePolicy`` via self-play and save the Q-table.

Pure self-play: two ``TabularMCBattlePolicy`` instances share a single
underlying Q-table. Action selection during training is ε-greedy with ε
annealing linearly from ``--eps-start`` (default 0.1) to ``--eps-end``
(default 0.05) across the run; first-visit MC is applied to the **winner's**
trajectory after each episode (terminal reward +1 to the last step on the
winning side; losing-side trajectories are discarded — the simplest baseline
before re-introducing the loss signal, which is intentionally out of scope
for this task).

Output: ``models/tabular_mc.json`` (gitignored). ``cli.POLICIES["tabular_mc"]``
auto-loads it when the registry instantiates the policy.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from vgc2.agent.battle import RandomBattlePolicy
from vgc2.battle_engine import BattleEngine, BattleRuleParam, State
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import label_teams, run_battle
from vgc2.util.generator import gen_team

from vgc_ai.policies.tabular_mc import (
    ActionKey,
    StateKey,
    TabularMCBattlePolicy,
    canonicalize_action,
    encode_state,
)


@dataclass
class _TrainStats:
    episodes: int
    wins_0: int
    wins_1: int
    ties: int
    elapsed_sec: float
    state_keys: int
    state_action_keys: int

    @property
    def eps_per_sec(self) -> float:
        return self.episodes / self.elapsed_sec if self.elapsed_sec > 0 else 0.0


class _RecordingPolicy:
    """Wrap ``TabularMCBattlePolicy`` to record (state_key, action_key) pairs.

    The wrapper plays an ε-greedy variant of the wrapped policy: with prob ε
    a uniformly-random legal joint action, otherwise the underlying policy's
    greedy choice. Each chosen action is recorded as a canonical key against
    the current encoded state so the trainer can apply MC updates from the
    completed trajectory.
    """

    def __init__(self, wrapped: TabularMCBattlePolicy, rng: np.random.Generator, epsilon: float):
        self._wrapped = wrapped
        self._rng = rng
        self._epsilon = epsilon
        self.trajectory: list[tuple[StateKey, ActionKey, float]] = []

    def set_epsilon(self, epsilon: float) -> None:
        self._epsilon = epsilon

    def reset_trajectory(self) -> None:
        self.trajectory = []

    def decision(self, state: State, opp_view: TeamView | None = None) -> Any:
        from vgc2.agent.battle import get_actions

        team_pair = (state.sides[0].team, state.sides[1].team)
        joint_actions = get_actions(team_pair)
        if not joint_actions:
            return [(0, 0)] * len(state.sides[0].team.active)

        s_key = encode_state(state)
        if self._rng.random() < self._epsilon:
            idx = int(self._rng.integers(0, len(joint_actions)))
        else:
            chosen = self._wrapped.decision(state, opp_view)
            idx = next(
                (i for i, ja in enumerate(joint_actions) if list(ja) == list(chosen)),
                int(self._rng.integers(0, len(joint_actions))),
            )

        joint = joint_actions[idx]
        a_key = canonicalize_action(joint, team_pair)
        self.trajectory.append((s_key, a_key, 0.0))
        return list(joint)

    def on_new_battle(self) -> None:
        if hasattr(self._wrapped, "on_new_battle"):
            self._wrapped.on_new_battle()

    def set_params(self, params: BattleRuleParam) -> None:
        if hasattr(self._wrapped, "set_params"):
            self._wrapped.set_params(params)


def _run_training_episode(
    shared: TabularMCBattlePolicy,
    rng: np.random.Generator,
    epsilon: float,
    *,
    team_size: int,
    n_active: int,
    max_pkm_moves: int,
    params: BattleRuleParam,
) -> int:
    """Play one self-play episode, MC-update from the winner's trajectory.

    Returns the winner index (0 or 1; -1 on tie).
    """
    team = (gen_team(team_size, max_pkm_moves), gen_team(team_size, max_pkm_moves))
    label_teams(team)
    team_view = (TeamView(team[0]), TeamView(team[1]))
    state = State(get_battle_teams(team, n_active))
    state_view = (
        StateView(state, 0, team_view),
        StateView(state, 1, team_view),
    )
    engine = BattleEngine(state, debug=False)

    a = _RecordingPolicy(shared, rng, epsilon)
    b = _RecordingPolicy(shared, rng, epsilon)
    a.set_params(params)
    b.set_params(params)

    winner = run_battle(engine, (a, b), team_view, state_view, client=None)

    if winner == 0:
        traj = a.trajectory
    elif winner == 1:
        traj = b.trajectory
    else:
        return -1

    if not traj:
        return winner
    s_key, a_key, _ = traj[-1]
    traj[-1] = (s_key, a_key, 1.0)
    shared.learn([traj])
    return winner


def train(
    *,
    episodes: int,
    output: Path,
    team_size: int,
    n_active: int,
    max_pkm_moves: int,
    eps_start: float,
    eps_end: float,
    seed: int,
    log_every: int,
) -> _TrainStats:
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    params = BattleRuleParam()
    shared = TabularMCBattlePolicy(rng_seed=seed)

    wins_0 = 0
    wins_1 = 0
    ties = 0
    t0 = time.perf_counter()
    for ep in range(episodes):
        progress = ep / max(episodes - 1, 1)
        epsilon = eps_start + progress * (eps_end - eps_start)
        winner = _run_training_episode(
            shared,
            rng,
            epsilon,
            team_size=team_size,
            n_active=n_active,
            max_pkm_moves=max_pkm_moves,
            params=params,
        )
        if winner == 0:
            wins_0 += 1
        elif winner == 1:
            wins_1 += 1
        else:
            ties += 1
        if log_every and (ep + 1) % log_every == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"[train] ep={ep + 1}/{episodes} "
                f"wins_0={wins_0} wins_1={wins_1} ties={ties} "
                f"sa_keys={shared.num_state_action_keys()} "
                f"states={len(shared._n)} "
                f"eps={epsilon:.3f} "
                f"eps/s={(ep + 1) / elapsed:.1f}",
                flush=True,
            )
    elapsed = time.perf_counter() - t0
    shared.save(output)
    return _TrainStats(
        episodes=episodes,
        wins_0=wins_0,
        wins_1=wins_1,
        ties=ties,
        elapsed_sec=elapsed,
        state_keys=len(shared._n),
        state_action_keys=shared.num_state_action_keys(),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="train_tabular_mc")
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--output", type=Path, default=Path("models/tabular_mc.json"))
    p.add_argument("--team-size", type=int, default=4)
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument("--max-pkm-moves", type=int, default=4)
    p.add_argument("--eps-start", type=float, default=0.1)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=1000)
    args = p.parse_args(argv)

    # Touch RandomBattlePolicy to keep the import alive — it's pulled in from
    # vgc2 for parity with future opponent-warmup variants of this script.
    _ = RandomBattlePolicy

    stats = train(
        episodes=args.episodes,
        output=args.output,
        team_size=args.team_size,
        n_active=args.n_active,
        max_pkm_moves=args.max_pkm_moves,
        eps_start=args.eps_start,
        eps_end=args.eps_end,
        seed=args.seed,
        log_every=args.log_every,
    )
    print(
        f"[train] DONE episodes={stats.episodes} "
        f"wins_0={stats.wins_0} wins_1={stats.wins_1} ties={stats.ties} "
        f"states={stats.state_keys} sa_keys={stats.state_action_keys} "
        f"elapsed={stats.elapsed_sec:.2f}s eps/s={stats.eps_per_sec:.2f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
