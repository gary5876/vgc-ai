"""Training driver for ``TabularMCBattlePolicy``.

Two training regimes, switched by ``--warmup``:

1. **Warmup (episodes ``[0, --warmup)``)** — single-sided rollouts vs
   :class:`vgc2.agent.battle.GreedyBattlePolicy`. One side runs the
   learning ``TabularMCBattlePolicy``, the other side runs a fresh
   ``GreedyBattlePolicy``. The tabular side alternates each episode for
   balanced state coverage (even ep → tabular on side 0, odd ep → tabular
   on side 1). Only the tabular side's trajectory is recorded for MC
   updates; the greedy side is not learning.
2. **Self-play (episodes ``[--warmup, --episodes)``)** — both sides are
   the same ``TabularMCBattlePolicy`` instance (shared Q-table). Both
   trajectories are recorded.

In either regime, at each turn each tabular side encodes its own
``StateView`` perspective, picks an ε-greedy joint action via the shared
Q-table, and records ``(state_key, action_idx, 0.0)``. On terminal, the
winning side's last step gets reward ``+1.0`` and the losing side's last
step gets ``-1.0`` (loss signal retained from the
``policy-tabular-mc-train-with-loss-signal`` change). Ties are discarded.

Trajectories are flushed to ``policy.learn(...)`` every ``--batch``
episodes. A checkpoint is saved to ``--model-path`` every
``--checkpoint`` episodes and at exit.

ε decays linearly from 0.1 (episode 0) to 0.05 (episode ``--decay-end``),
then stays flat at 0.05.

Resume: if ``--resume`` is passed and the model file exists, the table is
loaded before training continues.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
from vgc2.agent.battle import GreedyBattlePolicy, get_actions
from vgc2.battle_engine import BattleEngine, BattleRuleParam, State
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import label_teams
from vgc2.util.generator import gen_team

from vgc_ai.policies.tabular_mc import (
    StateKey,
    TabularMCBattlePolicy,
    encode_state,
)


def epsilon_at(episode: int, decay_end: int) -> float:
    """Linear decay 0.1 → 0.05 across ``[0, decay_end]``, flat 0.05 after."""
    if decay_end <= 0 or episode >= decay_end:
        return 0.05
    return 0.1 - (0.1 - 0.05) * episode / decay_end


def _epsilon_greedy_index(
    policy: TabularMCBattlePolicy,
    state_key: StateKey,
    n_actions: int,
    epsilon: float,
    rng: random.Random,
) -> int:
    """Pick a joint-action index using ε-greedy on the policy's current Q-table.

    Reads ``policy._q`` / ``policy._n`` directly: training shares a module-private
    layout with :class:`TabularMCBattlePolicy.decision` and we want the same
    tie-breaking behaviour without paying for a public-API roundtrip.
    """
    if n_actions <= 0:
        return 0
    if rng.random() < epsilon:
        return rng.randrange(n_actions)
    q_row = policy._q.get(state_key)
    n_row = policy._n.get(state_key)
    best_idx = -1
    best_val = float("-inf")
    if q_row is not None and n_row is not None:
        limit = min(n_actions, len(q_row))
        for idx in range(limit):
            if n_row[idx] > 0 and q_row[idx] > best_val:
                best_val = q_row[idx]
                best_idx = idx
    if best_idx < 0:
        return rng.randrange(n_actions)
    return best_idx


def _run_training_episode(
    policy: TabularMCBattlePolicy,
    epsilon: float,
    rng: random.Random,
    np_rng: np.random.Generator,
    params: BattleRuleParam,
    team_size: int,
    n_active: int,
    max_pkm_moves: int,
    opponent_policy: GreedyBattlePolicy | None = None,
    tabular_side: int = 0,
) -> tuple[
    list[tuple[StateKey, int, float]] | None,
    list[tuple[StateKey, int, float]] | None,
    int,
]:
    """Play one training battle. Return ``(traj_0, traj_1, winner)``.

    If ``opponent_policy`` is ``None``, both sides are the learning tabular
    policy (self-play) — both trajectories are returned. Otherwise the
    tabular policy plays ``tabular_side`` and ``opponent_policy`` plays the
    other side; only the tabular trajectory is returned (the opponent side
    is ``None`` regardless of outcome).

    On a non-tie outcome the winning side's last step has reward ``+1.0``
    and the losing side's last step has reward ``-1.0``. On a tie both
    returned trajectories are ``None``.
    """
    team = (
        gen_team(team_size, max_pkm_moves, rng=np_rng),
        gen_team(team_size, max_pkm_moves, rng=np_rng),
    )
    label_teams(team)
    team_view = (TeamView(team[0]), TeamView(team[1]))
    state = State(get_battle_teams(team, n_active))
    view_0 = StateView(state, 0, team_view)
    view_1 = StateView(state, 1, team_view)
    engine = BattleEngine(state, params, debug=False)

    traj_0: list[tuple[StateKey, int, float]] = []
    traj_1: list[tuple[StateKey, int, float]] = []

    tabular_0 = opponent_policy is None or tabular_side == 0
    tabular_1 = opponent_policy is None or tabular_side == 1

    while not engine.finished():
        joint_0 = get_actions((state.sides[0].team, state.sides[1].team)) if tabular_0 else None
        joint_1 = get_actions((state.sides[1].team, state.sides[0].team)) if tabular_1 else None
        if (tabular_0 and not joint_0) or (tabular_1 and not joint_1):
            break
        if tabular_0:
            assert joint_0 is not None
            key_0 = encode_state(view_0)
            idx_0 = _epsilon_greedy_index(policy, key_0, len(joint_0), epsilon, rng)
            traj_0.append((key_0, idx_0, 0.0))
            action_0 = list(joint_0[idx_0])
        else:
            assert opponent_policy is not None
            action_0 = list(opponent_policy.decision(view_0, team_view[1]))
        if tabular_1:
            assert joint_1 is not None
            key_1 = encode_state(view_1)
            idx_1 = _epsilon_greedy_index(policy, key_1, len(joint_1), epsilon, rng)
            traj_1.append((key_1, idx_1, 0.0))
            action_1 = list(joint_1[idx_1])
        else:
            assert opponent_policy is not None
            action_1 = list(opponent_policy.decision(view_1, team_view[0]))
        engine.run_turn((action_0, action_1))

    winner = engine.winning_side
    if winner not in (0, 1):
        return None, None, -1
    if traj_0:
        last = traj_0[-1]
        traj_0[-1] = (last[0], last[1], 1.0 if winner == 0 else -1.0)
    if traj_1:
        last = traj_1[-1]
        traj_1[-1] = (last[0], last[1], 1.0 if winner == 1 else -1.0)
    out_0 = traj_0 if tabular_0 and traj_0 else None
    out_1 = traj_1 if tabular_1 and traj_1 else None
    return out_0, out_1, winner


def train(
    *,
    episodes: int,
    batch_size: int,
    checkpoint_every: int,
    decay_end: int,
    model_path: Path,
    resume: bool,
    seed: int,
    warmup: int = 5000,
    team_size: int = 4,
    n_active: int = 2,
    max_pkm_moves: int = 4,
) -> dict[str, float | int]:
    """Drive training. Returns a small summary dict for logging/reporting."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    params = BattleRuleParam()

    policy = TabularMCBattlePolicy(
        rng_seed=seed,
        model_path=model_path if resume else None,
    )
    greedy_opponent = GreedyBattlePolicy()
    greedy_opponent.set_params(params)

    batch: list[list[tuple[StateKey, int, float]]] = []
    wins_0 = 0
    wins_1 = 0
    ties = 0
    warmup_wins_tabular = 0
    warmup_wins_opponent = 0
    t0 = time.perf_counter()
    last_log = t0

    for ep in range(episodes):
        epsilon = epsilon_at(ep, decay_end)
        in_warmup = ep < warmup
        opponent = greedy_opponent if in_warmup else None
        tabular_side = ep % 2 if in_warmup else 0
        traj_0, traj_1, winner = _run_training_episode(
            policy,
            epsilon=epsilon,
            rng=rng,
            np_rng=np_rng,
            params=params,
            team_size=team_size,
            n_active=n_active,
            max_pkm_moves=max_pkm_moves,
            opponent_policy=opponent,
            tabular_side=tabular_side,
        )
        if in_warmup and winner in (0, 1):
            if winner == tabular_side:
                warmup_wins_tabular += 1
            else:
                warmup_wins_opponent += 1
        if winner == 0:
            wins_0 += 1
        elif winner == 1:
            wins_1 += 1
        else:
            ties += 1
        if traj_0 is not None:
            batch.append(traj_0)
        if traj_1 is not None:
            batch.append(traj_1)
        if (ep + 1) % batch_size == 0 and batch:
            policy.learn(batch)
            batch.clear()
        if (ep + 1) % checkpoint_every == 0:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            policy.save(model_path)
            now = time.perf_counter()
            eps_per_sec = (ep + 1) / (now - t0)
            phase = "warmup" if in_warmup else "self-play"
            print(
                f"[ep {ep + 1}/{episodes} {phase}] "
                f"eps_per_sec={eps_per_sec:.1f} "
                f"epsilon={epsilon:.3f} "
                f"wins_0={wins_0} wins_1={wins_1} ties={ties} "
                f"warmup_tab/opp={warmup_wins_tabular}/{warmup_wins_opponent} "
                f"|Q|={len(policy._q)} "
                f"checkpoint→{model_path}",
                flush=True,
            )
            last_log = now

    # Flush any tail batch and save the final table.
    if batch:
        policy.learn(batch)
        batch.clear()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    policy.save(model_path)

    elapsed = time.perf_counter() - t0
    return {
        "episodes": episodes,
        "elapsed_sec": round(elapsed, 2),
        "eps_per_sec": round(episodes / elapsed, 2) if elapsed > 0 else 0.0,
        "wins_0": wins_0,
        "wins_1": wins_1,
        "ties": ties,
        "warmup": warmup,
        "warmup_wins_tabular": warmup_wins_tabular,
        "warmup_wins_opponent": warmup_wins_opponent,
        "q_size": len(policy._q),
        "last_log": last_log,  # for completeness, harmless
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="train_tabular_mc")
    p.add_argument("--episodes", type=int, default=10000)
    p.add_argument("--batch", type=int, default=100, dest="batch_size")
    p.add_argument("--checkpoint", type=int, default=1000, dest="checkpoint_every")
    p.add_argument("--decay-end", type=int, default=10000)
    p.add_argument(
        "--warmup",
        type=int,
        default=5000,
        help="Run [0, warmup) episodes vs GreedyBattlePolicy before self-play. 0 disables warmup.",
    )
    p.add_argument("--model-path", type=Path, default=Path("models/tabular_mc.json"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--team-size", type=int, default=4)
    p.add_argument("--n-active", type=int, default=2)
    p.add_argument("--max-pkm-moves", type=int, default=4)
    args = p.parse_args(argv)

    summary = train(
        episodes=args.episodes,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        decay_end=args.decay_end,
        warmup=args.warmup,
        model_path=args.model_path,
        resume=args.resume,
        seed=args.seed,
        team_size=args.team_size,
        n_active=args.n_active,
        max_pkm_moves=args.max_pkm_moves,
    )
    print(
        "training done: "
        f"episodes={summary['episodes']} "
        f"elapsed_sec={summary['elapsed_sec']} "
        f"eps_per_sec={summary['eps_per_sec']} "
        f"wins_0={summary['wins_0']} wins_1={summary['wins_1']} ties={summary['ties']} "
        f"warmup={summary['warmup']} "
        f"warmup_tab/opp={summary['warmup_wins_tabular']}/{summary['warmup_wins_opponent']} "
        f"|Q|={summary['q_size']} "
        f"saved→{args.model_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
