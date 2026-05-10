# Reference Study: VGC Battle Agents

This is a citation-backed survey of three Pokemon VGC battle agent projects, written to inform code structure and training decisions for `vgc-ai`. The three projects sit on three different points of the design space: VGC-Bench is the published benchmark with strong PSRO baselines and a 700k-log replay dataset; EliteFurretAI is an ambitious in-progress league-training pipeline with a 125M-parameter transformer; PokéChamp is an LLM-driven minimax agent with VGC support bolted on. All three depend on `poke-env`, and all three have had to extend or work around its doubles primitives in some way. Commit hashes below are the latest as of May 2026 unless otherwise noted.

---

## 1. VGC-Bench

Repo: https://github.com/cameronangliss/vgc-bench — latest commit `0c623b7` (Apr 30 2026). Paper: arXiv 2506.10326 (AAMAS '25). License: MIT-ish (see `LICENSE`).

### A. Code structure & layout

Top-level (commit `0c623b7`):

```
vgc-bench/
  vgc_bench/            # the Python package
    src/                # core library: env, policy, players, llm, teams, utils, callback
    train.py            # RL entry point
    pretrain.py         # behavior-cloning entry point
    eval.py             # cross-evaluation matrices
    play.py             # deploy to live Showdown
    scrape_logs.py / scrape_teams.py / scrape_data.py / logs2trajs.py
    visualize.py
  data/                 # scraped replays / trajectory files (gitignored at scale)
  teams/                # Showdown-format team files
  pokemon-showdown/     # git submodule of the actual showdown server
  unit_tests/  integration_tests/
  *.sh / *.ps1          # train.sh, eval.sh, pretrain.sh, play.sh, train_matchup.{sh,ps1}
  pyproject.toml
```

Entry points are plain Python scripts driven by **argparse**, not yaml. `train.py` exposes `--exploiter`, `--self_play`, `--fictitious_play`, `--double_oracle` (mutually exclusive) plus `--num_envs`, `--total_steps`, `--device`, `--port`; the shell scripts (`train.sh`, `eval.sh`) wrap those flags and also launch the Showdown server as a sibling process. There is no central config object — each script defines its own argparse block. Tests live in `unit_tests/` and `integration_tests/`; the README does not advertise coverage and the CI workflow under `.github/workflows/` is minimal.

### B. poke-env doubles extension

VGC-Bench is the cleanest of the three on this axis because it adopts poke-env's newer `DoublesEnv` directly:

- `vgc_bench/src/env.py:25-31` (commit `0c623b7`):
  ```
  from poke_env.environment import DoublesEnv, SingleAgentWrapper
  ...
  class ShowdownEnv(DoublesEnv):
      """Gymnasium environment for Pokemon VGC doubles battles."""
  ```
  The env keeps the parent's PettingZoo-style two-agent action space (poke-env's `DoublesEnv` exposes `MultiDiscrete([N, N])` where `N=127` in Gen 9, see `poke_env/environment/doubles_env.py:48-79`, and encodes moves as `7 + 5*move_index + target + 20*gimmick` with gimmick values 0=none, 1=mega, 2=z-move, 3=dynamax, 4=tera — so terastallization is supported natively). VGC-Bench does **not** override `_action_to_order` or write its own action mask in `env.py`.
- The observation is overridden. `env.py:36-38` declares `Box(-1, len(moves), shape=(12 * chunk_obs_len,), dtype=np.float32)` (12 = 6 own + 6 opponent slots) and `env.py:116-125` defers the actual embedding to `PolicyPlayer.embed_battle(battle, fake_rating=2000)`.
- The reward in `env.py:107-114` is the textbook sparse terminal `+1 / -1 / 0`.
- `vgc_bench/src/policy_player.py:20-37` imports `AbstractBattle`, `DoubleBattle`, plus the rest of the doubles-relevant enums (`Target`, `SideCondition`, ...) and subclasses `poke_env.player.Player`. `choose_move` asserts `isinstance(battle, DoubleBattle)` and uses `DoublesEnv` helpers to convert action indices to `BattleOrder`s. So the **player** is a plain `Player`, but the **env** is a `DoublesEnv` — they cooperate rather than duplicate.

Observation features: per-Pokemon tokens including ability, item, move ids (separate `nn.Embedding` tables, `policy.py:109-117`), status, type, gender, weather/field flags, plus a `fake_rating=2000` placeholder so the policy generalizes across ladder ratings. No belief state over opponent's hidden team members beyond what poke-env already exposes.

### C. Training methodology

Algorithm family: **PPO inside SB3, wrapped in PSRO meta-game logic.** `train.py` imports `PPO` from `stable_baselines3` and `SubprocVecEnv` for parallel envs; the policy is a custom `MaskedActorCriticPolicy` (see D). The four PSRO variants — pure self-play, fictitious play, double oracle, exploiter — differ only in how opponent checkpoints are sampled into the env at reset.

Key training hyperparameters from `train.py` (around lines 62-73, commit `0c623b7`):
- learning rate schedule `1e-5 * 0.3 ** (1 - p)`
- batch size 512, `gamma=1.0` (undiscounted — relies on terminal-only reward)
- `n_steps` varies per learning style
- `d_model=256`, `choose_on_teampreview=True`

Data and BC: the three-stage `scrape_logs.py → logs2trajs.py → pretrain.py` pipeline harvests Showdown replays (the 700k-replay claim is in the paper abstract and on the Hugging Face mirror `cameronangliss/vgc-bench-models`), converts them to `(obs, action_mask, action)` trajectories, then runs behavior cloning. PSRO runs can be initialized from a BC checkpoint (the README's "BC-then-PSRO" recipe, abbreviated BCSP / BCDO in the paper).

Infra: single-machine, SB3 SubprocVecEnv. The shell scripts launch one local Showdown node per training run on distinct ports. No Ray, no rllib. Eval is a cross-play win-rate matrix (`eval.py` → `visualize.py` heatmaps). The paper reports BCSP/BCDO cross-evaluation win rates between 0.26 and 0.74 across team-pool sizes of 1/4/16/64, with the strong claim that the single-team mirror-match agent beats a professional VGC competitor.

### D. Model architecture

`vgc_bench/src/policy.py` (commit `0c623b7`):
- `AttentionExtractor` (around lines 99-135): three `nn.Embedding` tables for ability, item, move (`embed_len=32`), a linear projection `pokemon_proj = nn.Linear(chunk_obs_len + 6*(embed_len-1), d_model)` that produces a per-Pokemon token, prepended with a learned `cls_token`. The 12 Pokemon tokens (6 own + 6 opponent slots) pass through a Transformer encoder of `embed_layers=3` and `num_heads=4`.
- Heads: MLP policy head on the CLS token producing logits over the `DoublesEnv` action space (per slot); MLP value head. Action mask is applied as `mask = torch.where(mask==1, 0, float('-inf'))` then added to logits before softmax (`policy.py:73-79`).
- Notable choices: action masking integrated into the SB3 distribution (so logprob/entropy stay finite); `gamma=1` paired with sparse rewards; `d_model=256` is small enough to train comfortably on a single GPU. No recurrence; the CLS token is the only cross-Pokemon mixing surface.

Parameter count is not advertised, but at `d_model=256`, 3 layers, 4 heads, plus three 32-dim embedding tables over move/item/ability vocabularies, this is in the low single-digit millions — orders of magnitude smaller than EFAI.

---

## 2. EliteFurretAI

Repo: https://github.com/caymansimpson/EliteFurretAI — latest commit `e442509` (May 2 2026). License: MIT. Note the recent commit `e442509` is literally titled "Fix unit_tests against forked poke-env and resolve RL training NaN", which is a yellow flag for stability.

### A. Code structure & layout

```
EliteFurretAI/
  src/elitefurretai/
    engine/             # battle simulation, rust bindings (rust_battle_engine.py, ENGINE.md)
    etl/                # data ingestion; encoder.py defines MDBO action encoding
    inference/          # battle_inference.py, item_inference.py, speed_inference.py, meta_db.py
    rl/                 # train.py, learner.py, multiprocess_actor.py, opponent_pool.py,
                        # exploiter_train.py, players.py, config.py, model_io.py,
                        # fast_action_mask.py, configs/, analyze/, RL.md
    supervised/         # train.py, train_non_traj.py, train_sweep.py, fine_tune.py,
                        # model_archs.py, behavior_clone_player.py, configs/, sweep_configs/,
                        # SUPERVISED.md
  examples/  scripts/  planning/  docs/
  unit_tests/  conftest.py  pyproject.toml  requirements*.txt
  GETTING_STARTED.md
```

This is the most ambitious layout of the three. Configs are real: `rl/config.py` defines an `RNaDConfig` dataclass that round-trips to yaml (`RNaDConfig().save("config.yaml")`), and `supervised/configs/` plus `supervised/sweep_configs/` host yamls for SL runs. Tests live in `unit_tests/`; the May 2 commit suggests they were broken until very recently. The repo's `GETTING_STARTED.md` explicitly warns the project "isn't completely stable" and tells users to fork — it is not packaged for downstream consumers.

Notable choice: a Rust battle engine (`engine/rust_battle_engine.py`, see `engine/ENGINE.md`) used to bypass JavaScript Showdown for training-time rollouts. This is a significant scope expansion beyond what VGC-Bench attempts.

### B. poke-env doubles extension

EFAI uses a **forked** poke-env (per the May 2 commit message). The standard player classes in `rl/players.py` (commit `e442509`) are:
- `BatchInferencePlayer(Player)` at `src/elitefurretai/rl/players.py:178` — async batched inference player, the actor-side rollout player during RL.
- `RNaDAgent(torch.nn.Module)` at `players.py:877` — neural-net wrapper, not a poke-env class.
- `MaxDamagePlayer(Player)` at `players.py:923` — heuristic baseline.

So EFAI **stays at the `Player` level rather than adopting poke-env's `DoublesEnv`.** Doubles support is built up via:
- `BatchInferencePlayer` reads `battle.available_moves[idx]` per slot and assembles a `DoubleBattleOrder` per turn.
- An external action encoder `MDBO` (Move-Double-Battle-Order) defined in `src/elitefurretai/etl/encoder.py:94-96`, subclassing `BattleOrder`. `MDBO.action_space()` returns `len(_INT_TO_ORDER_MAPPINGS) ** 2 = 45 * 45 = 2025` (encoder.py around 269-280): each slot has 45 atomic actions (moves 1-4 × 5 targets in {-2,-1,0,1,2}, plus per-move terastallize variants, plus 4 switches, plus pass), and the two slots are combined as `slot0 * 45 + slot1`.
- A separate **teampreview action space of 90** (the 90 = C(6,4) * 4! / something close; the model treats it as a discrete head — see the model section).
- Action masking via `rl/fast_action_mask.py` — the RL.md doc claims a "52,000x speedup" over the naive mask, which signals how heavy the mask is.

Dynamax is not the focus (VGC 2024+ has no dynamax); terastallization is encoded as additional per-move action slots; mega/z-move not in scope for Gen 9 VGC.

The `inference/` subpackage is unique to EFAI: `battle_inference.py` tracks belief over hidden info (`item_inference.py`, `speed_inference.py`) — opponent-modeling beyond what `poke-env` ships. Nothing comparable in VGC-Bench or PokéChamp.

### C. Training methodology

Algorithm: **RNaD (Regularized Nash Dynamics)** — the same family DeepMind used for Stratego. The loss form per `rl/RL.md` is `L_total = L_policy + β·L_value − γ·H + α·L_RNaD`, i.e. PPO-style policy gradient with an explicit KL term against a frozen reference snapshot to prevent policy collapse. Two variants exist: standard `RNaDLearner` with one reference, and `PortfolioRNaDLearner` with multiple reference snapshots (`rl/learner.py` around lines 262-268, 1031-1091).

League composition (RL.md, `rl/opponent_pool.py`):
- self vs. recent main checkpoints
- "ghosts" — historical checkpoint pool with PFSP curriculum
- exploiter agents trained periodically (every 5000 updates; winners with >60% win-rate join the pool) — see `rl/exploiter_train.py`
- behavioral-cloning baseline from supervised training
- heuristics: `MaxDamagePlayer`, `RandomPlayer`, `MaxBasePowerPlayer`, `SimpleHeuristic*` from poke-env

Infrastructure: IMPALA-style multi-process; CPU actors and a GPU learner communicating over `multiprocessing.Queue` (`mp_traj_queue` for trajectories, `weight_queues` for policy broadcast). Each worker has its own asyncio event loop and Showdown server connection. Reported peak: ~3100 battles/hour on 4 actors × 4 Showdown servers, 8-core CPU; 4-12 GB RAM. Scaling is sub-linear past 4 actors — the Showdown websocket is the bottleneck, which is the stated motivation for the Rust battle engine.

Data sources: supervised pre-training on human tournament logs in `pkmn`'s Showdown log format converted to `BattleData` objects (no public dataset; the README says "you will have to generate/port your own"), then RL on top.

Eval: cross-play win rates vs. league members. The README quotes 41% top-1 action accuracy and 0.82 correlation with win on the SL model; no ladder Elo number is published.

### D. Model architecture

`src/elitefurretai/supervised/model_archs.py`, lines 2181-2748 (commit `e442509`), defines `TransformerThreeHeadedModel`. Salient details:
- Feature encoder (grouped projection) → early feed-forward stack → **three learned "decision tokens"** (query vectors; `model_archs.py:2508-2526`) prepended/appended to the sequence → Transformer encoder of `transformer_layers=6` (default) → late feed-forward → three heads:
  1. **Turn action head** (`:2539-2547`): logits over the 2025-way `MDBO` action space.
  2. **Teampreview head** (`:2486-2496`): logits over 90 teampreview actions; branches *before* the Transformer to keep its gradients from polluting the in-battle representation.
  3. **Win head / C51** (`:2549-2559`): distributional value, 51 bins over `[-1, 1]`.
- Forward signature `forward(x: (B, T, D)) → (turn_logits, tp_logits, win_value, win_dist_logits)`; an online-RL variant takes a hidden-state context for streaming inference.
- The LSTM cousin `FlexibleThreeHeadedModel` is ~138.8M params and ~529 MB on disk; the transformer is in the same ballpark — RL.md cites "~125M-param transformer" in the project's own framing.
- Embedding dimension reported in RL.md: 9223. C51 distributional value is empirically reported to beat scalar regression on this task.

Design choices to flag: decision tokens (rather than CLS pooling), separate teampreview branch with detached gradients, distributional value head, and supervised-warmstart-then-RL. This is the most architecturally aggressive of the three.

---

## 3. PokéChamp (VGC parts)

Repo: https://github.com/sethkarten/pokechamp — latest commit `0f84c46` (Oct 27 2025). ICML '25 Spotlight. The project is primarily a singles minimax-LLM agent; VGC support exists but is the smaller half of the codebase.

### A. Code structure & layout

```
pokechamp/
  pokechamp/                # core LLM-player implementations
    llm_player.py           # base
    llm_vgc_player.py       # VGC doubles variant
    gpt_player.py / gemini_player.py / llama_player.py / ollama_player.py
    openrouter_player.py / mcp_player.py / timeout_llm_player.py
    minimax_optimizer.py    # MinimaxOptimizer, LocalSimPool, MinimaxCache, OptimizedSimNode
    translate.py            # battle-state → natural language
    prompts.py / prompt_eval.py / depth_translate.py / sim_constants.py
    data_cache.py / visual_effects.py
  bayesian/                 # opponent prediction system
  poke_env/                 # vendored / forked poke-env
  scripts/
    battles/  evaluation/  training/   # incl. run_with_timeout_vgc.py, local_1v1.py
  bots/  tests/  bayesian_dataset/  resource/
```

There is no top-level config system. Entry points take argparse flags like `--battle_format gen9vgc2025regi`. `scripts/run_with_timeout_vgc.py` is the VGC tournament runner. PokéChamp vendors its own copy of `poke_env/`, which means the project can modify poke-env internals freely — but it also means any upstream poke-env fix has to be backported manually.

### B. poke-env doubles extension

The VGC player is `LLMVGCPlayer(Player)` at `pokechamp/llm_vgc_player.py:53` (commit `0f84c46`). Like EFAI, it subclasses `Player`, not `DoublesEnv`. The class is doubles-aware:
- `llm_vgc_player.py:347`: `moves = [move.id for move in battle.available_moves[idx]]` — per-slot move enumeration.
- `llm_vgc_player.py:371`: `tera_format = ' or {"terastallize":"<move_name>"}' if battle.can_tera else ''`.
- `llm_vgc_player.py:373-376`: per-slot switch enumeration with cross-slot duplicate exclusion.
- `llm_vgc_player.py:395-411`: dynamic JSON schema for the LLM that lists `move/target/switch/dynamax/terastallize` options as available.
- `llm_vgc_player.py:220-262`: `_parse_target_string` maps natural-language target descriptions (`"1 = left opponent"`, etc.) to integer targets in `{-2, -1, 0, 1, 2}` — same target convention as poke-env / Showdown.
- `choose_move` (around line 330) returns a `DoubleBattleOrder` assembled slot-by-slot.

Action space is **not** a fixed integer space — actions are strings in a constrained JSON schema, validated and parsed back into poke-env `BattleOrder`s. This is unique among the three.

Minimax: `minimax_optimizer.py` provides infrastructure (sim pool, cache, optimized nodes) but the actual search logic for joint-action enumeration lives in `LLMVGCPlayer.tree_search` (`llm_vgc_player.py:1027`) and `tree_search_optimized` (`:1173`), called via the `io()` method. The LLM scores leaves; minimax-style argmax/argmin picks the joint action. Depth is small (1-2 plies in practice for VGC, given the 2025-action joint space).

### C. Training methodology

There is no training in the RL sense. The LLM is a frozen API call (GPT/Gemini/Llama/etc.); behavior emerges from prompting + minimax. Three things substitute for training:
- **Prompt strategies** (`pokechamp/prompts.py`): input/output, self-consistency, chain-of-thought, tree-of-thought, minimax variants.
- **Bayesian opponent prediction** (`bayesian/`, with `bayesian_dataset/`): a learned prior over what opponents pick.
- **`prompt_eval.py`**: harness for A/B-ing prompt strategies offline.

Evaluation is direct: head-to-head play, ladder ELO, win rate vs heuristic baselines. `scripts/run_with_timeout_vgc.py` runs concurrent VGC battles with a wall-clock timeout — critical because LLM API latency would otherwise dominate.

PokéChamp's value here is **not** a model to imitate but two design ideas: (a) action-as-structured-JSON-constrained-by-available-actions, and (b) a minimax-with-learned-leaf-evaluator decomposition.

---

## Cross-cutting observations

**Where the three agree:**
- All three depend on `poke-env` and at least one of them (PokéChamp) and arguably another (EFAI) vendor or fork it.
- All three target Gen 9 VGC and handle terastallization; none focus on dynamax.
- All three have a custom action encoding for doubles. VGC-Bench inherits poke-env's `MultiDiscrete([127, 127])`; EFAI flattens to a single 2025 head via `MDBO`; PokéChamp uses JSON action schemas. There is no shared standard.
- All three keep terminal-only sparse rewards (`+1/-1`) and let the value head/learner handle credit assignment.
- All three include heuristic baselines (`RandomPlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer`) from poke-env directly.

**Where they disagree:**
- *Base class.* VGC-Bench subclasses `DoublesEnv` and uses it as the gym surface; EFAI and PokéChamp stay at `Player` and assemble actions manually. The `DoublesEnv` route is newer and simpler if you want a clean Gymnasium env.
- *Algorithm.* PPO+PSRO (VGC-Bench) vs. RNaD+league (EFAI) vs. minimax+LLM (PokéChamp). VGC-Bench is the most reproducible; EFAI is the most ambitious; PokéChamp is the only one usable today without training compute.
- *Action representation.* `MultiDiscrete[127,127]` (per-slot, poke-env native) vs. flat `Discrete(2025)` (joint, EFAI) vs. constrained-JSON (PokéChamp). The flat 2025-way head is much harder to mask correctly but matches the joint action distribution honestly.
- *Model size.* O(1M) (VGC-Bench) vs. O(125M) (EFAI). VGC-Bench's results say the small transformer is enough for mirror-match VGC; EFAI is betting capacity will matter at league scale.
- *Configs.* argparse + shell scripts (VGC-Bench) vs. dataclass-yaml (EFAI) vs. argparse (PokéChamp). EFAI is the only one where you can serialize a full training run config.

**Where the field disagrees with itself:**
- VGC-Bench's `gamma=1` (undiscounted, sparse terminal) vs. EFAI's C51 distributional value. Different bets on how to do credit assignment in a long-horizon turn-based game.
- VGC-Bench treats teampreview as the same policy head with a flag; EFAI gives teampreview its own head with detached gradients. The detached design is conservative and probably right if teampreview ends up dominating early gradients.

---

## Recommendations for vgc-ai

These are opinionated. Where I cite a project, that's the influence.

### 1. Code structure

Suggested layout for `src/vgc_ai/`:

```
src/vgc_ai/
  agents/
    base.py              # protocols / ABCs
    random_agent.py      # poke-env RandomPlayer wrapper
    heuristic_agent.py   # MaxBasePowerPlayer wrapper for baseline
    policy_agent.py      # neural-net policy player (the main thing)
  env/
    showdown_env.py      # subclass poke_env.environment.DoublesEnv (per VGC-Bench)
    rewards.py           # sparse + optional shaped rewards
    action_mask.py       # legality mask helpers
  encoding/
    battle_embedding.py  # battle -> tensor; document every feature
    action_space.py      # thin wrapper over DoublesEnv action ids
  models/
    transformer.py       # per-Pokemon tokens + small encoder (VGC-Bench-style first)
    heads.py             # policy/value heads; teampreview head later
  training/
    bc.py                # behavior cloning entry
    rl.py                # PPO entry
    config.py            # pydantic/dataclass configs, yaml-loadable (EFAI-influenced)
  eval/
    crossplay.py         # win-rate matrix utility (VGC-Bench-style)
    ladder.py            # live Showdown play
  data/
    replays.py           # replay scraping + parsing (don't ship the data itself)
  cli.py                 # single entry point with subcommands (train-bc, train-rl, eval, play)
configs/                  # yaml configs at repo root, not under src/
scripts/                  # shell helpers: start-showdown.sh, etc.
tests/
```

Rationale: VGC-Bench's flat `src/` layout is fine for a research project but for `vgc-ai` we want clearer module boundaries. EFAI's per-area subpackages (engine/etl/inference/rl/supervised) is closer to right but overshoots — keep it lighter until we need a Rust engine or an inference subsystem. EFAI's yaml-loadable config dataclasses are worth copying from day one — argparse will become painful once we have 30 hyperparameters.

### 2. poke-env approach

**Adopt VGC-Bench's pattern: subclass `poke_env.environment.DoublesEnv` for training, subclass `poke_env.player.Player` for live-play / opponent roles.** Two reasons:
- It is the only one of the three that doesn't fork or vendor poke-env. We avoid maintaining a fork.
- `DoublesEnv` already encodes target × gimmick × move in its `MultiDiscrete` (poke-env `doubles_env.py:48-79`, `:463-478`). We do not need to invent an `MDBO`.

Concrete plan: `ShowdownEnv(DoublesEnv)` overrides `embed_battle`, `calc_reward`, and (later) `get_action_mask`. The policy player is a `Player` that uses the trained model — match the VGC-Bench `ShowdownEnv` ↔ `PolicyPlayer` split (`vgc_bench/src/env.py:25-31`, `vgc_bench/src/policy_player.py:54`). Document every feature in `encoding/battle_embedding.py` with a comment citing where it comes from in `poke_env.battle.*` — observation drift is the silent killer.

Caveat in our `CLAUDE.md`: poke-env doubles support is "preliminary". If we hit a real gap (e.g. teampreview action format, or a particular speed-tie edge case), prefer to (a) patch upstream and pin to a commit, or (b) carry a *narrow* patch as a documented monkey-patch in `env/`. Do not vendor the whole library.

### 3. First milestone

Smallest end-to-end loop: **random doubles player vs random doubles player, 10 self-play battles, structured log of every move, no neural net.** Specifically:

1. `scripts/start-showdown.sh` boots local Showdown on `ws://localhost:8000`.
2. `RandomDoublesPlayer` subclassing `poke_env.player.RandomPlayer` configured for `gen9vgc2025regi`.
3. A test team file under `teams/` (one team, both sides use it — mirror match).
4. `python -m vgc_ai.cli play --vs random --n 10` runs ten battles, logs `(turn, observation_summary, action, reward)` to `runs/<timestamp>/episodes.jsonl`.
5. A pytest that asserts: ten files appear, every battle terminates, win rate is in `[0.3, 0.7]` (sanity check, not a real test of strength).

This forces us to debug Showdown connection, team registration, doubles teampreview, and logging end-to-end before any ML touches the code.

### 4. Training plan ordering

1. **Heuristic baseline.** Wire up `MaxBasePowerPlayer` and `SimpleHeuristicsPlayer` as opponents and as comparison points. Measure their head-to-head win rate. If we cannot beat `MaxBasePowerPlayer` we have a bug, not an ML problem.
2. **BC on replays.** Scrape Showdown VGC replays (smaller scale than VGC-Bench's 700k initially — even 10-50k is enough to validate the pipeline). VGC-Bench's `scrape_logs.py → logs2trajs.py → pretrain.py` is the reference order. Target: BC win-rate > random and > MaxBasePower on held-out teams.
3. **Self-play PPO from BC checkpoint.** Single team, single Showdown server, SB3 PPO with masked actor-critic — directly model after VGC-Bench's `train.py` until we have something working. Eval as a cross-play matrix `eval/crossplay.py`.
4. **Multi-team generalization.** Expand from 1 team → 4 → 16 (VGC-Bench shows degradation here is real and informative).
5. **Optional: league / RNaD.** Only if PSRO plateaus and we have spare compute. EFAI's IMPALA-style multiprocess actor/learner is the model to copy; do not pre-build it.

Eval methodology from day one: cross-play win-rate matrix (VGC-Bench), plus an occasional live ladder run (PokéChamp's `run_with_timeout_vgc.py` style).

### 5. What to deliberately not copy

- **EFAI's Rust battle engine.** It exists because their actor throughput plateau'd; we will not have that problem for months.
- **EFAI's inference subsystem (item/speed/belief).** Tempting but high-effort. Defer until BC + PPO baselines are working.
- **EFAI's 125M-param transformer.** Start at d_model=128–256, 2–4 layers (closer to VGC-Bench). We can always scale up; we cannot un-spend a week debugging a 125M model that won't train.
- **PokéChamp's vendored poke-env.** Stay on upstream; pin a commit in `pyproject.toml`.
- **VGC-Bench's argparse-only configs.** They are fine at small scale, painful at sweep scale. Use dataclass-yaml from day one.
- **Anyone's "scrape 700k replays" milestone as a prerequisite.** Build the pipeline; run it at 10k first. Don't block on dataset size before you have a working training loop.
- **Premature LLM integration.** VGC-Bench includes an LLM wrapper; we don't need one. If we add LLM-as-baseline later, it slots into `agents/` alongside `random_agent.py`.

---

## Open questions

Things I could not fully resolve via WebFetch and would need the user to decide or require cloning the repos to verify:

1. **Total training compute for VGC-Bench's BCDO result.** The arXiv abstract does not state GPU-hours and the README does not either. If we want to estimate "how long until we can be competitive at 16-team", we need that number — probably in §4 of the paper.
2. **EFAI's forked poke-env divergence.** The May 2 commit says they fixed tests against a fork. Whether they patched real bugs or added features we'd want is unknown without reading the fork. If we hit doubles-related poke-env bugs ourselves, EFAI's fork should be the first place to look for prior fixes.
3. **What `gen9vgc2025regi` means in our setup.** PokéChamp's `local_1v1.py --battle_format gen9vgc2025regi` works against their vendored poke-env. Upstream poke-env's support for the current VGC regulation format (regulation H/I/etc.) needs to be checked against the actual format we want to target — *user decision*: do we target the current official VGC regulation, or freeze on a specific one for reproducibility?
4. **Team source.** VGC-Bench ships `teams/`; EFAI relies on user-supplied teams in PokéPaste format; PokéChamp scrapes. *User decision*: do we curate a small set of strong meta teams, or sample from a larger pool from the start? This decision interacts with the 1/4/16-team generalization curriculum.
5. **Replay licensing.** Showdown replays are user-generated; VGC-Bench publishes a derivative dataset. If we publish anything trained on scraped replays, what's our position on attribution / takedown? Not urgent but worth deciding before we ship a model card.
6. **Whether to do teampreview at all in milestone 1.** Both VGC-Bench (`choose_on_teampreview` flag) and EFAI (separate head) treat it as first-class. Random play is fine for milestone 1, but for milestone 2 onward we need a plan. Cheapest: random teampreview, learn battle policy only. Best long-term: learned teampreview from the start.
