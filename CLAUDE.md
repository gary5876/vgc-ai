# CLAUDE.md

Project rules for Claude Code. Read this before any work.

## Project

`vgc-ai` is a competitor for the **IEEE VGC AI Competition 2026** (4th edition), held at IEEE Conference on Games (CoG) 2026 in Madrid, Sept 1-4, 2026. The competition is organized by Simão Reis (Vortex-CoLab / LIACC, Univ. Porto). Goal: top-3 finish across the three tracks.

The competition runs on the **`vgc2`** framework ([`pokemon-vgc-engine`](https://gitlab.com/DracoStriker/pokemon-vgc-engine)) — a standalone abstracted Pokemon-like simulator. **It is NOT Pokemon Showdown.** Supports both singles and doubles via the `n_active` parameter; the **Battle Track default is `n_active=2` (doubles)** per `organization/run_battle_track.py`. Fictional roster, parametric moves. Mechanics are simplified for tractable AI work.

## Decisions (locked, ask before changing)

- **Target competition**: IEEE VGC AI Competition 2026, all three tracks (Battle, Championship, Rules Balance).
- **Framework**: `vgc2` from `pokemon-vgc-engine`, pinned to commit `b0b77f9b` in `pyproject.toml`. Update only when Reis announces the 4th-edition official pin.
- **Approach**: **MCTS / tabular Monte Carlo + heuristic eval for battle policy**; **GA or LP for team building**. Not deep RL — the 2024 3rd-place writeup (AurelianTactics) demonstrated deep RL fails under the engine's randomness, and tabular MC won.
- **Default battle policy**: `GreedyBattlePolicy` (submission-viable). `TreeSearchBattlePolicy` is too slow for doubles (~11 s/turn measured at `max_depth=1, n_active=2`) and is reserved for **local benchmarking only**, not the submission. Any new battle policy must beat Greedy AND fit a sub-second-per-turn budget.
- **Submission**: Python `Competitor` subclass (see `vgc2.competition.Competitor`), submitted via Google Form per the 2025 procedure. Confirm format with organizers once 4th-edition rules wiki is published.

## Stack

- Python **3.12**, managed via **uv**
- `vgc2` (the competition framework) — pinned git dep, no PyTorch needed
- `gymnasium`, `numpy` — explicit deps
- `pytest`, `ruff`, `mypy` for tooling

No web server, no websocket, no neural-network dependency unless we decide to add one later for a learned eval function.

## Conventions

- **Source layout**: code in `src/vgc_ai/`, tests in `tests/`. Imports use absolute paths (`from vgc_ai.policies import ...`).
- **Formatting**: `ruff format` (configured in `pyproject.toml`).
- **Linting**: `ruff check` should pass. Fix lint issues, don't `# noqa` them unless there's a real reason.
- **Type hints**: required on public functions and method signatures. `mypy --strict` is the bar. `vgc2` is untyped — annotate our interfaces against its runtime behavior.
- **Comments**: only when *why* is non-obvious. Don't narrate *what* the code does.

## Scope discipline (important)

- **One task, one concern.** If you spot an unrelated issue while working, *note it in your response* — do not fix it in the same change.
- **Don't refactor adjacent code** unless asked. "While I was in there" is how PRs grow legs.
- **No new dependencies without explicit approval.** If you think a new dep is needed, stop and ask.
- **No silent scope expansion.** If the task as stated turns out to require more than expected, surface the gap before doing the extra work.

## Never do

- **Don't add Pokemon mechanics from memory.** The `vgc2` engine has its own move/type/stat semantics. Read the source — don't assume Gen 9 VGC rules apply.
- **Don't reach for deep RL by default.** The 2024 winner's evidence is that classical search beats neural in this engine due to high variance. Prove tabular/search baselines are insufficient before training networks.
- **Don't skip verification by reading code instead of running it.** "It should work" is not "it works." Run the test, run the script, paste the output.
- **Don't trust your own summary.** After non-trivial changes, ask the user to read the diff, or show it explicitly.
- **Don't commit large simulation artifacts.** Training logs, ELO traces, MC visit tables — all gitignored.

## Reference materials

Source-of-truth for the framework and competition:

- **Framework repo**: <https://gitlab.com/DracoStriker/pokemon-vgc-engine> (pinned commit `b0b77f9b`). The `template/`, `tutorial/`, and `organization/` folders are the API surface.
- **Companion baselines**: <https://gitlab.com/DracoStriker/vgc-agents> — Reis-authored Team Builder and Meta-Game Balance baselines.
- **CoG 2025 competition site (last edition)**: <https://cog2025.inesc-id.pt/vgc-ai-competition/>.
- **CoG 2026 competitions list**: <https://cog2026.org/competitions>. The 4th-edition rules wiki page is not yet published as of 2026-05-11.
- **Discord**: <https://discord.gg/GwKHqXpdjf>. Email: simao.reis@vortex-colab.com.

Foundational papers (cite when relevant):

- Reis, Reis, Lau — *VGC AI Competition - A New Model of Meta-Game Balance AI Competition*, CoG 2021 — [IEEE 9618985](https://ieeexplore.ieee.org/document/9618985).
- Reis et al. — *An Adversarial Approach for Automated Pokemon Team Building*, IEEE ToG 2023 — [IEEE 10115492](https://ieeexplore.ieee.org/document/10115492).
- Reis et al. — *A New Rules Balance Track*, CoG 2025 — [IEEE 11114412](https://ieeexplore.ieee.org/document/11114412).

Strong prior result to study:

- AurelianTactics, *VGC AI Competition 2024 Edition 3rd Place Submission* — [Medium](https://medium.com/@aureliantactics/vgc-ai-competition-2024-edition-3rd-place-submission-5420d2f6aafe). Tabular first-visit Monte Carlo, ~30M trials, 11-dim collapsed observation space.

See `REFERENCE_STUDY.md` for the consolidated competition study and `docs/archive/study_pokeenv.md` for the historical record of why we are NOT on Pokemon Showdown.

## Tooling commands

```bash
uv sync                          # install / sync deps (pulls vgc2 from gitlab)
uv run pytest                    # run tests
uv run pytest -k <name>          # run one test
uv run ruff format .             # format
uv run ruff check .              # lint
uv run mypy src                  # type check
```

## Hooks

`.claude/settings.json` includes:
- **PostToolUse** on Edit/Write/MultiEdit → `uv run ruff format .`
- **Stop** → `uv run ruff check src tests`

These require `uv sync` to have been run at least once. If hooks fail with "command not found," restart Claude Code so it picks up `uv` on PATH, then run `uv sync`.

## Slash commands

- `/review-diff` — read staged + unstaged changes, summarize every change, flag risks
- `/scope-check` — compare current changes to the originally-stated task
