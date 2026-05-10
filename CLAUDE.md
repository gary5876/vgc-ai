# CLAUDE.md

Project rules for Claude Code. Read this before any work.

## Project

`vgc-ai` is an RL battle agent for **Pokemon VGC doubles** (Gen 9 regulation format) on Pokemon Showdown. We connect a Python agent to a locally-hosted Showdown server via `poke-env`. The goal is a competitive doubles agent — not singles, not random battles.

## Decisions (locked, ask before changing)

- **VGC regulation format**: track the **current official** VGC regulation (whatever the Pokemon Co. has active). Trade-off accepted: trained models may age out when regs rotate.
- **Team source**: **4-8 curated meta teams** maintained in `teams/` (PokéPaste-style). Not scraped, not single-team. Start at 1-2 teams for milestone 1, expand to 4-8 once training works.
- **Reference pattern**: follow VGC-Bench's `DoublesEnv` subclass approach (no poke-env fork). See `REFERENCE_STUDY.md` for the full rationale.

## Stack

- Python **3.12**, managed via **uv**
- `poke-env` for the Showdown client (note: doubles support is "preliminary" upstream — expect to extend it)
- PyTorch for models
- pytest, ruff, mypy

Showdown runs locally over websocket at `ws://localhost:8000/showdown/websocket` by default.

## Conventions

- **Source layout**: code in `src/vgc_ai/`, tests in `tests/`. Imports use absolute paths (`from vgc_ai.agents import ...`).
- **Formatting**: `ruff format` (configured in `pyproject.toml`). Don't argue about style.
- **Linting**: `ruff check` should pass. Fix lint issues, don't `# noqa` them unless there's a real reason.
- **Type hints**: required on public functions and method signatures. `mypy --strict` is the bar.
- **Tests**: pytest, async tests use `pytest-asyncio` (already configured with `asyncio_mode = "auto"`).
- **Comments**: only when *why* is non-obvious. Don't narrate *what* the code does.

## Scope discipline (important)

- **One task, one concern.** If you spot an unrelated issue while working, *note it in your response* — do not fix it in the same change.
- **Don't refactor adjacent code** unless asked. "While I was in there" is how PRs grow legs.
- **No new dependencies without explicit approval.** If you think a new dep is needed, stop and ask.
- **No silent scope expansion.** If the task as stated turns out to require more than expected, surface the gap before doing the extra work.

## Never do

- **Don't fake or mock data when poke-env is unavailable.** If you can't connect to Showdown, say so — don't invent battle states. Mocked tests that fool you are worse than no tests.
- **Don't skip verification by reading code instead of running it.** "It should work" is not "it works." Run the test, run the script, paste the output.
- **Don't trust your own summary.** After non-trivial changes, ask the user to read the diff, or show it explicitly.
- **Don't add VGC mechanics from memory.** Pokemon mechanics are full of edge cases (terrain × ability × item interactions, etc.). Always cite the source (Showdown's `data/` files, Smogon's calc, `@smogon/calc`) or verify against a known case.
- **Don't commit data, checkpoints, or replay archives.** They're gitignored — keep it that way.

## Reference projects (study, don't copy blindly)

When you're unsure how to structure something, check what these projects did:

- **VGC-Bench** ([github.com/cameronangliss/vgc-bench](https://github.com/cameronangliss/vgc-bench)) — the published VGC doubles RL benchmark. Look here for: PSRO baselines, behavior cloning setup, eval methodology.
- **EliteFurretAI** ([github.com/caymansimpson/EliteFurretAI](https://github.com/caymansimpson/EliteFurretAI)) — current SOTA for VGC doubles. Look here for: how to extend poke-env for VGC, transformer model design.
- **PokéChamp** ([github.com/sethkarten/pokechamp](https://github.com/sethkarten/pokechamp)) — LLM minimax with `gen9vgc2025regi` support. Look here for: action sampling, opponent modeling.
- **poke-env** itself ([github.com/hsahovic/poke-env](https://github.com/hsahovic/poke-env)) — the library we depend on. Read its source when something doesn't behave as expected.

When citing prior art in code or PRs, link to the file and commit, not just the repo.

## Tooling commands

```bash
uv sync                          # install / sync deps
uv run pytest                    # run tests
uv run pytest -k <name>          # run one test
uv run ruff format .             # format
uv run ruff check .              # lint
uv run mypy src                  # type check
```

## Hooks

`.claude/settings.json` includes:
- **PostToolUse** on Edit/Write/MultiEdit → `uv run ruff format .` (idempotent, fast)
- **Stop** → `uv run ruff check src tests` (non-blocking; surfaces lint issues at end of turn)

These require `uv sync` to have been run at least once. If hooks fail with "command not found," run `uv sync`.

## Slash commands

- `/review-diff` — read staged + unstaged changes, summarize every change, flag risks
- `/scope-check` — compare current changes to the originally-stated task
- `/setup-showdown` — walk through setting up a local Pokemon Showdown server
