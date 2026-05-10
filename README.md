# vgc-ai

A reinforcement-learning battle agent for Pokemon VGC (doubles format) on [Pokemon Showdown](https://pokemonshowdown.com/).

**Status:** very early. Project scaffolding only — no agent yet.

## What this is

VGC (Video Game Championships) is the official competitive Pokemon doubles format. This project aims to train an RL agent that can play Gen 9 VGC regulation matches on a locally-hosted Pokemon Showdown server.

The plan, roughly:
1. Connect to a local Pokemon Showdown server via [poke-env](https://github.com/hsahovic/poke-env).
2. Implement baseline heuristic / random agents for sanity checks.
3. Train neural policies (PyTorch) using self-play and/or behavior cloning on human replays.
4. Evaluate against published baselines.

## Stack

- **Python 3.12** with [uv](https://github.com/astral-sh/uv) for env + dep management
- **poke-env** — Showdown client / Gymnasium-style wrapper (de facto standard in this space)
- **PyTorch** — model + training
- **pytest / ruff / mypy** — testing, linting, type checking

## Quickstart

```bash
# Install uv (if you don't have it)
#   macOS / Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
#   Windows:        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install deps into a managed venv
uv sync

# Run the smoke test
uv run pytest
```

You will also need a local Pokemon Showdown server. See `.claude/commands/setup-showdown.md` (or [the upstream docs](https://github.com/smogon/pokemon-showdown)) for setup.

## Project layout

```
src/vgc_ai/
  agents/      # Player implementations (heuristic, RL, etc.)
  models/      # Neural network architectures
  training/    # Training loops
tests/         # pytest tests
scripts/       # CLI utilities (run ladder games, eval, etc.)
configs/       # Training / eval configs
```

## Reference projects

Prior art worth knowing about:

- [**VGC-Bench**](https://github.com/cameronangliss/vgc-bench) (AAMAS '25) — first VGC doubles RL benchmark; ships >700k human battle logs and PSRO baselines.
- [**EliteFurretAI**](https://github.com/caymansimpson/EliteFurretAI) — 125M-param transformer for VGC doubles, supervised + RNaD RL.
- [**PokéChamp**](https://github.com/sethkarten/pokechamp) (ICML '25 spotlight) — LLM minimax agent with a VGC variant (`gen9vgc2025regi`).
- [**Metamon**](https://github.com/UT-Austin-RPL/metamon) — offline RL on millions of human replays; singles only, doubles in development.
- [**foul-play**](https://github.com/pmariglia/foul-play) — strongest non-ML public bot, search-based.

## License

MIT — see [LICENSE](LICENSE).
