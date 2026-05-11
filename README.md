# vgc-ai

A Pokemon VGC AI competitor targeting the **[IEEE VGC AI Competition 2026](https://cog2026.org/competitions)** at IEEE Conference on Games (CoG) 2026, Madrid, Sept 1-4, 2026.

**Status:** early. Project pivoted from a Pokemon-Showdown-based RL agent (see `docs/archive/study_pokeenv.md`) to the IEEE competition stack on 2026-05-11. Currently building the Competitor skeleton on the `vgc2` framework.

## What this is

The IEEE VGC AI Competition runs on [`pokemon-vgc-engine`](https://gitlab.com/DracoStriker/pokemon-vgc-engine) — a standalone abstracted Pokemon-like simulator with three tracks:

- **Battle Track** — pure battle-policy contest with randomly generated teams; single-elimination bracket.
- **Championship Track** — team-build + battle policy across many ELO matchups.
- **Rules Balance Track** — design rules that induce target move-usage distributions (game-design, not gameplay).

This project will submit entries to all three.

## Approach

Classical search, not deep RL. The 2024 3rd-place submission ([AurelianTactics writeup](https://medium.com/@aureliantactics/vgc-ai-competition-2024-edition-3rd-place-submission-5420d2f6aafe)) demonstrated that the engine's variance breaks deep RL and that tabular Monte Carlo / search beats it. Plan:

1. Get a `Competitor` skeleton running with all three default policies (random baselines).
2. MCTS / tabular MC for the battle policy with a hand-crafted eval function.
3. Genetic-algorithm or LP-based team builder.
4. Rules Balance submission as a low-effort second-track entry.

## Stack

- **Python 3.12** with [uv](https://github.com/astral-sh/uv).
- **[`vgc2`](https://gitlab.com/DracoStriker/pokemon-vgc-engine)** (pinned commit `b0b77f9b`) — the competition framework.
- **`gymnasium`, `numpy`** — explicit deps.
- **`pytest`, `ruff`, `mypy`** — testing, linting, type checking.

## Quickstart

```bash
# Install uv (if you don't have it)
#   macOS / Linux:  curl -LsSf https://astral.sh/uv/install.sh | sh
#   Windows:        powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Install deps (pulls vgc2 from gitlab — needs git on PATH)
uv sync

# Run tests
uv run pytest
```

## Project layout

```
src/vgc_ai/
  competitor.py         # our Competitor subclass — entry point for submission
  policies/
    battle.py           # battle-turn move selection (target: MCTS)
    selection.py        # picking which Pokemon to bring
    teambuild.py        # team building from the roster
tests/                  # pytest tests
scripts/                # CLI utilities
configs/                # configuration files
docs/archive/           # historical material (the Showdown study lives here)
```

## References

- **Foundational paper**: Reis, Reis, Lau, *VGC AI Competition - A New Model of Meta-Game Balance AI Competition*, CoG 2021 — [IEEE 9618985](https://ieeexplore.ieee.org/document/9618985).
- **Adversarial team building**: Reis et al., IEEE ToG 2023 — [IEEE 10115492](https://ieeexplore.ieee.org/document/10115492).
- **Rules Balance Track**: Reis et al., CoG 2025 — [IEEE 11114412](https://ieeexplore.ieee.org/document/11114412).
- **AurelianTactics 2024 writeup**: [Medium](https://medium.com/@aureliantactics/vgc-ai-competition-2024-edition-3rd-place-submission-5420d2f6aafe).
- **Reis baseline agents**: <https://gitlab.com/DracoStriker/vgc-agents>.

## License

MIT — see [LICENSE](LICENSE).
