# vgc-ai TASKS

> **Work queue for the autonomous Claude loop running on the GCP VM.**
> The loop reads this file, picks the top **Approved** item with status `todo`, and works only on that.
> The loop NEVER invents tasks. If it has an idea, it appends to **Proposed for human review**.
>
> **You** (the human) maintain the Approved list. **Claude** updates statuses (`todo` → `wip` → `done` / `blocked`).
> Format: `- [STATUS] <slug>: <description>`.

## Approved

<!-- Pre-approved tasks Claude can pick from. Keep ordered by priority — Claude takes the top todo. -->

- [wip] policy-registry: Extract `POLICIES` out of `src/vgc_ai/cli.py` into a new module `src/vgc_ai/policies/registry.py` so `bench/` and other consumers can import it without depending on the CLI module. Update `bench/run_once.py` and `bench/run_continuous.py` to import from the new location. Update `src/vgc_ai/cli.py` to re-export for backwards compatibility. Acceptance: `python -c "from vgc_ai.policies.registry import POLICIES; print(sorted(POLICIES))"` prints at least `['greedy', 'random', 'tree']`; `uv run pytest`, `uv run ruff check`, `uv run mypy --strict src` all pass; bench/run_once.py still works. No bench evidence required (this is a refactor, not a policy change).

- [todo] bench-turn-timing: Add `avg_turn_ms_a` and `avg_turn_ms_b` columns to the bench output (both `run_once.py` JSON and `run_continuous.py` CSV). To get per-turn timing, instrument `vgc_ai/eval/duel.py` to wrap each policy's `get_action` call (or equivalent) with `time.perf_counter()` and accumulate. If vgc2's `BattlePolicy` interface doesn't expose a hookable method, mark `blocked` with a concrete reason — do not monkey-patch vgc2 internals. Acceptance: leaderboard.csv has the two new columns populated with non-zero values; existing rows are preserved (the writer should handle the schema migration via header-rewrite on first new round). No policy bench evidence required.

- [todo] policy-heuristic-eval: Implement a heuristic state-evaluation function at `src/vgc_ai/eval/heuristic.py` returning a single float for "side 0's expected advantage", based on: remaining HP totals, fainted count, simple type-advantage check using vgc2's type chart, and currently-active Pokemon's hp/maxhp ratio. Wire it into a new `HeuristicBattlePolicy` (registered as `heuristic` in `POLICIES`) that, for each action choice, simulates the action 1 ply forward and picks the action maximizing the heuristic. Acceptance: 200-game bench `heuristic` vs `greedy` shows `win_rate_a > 0.5` AND `ci95_low > 0.5` AND `avg_battle_ms < 5000`. If `avg_battle_ms` is over budget, mark blocked — do not weaken the heuristic just to fit budget.

## Proposed for human review

<!-- Claude appends here when it has an idea but is not allowed to act on it. You move them to Approved (or delete) when you triage. -->

## Done

<!-- Most-recent first. Prune manually when this gets long. -->

- [done] policy-random-bench-baseline: `RandomBattlePolicy` was already exposed by vgc2 and registered in `vgc_ai.cli.POLICIES`. Bench round confirms Greedy ~90% vs Random in doubles. No code change needed. (Pre-loop, marked done at deploy.)
- [done] bench-leaderboard: Implemented in `bench/run_continuous.py` (B2 commit). Appends rows to `bench/leaderboard.csv` with the schema in the file's header.
- [done] bench-smoke: Implemented in `bench/run_once.py` (B2 commit). Prints structured JSON with `policy_a`, `policy_b`, `wins_a`, `wins_b`, `ties`, `win_rate_a`, Wilson `ci95_low`/`ci95_high`, `elapsed_sec`, `avg_battle_ms`. Smoke-tested on the VM (greedy vs random: 20-0-0 in 0.4s).

## Blocked

<!-- Tasks that hit an obstacle. Claude moves them here with a "Why: ..." line so you can unblock. -->
