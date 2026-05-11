# vgc-ai TASKS

> **Work queue for the autonomous Claude loop running on the GCP VM.**
> The loop reads this file, picks the top **Approved** item with status `todo`, and works only on that.
> The loop NEVER invents tasks. If it has an idea, it appends to **Proposed for human review**.
>
> **You** (the human) maintain the Approved list. **Claude** updates statuses (`todo` → `wip` → `done` / `blocked`).
> Format: `- [STATUS] <slug>: <description>`.

## Approved

<!-- Pre-approved tasks Claude can pick from. Keep ordered by priority — Claude takes the top todo. -->

- [blocked] policy-registry: Extract `POLICIES` out of `src/vgc_ai/cli.py` into a new module `src/vgc_ai/policies/registry.py` so `bench/` and other consumers can import it without depending on the CLI module. Update `bench/run_once.py` and `bench/run_continuous.py` to import from the new location. Update `src/vgc_ai/cli.py` to re-export for backwards compatibility. Acceptance: `python -c "from vgc_ai.policies.registry import POLICIES; print(sorted(POLICIES))"` prints at least `['greedy', 'random', 'tree']`; `uv run pytest`, `uv run ruff check`, `uv run mypy --strict src` all pass; bench/run_once.py still works. No bench evidence required (this is a refactor, not a policy change). Why: refactor implemented on branch `auto/policy-registry-20260511`; ruff/pytest/import-acceptance pass; `uv run mypy --strict src` is pre-existing red on `main` — `src/vgc_ai/competitor.py:19` "Class cannot subclass 'Competitor' (has type 'Any')". Filed as issue #1; unblock by fixing that, then re-mark `todo`.

- [done] bench-turn-timing: Add `avg_turn_ms_a` and `avg_turn_ms_b` columns to the bench output (both `run_once.py` JSON and `run_continuous.py` CSV). To get per-turn timing, instrument `vgc_ai/eval/duel.py` to wrap each policy's `get_action` call (or equivalent) with `time.perf_counter()` and accumulate. If vgc2's `BattlePolicy` interface doesn't expose a hookable method, mark `blocked` with a concrete reason — do not monkey-patch vgc2 internals. Acceptance: leaderboard.csv has the two new columns populated with non-zero values; existing rows are preserved (the writer should handle the schema migration via header-rewrite on first new round). No policy bench evidence required. PR #2.

- [blocked] policy-heuristic-eval: Implement a heuristic state-evaluation function at `src/vgc_ai/eval/heuristic.py` returning a single float for "side 0's expected advantage", based on: remaining HP totals, fainted count, simple type-advantage check using vgc2's type chart, and currently-active Pokemon's hp/maxhp ratio. Wire it into a new `HeuristicBattlePolicy` (registered as `heuristic` in `POLICIES`) that, for each action choice, simulates the action 1 ply forward and picks the action maximizing the heuristic. Acceptance: 200-game bench `heuristic` vs `greedy` shows `win_rate_a > 0.5` AND `ci95_low > 0.5` AND `avg_battle_ms < 5000`. If `avg_battle_ms` is over budget, mark blocked — do not weaken the heuristic just to fit budget. Why: implemented on branch `auto/policy-heuristic-eval-20260511`; ruff/pytest/import-acceptance pass; 200-game bench `heuristic` vs `greedy`: win_rate_a=0.545, **ci95_low=0.4758** (< 0.5 gate), ci95_high=0.6125, avg_battle_ms=197 (well under budget). Win rate is positive but the Wilson lower bound is below 0.5, so the edge over Greedy is not statistically distinguishable from a coin flip at n=200. One weight-tuning iteration (FAINT_WEIGHT 3→6, MATCHUP_WEIGHT 0.5→1.0) at n=100 returned 50-50, so naïve weight bumps don't help. Unblock by: (a) tightening the heuristic — try a damage-prediction term in eval, or use ZERO_RNG-deterministic rollouts in the forward sim to reduce stochastic noise; or (b) widening n to ~500+ if true win rate is genuinely ~0.55, to push ci95_low above 0.5.

## Proposed for human review

<!-- Claude appends here when it has an idea but is not allowed to act on it. You move them to Approved (or delete) when you triage. -->

- policy-heuristic-eval-deterministic-rollout: Unblock path (a) for `policy-heuristic-eval`. Modify the 1-ply forward simulation inside `HeuristicBattlePolicy` to use vgc2's `ZERO_RNG` (or equivalent deterministic damage/accuracy path) so the eval comparison across candidate actions isn't dominated by simulator stochasticity. Keep the heuristic weights as currently checked in on `auto/policy-heuristic-eval-20260511`. Acceptance: same gate as the original task (n=200, `win_rate_a > 0.5`, `ci95_low > 0.5`, `avg_battle_ms < 5000`). Rationale: weight-tuning at n=100 returned 50-50, so the issue is more likely sim noise than weight calibration.

- policy-heuristic-eval-n500: Unblock path (b) for `policy-heuristic-eval`. Re-run the existing `heuristic` vs `greedy` bench on the auto/policy-heuristic-eval-20260511 branch at n=500 to see if the Wilson lower bound clears 0.5 (point estimate was 0.545 at n=200, ci95_low=0.4758). Acceptance: `bench/results/policy-heuristic-eval.json` at n=500 satisfies the standard gate. Cheap: ~5× the prior runtime (200 games took seconds). Do this before (a) if (a) turns out to be invasive.

- mypy-competitor-subclass-fix: Address issue #1 — `src/vgc_ai/competitor.py:19` "Class cannot subclass 'Competitor' (has type 'Any')" — so `mypy --strict src` is green on `main`. This unblocks `policy-registry`. Likely fix: add a `type: ignore[misc]` on the class line with a comment pointing to vgc2 being untyped, OR add a minimal `Protocol`/stub for `vgc2.competition.Competitor` under a `stubs/` dir referenced from `pyproject.toml`. Acceptance: `uv run mypy --strict src` exits 0; existing `uv run pytest` stays green. No bench evidence required (typing-only).

## Done

<!-- Most-recent first. Prune manually when this gets long. -->

- [done] policy-random-bench-baseline: `RandomBattlePolicy` was already exposed by vgc2 and registered in `vgc_ai.cli.POLICIES`. Bench round confirms Greedy ~90% vs Random in doubles. No code change needed. (Pre-loop, marked done at deploy.)
- [done] bench-leaderboard: Implemented in `bench/run_continuous.py` (B2 commit). Appends rows to `bench/leaderboard.csv` with the schema in the file's header.
- [done] bench-smoke: Implemented in `bench/run_once.py` (B2 commit). Prints structured JSON with `policy_a`, `policy_b`, `wins_a`, `wins_b`, `ties`, `win_rate_a`, Wilson `ci95_low`/`ci95_high`, `elapsed_sec`, `avg_battle_ms`. Smoke-tested on the VM (greedy vs random: 20-0-0 in 0.4s).

## Blocked

<!-- Tasks that hit an obstacle. Claude moves them here with a "Why: ..." line so you can unblock. -->
