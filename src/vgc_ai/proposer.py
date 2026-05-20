"""Compound-proposer loop — invents new strategies and ships them as PRs.

This is the autonomous-ideation half of the loop. Per wake (driver runs
hourly on the VM), the proposer:

1. **Pauses** if any reviewer-opened or proposer-opened PR is in flight
   (shared with the reviewer's pause-while-open guard). Zero Claude
   tokens.
2. **Refuses** if today's combined Claude invocation count hits the
   daily cap (shared budget with the reviewer). Zero tokens.
3. **Shapes a bench-context summary** — current default per track, the
   strongest and weakest existing entries, recent pooled head-to-head
   stats. Reads ``bench/strategies/{track}.csv`` plus the registry. Pure
   Python.
4. **Picks the most-contested track** — the track whose current default
   has the smallest win-rate margin against its closest opponent. That's
   the track with the most room to improve.
5. **Reads the attempt log** at ``ops/proposer_attempts.csv`` so Claude
   knows what's already been tried (and may have failed).
6. **Fires one-shot** ``claude -p`` with the bench context, attempt log,
   and an instruction to invent + implement a new compound for the
   chosen track. PR opens with a ``NEW COMPOUND`` body block.

Token discipline: per [[feedback_loop_token_budget]], the proposer
shares the reviewer's daily cap and never holds a persistent session.
Each fire is a one-shot ``claude -p`` invocation; the prompt is shaped
to be informative but bounded (~10-20K input tokens, heavier than the
reviewer's ~5K because Claude is writing code, not just swapping a
constant).

The shell loop lives in ``ops/run_proposer.sh`` (default cadence: 1h).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vgc_ai.reviewer import (
    DEFAULT_BUDGET_PATH,
    DEFAULT_CSV_DIR,
    DEFAULT_DAILY_CAP,
    DEFAULT_RECENT_ROWS,
    daily_claude_calls,
    list_open_reviewer_prs,
    load_recent_rows,
    wilson_ci_95,
)
from vgc_ai.strategies import (
    BATTLE_DEFAULT,
    BATTLE_STRATEGIES,
    CHAMPIONSHIP_DEFAULT,
    CHAMPIONSHIP_STRATEGIES,
)

DEFAULT_ATTEMPTS_PATH = Path("ops/proposer_attempts.csv")
CLAUDE_TIMEOUT_SEC = 1800  # proposer prompts take longer than reviewer's
NEW_COMPOUND_MARKER = "NEW COMPOUND"

GhRunner = Callable[[list[str]], str]
ClaudeRunner = Callable[[list[str], str], "tuple[int, str, str]"]


def _default_claude_runner(cmd: list[str], _prompt: str) -> tuple[int, str, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=CLAUDE_TIMEOUT_SEC)
    return result.returncode, result.stdout, result.stderr


def previous_attempts(attempts_path: Path) -> list[dict[str, str]]:
    """Read the proposer-attempts log; empty list if file is missing."""
    if not attempts_path.exists():
        return []
    with attempts_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _log_attempt(
    attempts_path: Path,
    track: str,
    compound_name: str,
    exit_code: int,
) -> None:
    attempts_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not attempts_path.exists()
    with attempts_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp", "track", "compound_name", "exit_code"])
        w.writerow(
            [
                datetime.now(UTC).isoformat(timespec="seconds"),
                track,
                compound_name,
                exit_code,
            ]
        )


def summarize_track(
    csv_path: Path,
    candidates: list[str],
    default: str,
    n_recent: int = DEFAULT_RECENT_ROWS,
) -> dict[str, Any]:
    """Pool recent rows and report per-candidate stats vs the default.

    Returns ``{"default": str, "rows_seen": int, "candidates": [...]}``
    where each candidate dict has ``name``, ``pooled_n``, ``pooled_wins``,
    ``win_rate``, ``ci95_low``, ``ci95_high``, ``vs_default_margin``.

    Empty CSV → empty candidates list (still includes ``default``).
    """
    rows = load_recent_rows(csv_path, n_recent)
    out: dict[str, Any] = {"default": default, "rows_seen": len(rows), "candidates": []}
    if not rows:
        return out
    for cand in candidates:
        if cand == default:
            continue
        pair_rows = [
            r for r in rows if r.get("strategy_a") == cand and r.get("strategy_b") == default
        ]
        if not pair_rows:
            continue
        wins = sum(int(r["wins_a"]) for r in pair_rows)
        n = sum(int(r["wins_a"]) + int(r["wins_b"]) for r in pair_rows)
        if n == 0:
            continue
        win_rate = wins / n
        ci_lo, ci_hi = wilson_ci_95(wins, n)
        out["candidates"].append(
            {
                "name": cand,
                "pooled_n": n,
                "pooled_wins": wins,
                "win_rate": round(win_rate, 4),
                "ci95_low": round(ci_lo, 4),
                "ci95_high": round(ci_hi, 4),
                "vs_default_margin": round(0.5 - win_rate, 4),
            }
        )
    return out


def pick_most_contested_track(summaries: dict[str, dict[str, Any]]) -> str:
    """Choose the track where the default has the smallest win-rate cushion.

    Smaller cushion = more room for a new compound to land a real win.
    If a track has no candidates measured yet, it's the most contested by
    default (nothing has been tested against the default at all).
    """
    best_track = ""
    best_cushion = float("inf")
    for track, s in summaries.items():
        cands = s.get("candidates", [])
        if not cands:
            return track  # untested track wins immediately
        # Default wins-vs-each-cand; the smallest (most threatening) candidate
        # margin tells us how much cushion the default has. Smaller = more
        # contested.
        worst_for_default = min(c["vs_default_margin"] for c in cands)
        if worst_for_default < best_cushion:
            best_cushion = worst_for_default
            best_track = track
    return best_track or "battle"


def build_prompt(
    track: str,
    summaries: dict[str, dict[str, Any]],
    attempts: list[dict[str, str]],
) -> str:
    """Construct the proposer's one-shot prompt.

    Self-contained: the proposer relies on Claude's per-cwd auto-load of
    CLAUDE.md for project conventions, so the prompt only adds bench
    context, the attempt log, and the implementation contract.
    """
    today = datetime.now(UTC).strftime("%Y%m%d")
    summary_blob = json.dumps(summaries, indent=2)
    attempts_blob = (
        "(no prior attempts)"
        if not attempts
        else "\n".join(
            f"- {a['timestamp']} {a['track']} {a['compound_name']} exit={a['exit_code']}"
            for a in attempts[-20:]
        )
    )

    return f"""You are proposing AND implementing a new compound strategy for the vgc-ai project.

Current track to improve: **{track}**

Bench context (recent pooled head-to-head, per track):
```
{summary_blob}
```

Prior proposer attempts (do NOT re-propose the same compound name):
```
{attempts_blob}
```

Task (read CLAUDE.md if you haven't; it's auto-loaded from this cwd):

1. **Invent** a new compound strategy for the {track} track. It must be meaningfully different from existing entries listed under `summaries["{track}"]`. Read the existing implementations under `src/vgc_ai/policies/`, `src/vgc_ai/eval/`, and `src/vgc_ai/strategies/registry.py` first.

2. **Implement it** as new policy class(es). Place new code under `src/vgc_ai/policies/`, `src/vgc_ai/eval/`, or `src/vgc_ai/strategies/`. You MAY modify existing files in those directories if needed, but do not touch `ops/`, `bench/`, `scripts/`, `.github/`, top-level config, or anything outside `src/vgc_ai/**` and `tests/**`.

3. **Add tests** in `tests/test_*.py` for any new functions/classes. Tests should run fast and not require network or external data.

4. **Register** the new compound in `src/vgc_ai/strategies/registry.py` by appending a new `{track.capitalize()}Strategy(...)` entry to the matching registry. Pick a snake_case name that's not in the prior-attempts list.

5. **Run** `uv run ruff format src tests && uv run ruff check src tests && uv run mypy --strict src && uv run pytest`. Fix any failures with the smallest possible change — do not refactor unrelated code.

6. **Branch** `auto/compound-{{compound_name}}-{today}`, commit per CLAUDE.md format (subject: `feat(policy): add {{compound_name}} compound for {track} track`), push, open PR via `gh pr create`.

7. **PR body** must include this exact block (the auto-handler parses it):

```
NEW COMPOUND
track={track}
compound_name=<your new compound's registry key>
rationale=<one sentence about why this should beat the current default>
```

Constraints (the auto-handler enforces these — violating them gets the PR closed):

- Scope: only files under `src/vgc_ai/policies/`, `src/vgc_ai/eval/`, `src/vgc_ai/strategies/`, or `tests/`. Anything else closes the PR.
- CI green: `ruff format --check`, `ruff check`, `mypy --strict`, `pytest` all pass.
- No bench evidence required at PR time — the compound is brand new. The bench loop will measure it after merge; the reviewer will promote it (via a separate BENCH GATE PR) if it actually beats the default.

Aim for compounds that have a credible theoretical reason to beat the default — see existing failure modes recorded as `[blocked]` entries in TASKS.md if you want concrete leverage points (encoder refinement, damage-prediction terms, LP-minimax selection, etc.).

Exit 0 only on success. If pytest fails after your edits and you cannot fix it within the prompt's budget, exit non-zero — do NOT merge a broken state.
"""


def _parse_token_usage(stdout: str) -> tuple[int, int]:
    try:
        out = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return (0, 0)
    if not isinstance(out, dict):
        return (0, 0)
    usage = out.get("usage") or out.get("token_usage") or {}
    if not isinstance(usage, dict):
        return (0, 0)
    try:
        return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    except (TypeError, ValueError):
        return (0, 0)


def invoke_claude(
    track: str,
    prompt: str,
    budget_path: Path,
    attempts_path: Path,
    claude_runner: ClaudeRunner | None = None,
) -> int:
    """Run one-shot ``claude -p`` with the proposer prompt. Logs usage."""
    runner = claude_runner or _default_claude_runner
    t0 = time.time()
    rc, stdout, stderr = runner(
        ["claude", "-p", prompt, "--output-format=json"],
        prompt,
    )
    elapsed = time.time() - t0
    input_tokens, output_tokens = _parse_token_usage(stdout)
    # Log to the shared budget CSV (same schema reviewer uses, with a
    # marker candidate string so post-hoc analysis can split the two).
    budget_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not budget_path.exists()
    with budget_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(
                [
                    "timestamp",
                    "track",
                    "candidate",
                    "input_tokens",
                    "output_tokens",
                    "elapsed_sec",
                    "exit_code",
                ]
            )
        w.writerow(
            [
                datetime.now(UTC).isoformat(timespec="seconds"),
                track,
                "proposer",  # the "candidate" slot doubles as caller tag
                input_tokens,
                output_tokens,
                round(elapsed, 1),
                rc,
            ]
        )
    # Also log to the proposer's own attempt log. We don't know the actual
    # compound_name Claude chose (it's in the PR body); we'd need to parse
    # gh state. For now record the bare attempt; gate the next prompt on
    # exit_code only.
    _log_attempt(attempts_path, track, f"<unknown:{int(t0)}>", rc)
    if rc != 0:
        print(f"proposer: claude exited {rc}: {stderr[:500]}", file=sys.stderr)
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vgc_ai.proposer")
    p.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    p.add_argument("--budget", type=Path, default=DEFAULT_BUDGET_PATH)
    p.add_argument("--attempts", type=Path, default=DEFAULT_ATTEMPTS_PATH)
    p.add_argument("--cap", type=int, default=DEFAULT_DAILY_CAP)
    p.add_argument("--recent", type=int, default=DEFAULT_RECENT_ROWS)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Decide track + print prompt; never invoke claude.",
    )
    args = p.parse_args(argv)

    # Step 1 — pause if any of OUR open PRs are still in flight (zero Claude tokens).
    # Shared with the reviewer to avoid stacking proposer + reviewer PRs.
    open_prs = list_open_reviewer_prs()
    if open_prs:
        print(
            f"proposer: {len(open_prs)} loop PR(s) open ({open_prs!r}), skipping",
            file=sys.stderr,
        )
        return 0

    # Step 2 — daily cap check (zero Claude tokens). Shared with reviewer.
    today_calls = daily_claude_calls(args.budget)
    if today_calls >= args.cap:
        print(
            f"proposer: daily cap hit ({today_calls}/{args.cap}), skipping",
            file=sys.stderr,
        )
        return 0

    # Step 3 — shape bench context per track (zero Claude tokens).
    summaries = {
        "battle": summarize_track(
            args.csv_dir / "battle.csv",
            list(BATTLE_STRATEGIES),
            BATTLE_DEFAULT,
            args.recent,
        ),
        "championship": summarize_track(
            args.csv_dir / "championship.csv",
            list(CHAMPIONSHIP_STRATEGIES),
            CHAMPIONSHIP_DEFAULT,
            args.recent,
        ),
    }

    # Step 4 — pick the most-contested track.
    track = pick_most_contested_track(summaries)
    print(
        f"proposer: picking track={track} (defaults={ {k: v['default'] for k, v in summaries.items()} })",
        file=sys.stderr,
    )

    # Step 5 — read prior attempts.
    attempts = previous_attempts(args.attempts)

    # Step 6 — build the prompt.
    prompt = build_prompt(track, summaries, attempts)
    if args.dry_run:
        print(prompt)
        return 0

    # Step 7 — one-shot claude invocation.
    return invoke_claude(track, prompt, args.budget, args.attempts)


if __name__ == "__main__":
    sys.exit(main())
