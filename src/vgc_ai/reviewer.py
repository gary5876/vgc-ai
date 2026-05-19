"""Reviewer loop wake handler — one decision per invocation.

Reads the latest tournament rows from
``bench/strategies/{battle,championship,balance}.csv``, pools by
(candidate, default) pair, and decides whether to invoke Claude to open a
PR promoting a new default.

Token-budget discipline per [[feedback_loop_token_budget]]:

- **Step 1**: pause if any of our open PRs are still in flight. Zero Claude
  tokens. Detection is body-content (PRs whose body contains
  ``BENCH GATE``), not label, so no label setup needed.
- **Step 2**: refuse if today's Claude invocations hit the daily cap.
  Zero tokens.
- **Step 3**: pure-Python gate evaluation on the CSV rows. Zero tokens.
- **Step 4**: only when a candidate's pooled ``ci95_low > 0.5`` does a
  one-shot ``claude -p`` fire — no persistent session, ~5K input tokens
  per call.

Token usage logged per invocation to ``ops/claude_budget.csv``. The loop
shell (``ops/run_reviewer.sh``) wakes this every 10 min; this module is
the per-wake logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from vgc_ai.strategies import (
    BATTLE_DEFAULT,
    BATTLE_STRATEGIES,
    CHAMPIONSHIP_DEFAULT,
    CHAMPIONSHIP_STRATEGIES,
)

DEFAULT_CSV_DIR = Path("bench/strategies")
DEFAULT_BUDGET_PATH = Path("ops/claude_budget.csv")
DEFAULT_DAILY_CAP = 12
DEFAULT_RECENT_ROWS = 50
BENCH_GATE_MARKER = "BENCH GATE"
CLAUDE_TIMEOUT_SEC = 900
GH_TIMEOUT_SEC = 30

GhRunner = Callable[[list[str]], str]
ClaudeRunner = Callable[[list[str], str], "tuple[int, str, str]"]


def wilson_ci_95(wins: int, n: int) -> tuple[float, float]:
    """Wilson 95% confidence interval on a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = wins / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    spread = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))


def _default_gh_runner(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=GH_TIMEOUT_SEC)
    if result.returncode != 0:
        raise RuntimeError(f"gh failed (rc={result.returncode}): {result.stderr.strip()}")
    return result.stdout


def list_open_reviewer_prs(gh_runner: GhRunner | None = None) -> list[int]:
    """Return PR numbers of open PRs whose body contains ``BENCH_GATE_MARKER``.

    Defaults to invoking ``gh pr list --state open --json number,body``. If
    ``gh`` is unavailable or returns malformed JSON, returns ``[0]`` so the
    reviewer pauses conservatively rather than firing into an unknown
    repository state.
    """
    runner = gh_runner or _default_gh_runner
    try:
        raw = runner(["gh", "pr", "list", "--state", "open", "--json", "number,body"])
    except (FileNotFoundError, subprocess.SubprocessError, RuntimeError, OSError):
        return [0]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [0]
    return [int(item["number"]) for item in data if BENCH_GATE_MARKER in (item.get("body") or "")]


def daily_claude_calls(budget_path: Path, today: str | None = None) -> int:
    """Count Claude invocations in ``budget_path`` whose timestamp begins with ``today`` (UTC)."""
    if not budget_path.exists():
        return 0
    today_str = today or date.today().isoformat()
    n = 0
    with budget_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("timestamp", "").startswith(today_str):
                n += 1
    return n


def load_recent_rows(csv_path: Path, n: int = DEFAULT_RECENT_ROWS) -> list[dict[str, str]]:
    """Read the last ``n`` rows from a tournament CSV; empty list if file missing."""
    if not csv_path.exists():
        return []
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[-n:]


def _pool_pair(rows: list[dict[str, str]]) -> tuple[int, int]:
    """Sum (wins_a, decided_n) across rows for a single ordered pair."""
    wins = sum(int(r["wins_a"]) for r in rows)
    n = sum(int(r["wins_a"]) + int(r["wins_b"]) for r in rows)
    return wins, n


def evaluate_gate(
    csv_path: Path,
    candidates: list[str],
    default: str,
    track: str,
    n_recent: int = DEFAULT_RECENT_ROWS,
) -> list[dict[str, Any]]:
    """Pool head-to-head between each non-default candidate and the default.

    Returns one dict per candidate that has at least one row vs the default
    in the last ``n_recent`` CSV rows. Each dict carries the pooled stats
    plus ``gate_fires = (ci95_low > 0.5)``.
    """
    rows = load_recent_rows(csv_path, n_recent)
    if not rows:
        return []
    out: list[dict[str, Any]] = []
    for cand in candidates:
        if cand == default:
            continue
        pair_rows = [
            r for r in rows if r.get("strategy_a") == cand and r.get("strategy_b") == default
        ]
        if not pair_rows:
            continue
        wins, n = _pool_pair(pair_rows)
        if n == 0:
            continue
        win_rate = wins / n
        ci_lo, ci_hi = wilson_ci_95(wins, n)
        out.append(
            {
                "track": track,
                "candidate": cand,
                "default": default,
                "pooled_n": n,
                "pooled_wins": wins,
                "win_rate": round(win_rate, 4),
                "ci95_low": round(ci_lo, 4),
                "ci95_high": round(ci_hi, 4),
                "gate_fires": ci_lo > 0.5,
            }
        )
    return out


def evaluate_all_tracks(csv_dir: Path, n_recent: int = DEFAULT_RECENT_ROWS) -> list[dict[str, Any]]:
    """Run gate evaluation for battle + championship. Balance has no gate yet."""
    out: list[dict[str, Any]] = []
    out.extend(
        evaluate_gate(
            csv_dir / "battle.csv",
            list(BATTLE_STRATEGIES),
            BATTLE_DEFAULT,
            "battle",
            n_recent,
        )
    )
    out.extend(
        evaluate_gate(
            csv_dir / "championship.csv",
            list(CHAMPIONSHIP_STRATEGIES),
            CHAMPIONSHIP_DEFAULT,
            "championship",
            n_recent,
        )
    )
    return out


def _log_budget(
    budget_path: Path,
    track: str,
    candidate: str,
    input_tokens: int,
    output_tokens: int,
    elapsed_sec: float,
    exit_code: int,
) -> None:
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
                candidate,
                input_tokens,
                output_tokens,
                round(elapsed_sec, 1),
                exit_code,
            ]
        )


def build_prompt(decision: dict[str, Any]) -> str:
    """Construct the minimal one-shot prompt for the candidate-promotion task.

    Kept terse: Claude already loads CLAUDE.md from cwd, so the prompt skips
    project conventions and only states what *this* invocation must do.
    """
    today = datetime.now(UTC).strftime("%Y%m%d")
    track = decision["track"]
    cand = decision["candidate"]
    default = decision["default"]
    return f"""You are extending the vgc-ai project to promote a new {track}-track strategy default.

The bench evidence shows {cand} has dethroned the current default ({default}):
- Pooled head-to-head: {decision["pooled_wins"]}/{decision["pooled_n"]} ({decision["win_rate"]:.1%})
- 95% CI: [{decision["ci95_low"]}, {decision["ci95_high"]}]
- Gate cleared (ci95_low > 0.5): yes

Task:
1. Edit `src/vgc_ai/strategies/registry.py`: change `{track.upper()}_DEFAULT = "..."` to `{track.upper()}_DEFAULT = "{cand}"`.
2. For the battle track only, also update `src/vgc_ai/policies/battle.py` so `VgcAiBattlePolicy` aliases the policy class produced by `BATTLE_STRATEGIES["{cand}"].battle_policy`.
3. Run `uv run ruff format src tests && uv run pytest`. If anything fails, fix only the smallest necessary thing — do not refactor.
4. Create branch `auto/promote-{cand}-{today}`, commit per CLAUDE.md format (subject: `feat(strategies): promote {cand} as {track} default`), push, open PR via `gh pr create`.

The PR body MUST include this exact block verbatim (the auto-handler parses it):
```
BENCH GATE
track={track}
candidate={cand}
default={default}
pooled_n={decision["pooled_n"]}
pooled_wins={decision["pooled_wins"]}
ci95_low={decision["ci95_low"]}
ci95_high={decision["ci95_high"]}
```

Exit 0 on success. Do not edit unrelated files. Do not add new tests. Do not amend existing commits.
"""


def _default_claude_runner(cmd: list[str], _prompt: str) -> tuple[int, str, str]:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=CLAUDE_TIMEOUT_SEC)
    return result.returncode, result.stdout, result.stderr


def _parse_token_usage(stdout: str) -> tuple[int, int]:
    """Extract (input_tokens, output_tokens) from the Claude CLI's JSON output.

    The exact shape has varied across CLI versions; we try the documented
    ``usage`` key and the legacy ``token_usage`` key, and fall back to ``(0, 0)``
    so an unknown format doesn't break the budget log.
    """
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
    decision: dict[str, Any],
    budget_path: Path,
    claude_runner: ClaudeRunner | None = None,
) -> int:
    """Run one-shot ``claude -p`` with the promotion prompt. Logs token usage."""
    runner = claude_runner or _default_claude_runner
    prompt = build_prompt(decision)
    t0 = time.time()
    rc, stdout, stderr = runner(
        ["claude", "-p", prompt, "--output-format=json"],
        prompt,
    )
    elapsed = time.time() - t0
    input_tokens, output_tokens = _parse_token_usage(stdout)
    _log_budget(
        budget_path,
        decision["track"],
        decision["candidate"],
        input_tokens,
        output_tokens,
        elapsed,
        rc,
    )
    if rc != 0:
        print(f"reviewer: claude exited {rc}: {stderr[:500]}", file=sys.stderr)
    return rc


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vgc_ai.reviewer")
    p.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    p.add_argument("--budget", type=Path, default=DEFAULT_BUDGET_PATH)
    p.add_argument("--cap", type=int, default=DEFAULT_DAILY_CAP)
    p.add_argument("--recent", type=int, default=DEFAULT_RECENT_ROWS)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Evaluate gate and print decision; never invoke claude.",
    )
    args = p.parse_args(argv)

    # Step 1 — pause if any reviewer PR is still in flight (zero Claude tokens).
    open_prs = list_open_reviewer_prs()
    if open_prs:
        print(
            f"reviewer: {len(open_prs)} reviewer PR(s) open ({open_prs!r}), skipping",
            file=sys.stderr,
        )
        return 0

    # Step 2 — daily cap check (zero Claude tokens).
    today_calls = daily_claude_calls(args.budget)
    if today_calls >= args.cap:
        print(
            f"reviewer: daily cap hit ({today_calls}/{args.cap}), skipping",
            file=sys.stderr,
        )
        return 0

    # Step 3 — pure-Python gate evaluation (zero Claude tokens).
    decisions = evaluate_all_tracks(args.csv_dir, args.recent)
    fired = [d for d in decisions if d["gate_fires"]]
    if not fired:
        print(
            f"reviewer: evaluated {len(decisions)} candidate pair(s), no gate fires",
            file=sys.stderr,
        )
        return 0

    # Pick the strongest (highest ci95_low) firing candidate.
    winner = max(fired, key=lambda d: float(d["ci95_low"]))
    print(
        f"reviewer: gate fires for {winner['candidate']} "
        f"(track={winner['track']}, ci95_low={winner['ci95_low']}, "
        f"n={winner['pooled_n']})",
        file=sys.stderr,
    )

    if args.dry_run:
        print(json.dumps(winner, indent=2))
        return 0

    # Step 4 — one-shot claude invocation. The ONLY token-spending path.
    return invoke_claude(winner, args.budget)


if __name__ == "__main__":
    sys.exit(main())
