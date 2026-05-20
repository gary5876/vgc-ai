"""PR auto-handler — adjudicates loop-opened PRs within the 5-min SLA.

Polls open PRs once per cycle (driver runs at ~1-min cadence on the VM).
Two PR classes are recognized by body markers:

- ``BENCH GATE`` — a reviewer-opened *default-swap* PR. Tight scope
  (only ``registry.py`` + ``battle.py``). Requires the bench gate to
  STILL fire when re-evaluated against current CSV rows.
- ``NEW COMPOUND`` — a proposer-opened *new-strategy* PR. Wider scope
  (any path under ``src/vgc_ai/policies/``, ``src/vgc_ai/eval/``,
  ``src/vgc_ai/strategies/``, or ``tests/``). No bench re-verification —
  the compound is brand new and hasn't been benched yet; CI alone gates.

Common gates for both classes:

1. **Age check** — wait if PR is younger than ``min_age_sec`` (PR may
   still be mid-creation).
2. **Scope check** — diff must stay within the class's allowed paths.
3. **CI gate** — ``ruff format --check``, ``ruff check``,
   ``mypy --strict``, ``pytest`` all green.

All gates are pure Python; zero Claude tokens are spent. Decision per PR
is ``merge`` / ``close`` / ``wait``. The driver lives in
``ops/run_pr_handler.sh``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vgc_ai.reviewer import (
    BENCH_GATE_MARKER,
    DEFAULT_CSV_DIR,
    evaluate_gate,
)

DEFAULT_MIN_AGE_SEC = 30
DEFAULT_MAX_AGE_SEC = 300  # 5-min SLA
DEFAULT_CMD_TIMEOUT_SEC = 600

# BENCH GATE (default-swap) PRs: tight whitelist by exact path.
BENCH_GATE_ALLOWED_PATHS: tuple[str, ...] = (
    "src/vgc_ai/strategies/registry.py",
    "src/vgc_ai/policies/battle.py",
)

# NEW COMPOUND PRs: any file whose path begins with one of these prefixes.
# Excludes ops/, bench/, scripts/, .github/, top-level — those are loop
# infrastructure or repo metadata that proposer-implementations should
# never touch.
NEW_COMPOUND_ALLOWED_PREFIXES: tuple[str, ...] = (
    "src/vgc_ai/policies/",
    "src/vgc_ai/eval/",
    "src/vgc_ai/strategies/",
    "tests/",
)

NEW_COMPOUND_MARKER = "NEW COMPOUND"

# Backwards-compatible alias used by older callers and tests.
DEFAULT_ALLOWED_PATHS = BENCH_GATE_ALLOWED_PATHS

CmdRunner = Callable[[list[str], Path, int], "tuple[int, str, str]"]
GhRunner = Callable[[list[str]], str]


def _default_gh_runner(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise RuntimeError(f"gh failed (rc={result.returncode}): {result.stderr.strip()}")
    return result.stdout


def _default_cmd_runner(cmd: list[str], cwd: Path, timeout: int) -> tuple[int, str, str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout, result.stderr


def parse_bench_gate(body: str) -> dict[str, str] | None:
    r"""Extract the ``BENCH GATE`` key-value block from a PR body.

    Block shape (the reviewer's prompt template demands this exact form):

        BENCH GATE
        track=battle
        candidate=foo
        default=bar
        pooled_n=100
        pooled_wins=70
        ci95_low=0.60
        ci95_high=0.78

    Tolerant of an enclosing markdown code fence: lines starting with
    ``\`\`\``` are skipped. Stops at the first blank or non-``k=v`` line
    after parsing began.
    """
    if BENCH_GATE_MARKER not in body:
        return None
    tail = body.split(BENCH_GATE_MARKER, 1)[1]
    parsed: dict[str, str] = {}
    for raw in tail.splitlines():
        line = raw.strip()
        if not line:
            if parsed:
                break
            continue
        if line.startswith("```"):
            if parsed:
                break
            continue
        if "=" not in line:
            if parsed:
                break
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        if k and v:
            parsed[k] = v
    return parsed or None


def parse_new_compound(body: str) -> dict[str, str] | None:
    """Extract the ``NEW COMPOUND`` key-value block from a PR body.

    Block shape (the proposer's prompt template demands this exact form):

        NEW COMPOUND
        track=battle
        compound_name=heuristic_det_damage_term
        rationale=Adds a base_power-weighted damage term to evaluate()

    Tolerance rules match ``parse_bench_gate``: optional markdown fence,
    stops at first blank or non-``k=v`` line after parsing began.
    """
    if NEW_COMPOUND_MARKER not in body:
        return None
    tail = body.split(NEW_COMPOUND_MARKER, 1)[1]
    parsed: dict[str, str] = {}
    for raw in tail.splitlines():
        line = raw.strip()
        if not line:
            if parsed:
                break
            continue
        if line.startswith("```"):
            if parsed:
                break
            continue
        if "=" not in line:
            if parsed:
                break
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip()
        if k and v:
            parsed[k] = v
    return parsed or None


def scope_check(
    changed_paths: list[str],
    allowed: tuple[str, ...] = BENCH_GATE_ALLOWED_PATHS,
) -> tuple[bool, str]:
    """All ``changed_paths`` must be in ``allowed`` (exact match).

    Used for ``BENCH GATE`` PRs — the diff should be a tiny, predictable
    constant swap and possibly a battle-policy alias update.
    """
    out_of_scope = [p for p in changed_paths if p not in allowed]
    if out_of_scope:
        return False, f"out-of-scope paths touched: {out_of_scope}"
    return True, ""


def scope_check_prefix(
    changed_paths: list[str],
    allowed_prefixes: tuple[str, ...] = NEW_COMPOUND_ALLOWED_PREFIXES,
) -> tuple[bool, str]:
    """Every path must start with one of ``allowed_prefixes``.

    Used for ``NEW COMPOUND`` PRs — the proposer may add new policy
    files, new tests, new eval helpers, but cannot touch ops/, bench/,
    scripts/, .github/, or top-level config.
    """
    out_of_scope = [
        p for p in changed_paths if not any(p.startswith(pfx) for pfx in allowed_prefixes)
    ]
    if out_of_scope:
        return False, f"out-of-scope paths touched: {out_of_scope}"
    return True, ""


def bench_gate_still_fires(
    parsed: dict[str, str],
    csv_dir: Path,
    n_recent: int = 50,
) -> tuple[bool, str]:
    """Re-evaluate the reviewer's gate against the latest CSV data."""
    track = parsed.get("track", "")
    candidate = parsed.get("candidate", "")
    default = parsed.get("default", "")
    if not candidate or not default:
        return False, "BENCH GATE block missing candidate or default"
    if track == "battle":
        csv_path = csv_dir / "battle.csv"
    elif track == "championship":
        csv_path = csv_dir / "championship.csv"
    else:
        return False, f"unknown track: {track!r}"
    decisions = evaluate_gate(csv_path, [candidate], default, track, n_recent)
    if not decisions:
        return False, "no recent rows for this candidate/default pair"
    d = decisions[0]
    if not d["gate_fires"]:
        return False, (
            f"gate no longer fires: ci95_low={d['ci95_low']} (was "
            f"{parsed.get('ci95_low', '?')}); n={d['pooled_n']}"
        )
    return True, ""


def parse_pr_age_seconds(created_at: str) -> float:
    """``created_at`` from gh JSON is an ISO-8601 timestamp; returns seconds since."""
    # gh emits a trailing Z which fromisoformat() doesn't parse pre-Python 3.11;
    # we're on 3.12 but the substitution is cheap insurance.
    dt = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    return (datetime.now(UTC) - dt).total_seconds()


def _has_handler_marker(body: str) -> bool:
    return BENCH_GATE_MARKER in body or NEW_COMPOUND_MARKER in body


def list_matching_prs(gh_runner: GhRunner | None = None) -> list[dict[str, Any]]:
    """Open PRs whose body contains either ``BENCH GATE`` or ``NEW COMPOUND``."""
    runner = gh_runner or _default_gh_runner
    try:
        raw = runner(
            [
                "gh",
                "pr",
                "list",
                "--state",
                "open",
                "--json",
                "number,body,files,createdAt,headRefName",
            ]
        )
    except (FileNotFoundError, subprocess.SubprocessError, RuntimeError, OSError) as e:
        print(f"auto-handler: gh list failed: {e}", file=sys.stderr)
        return []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    return [item for item in data if _has_handler_marker(item.get("body") or "")]


def run_ci_gate(
    workdir: Path,
    cmd_runner: CmdRunner | None = None,
    timeout: int = DEFAULT_CMD_TIMEOUT_SEC,
) -> tuple[bool, str]:
    """Run ruff/mypy/pytest sequentially; stop on first failure."""
    runner = cmd_runner or _default_cmd_runner
    commands = [
        ["uv", "run", "ruff", "format", "--check", "src", "tests"],
        ["uv", "run", "ruff", "check", "src", "tests"],
        ["uv", "run", "mypy", "--strict", "src"],
        ["uv", "run", "pytest"],
    ]
    for cmd in commands:
        rc, _stdout, stderr = runner(cmd, workdir, timeout)
        if rc != 0:
            return False, f"{' '.join(cmd[2:])} failed:\n{stderr[-800:]}"
    return True, ""


def evaluate_pr(
    pr: dict[str, Any],
    csv_dir: Path,
    workdir: Path,
    *,
    min_age_sec: int = DEFAULT_MIN_AGE_SEC,
    max_age_sec: int = DEFAULT_MAX_AGE_SEC,
    allowed_paths: tuple[str, ...] = BENCH_GATE_ALLOWED_PATHS,
    allowed_prefixes: tuple[str, ...] = NEW_COMPOUND_ALLOWED_PREFIXES,
    cmd_runner: CmdRunner | None = None,
) -> tuple[str, str]:
    """Decide what to do with a single PR — handles both PR classes.

    Returns ``(action, reason)`` where ``action`` is one of:

    - ``"merge"`` — all gates green, ready to merge.
    - ``"close"`` — at least one gate failed terminally (or SLA expired).
    - ``"wait"`` — too young, or CI flaked and SLA hasn't expired yet.

    The marker class determines which scope check applies and whether the
    bench-gate re-verification runs:

    - ``BENCH GATE``: ``scope_check`` (exact-path), bench re-verify required.
    - ``NEW COMPOUND``: ``scope_check_prefix`` (path-prefix), no bench
      re-verification (the compound hasn't been benched yet).
    """
    body = pr.get("body") or ""
    bench_block = parse_bench_gate(body)
    new_block = parse_new_compound(body)
    if bench_block is None and new_block is None:
        return "close", "PR body missing BENCH GATE or NEW COMPOUND block"

    created_at = pr.get("createdAt") or ""
    if not created_at:
        return "close", "PR has no createdAt timestamp"
    age = parse_pr_age_seconds(created_at)

    if age < min_age_sec:
        return "wait", f"PR is only {age:.0f}s old (min_age={min_age_sec})"

    changed = [f.get("path", "") for f in pr.get("files", [])]
    if bench_block is not None:
        ok, reason = scope_check(changed, allowed_paths)
    else:
        ok, reason = scope_check_prefix(changed, allowed_prefixes)
    if not ok:
        return "close", reason

    ok, reason = run_ci_gate(workdir, cmd_runner=cmd_runner)
    if not ok:
        if age > max_age_sec:
            return "close", f"CI gate failed after {age:.0f}s (SLA={max_age_sec}): {reason}"
        return "wait", f"CI failing, age {age:.0f}s, retry next cycle: {reason}"

    # Bench-gate re-verification is BENCH-GATE-only. NEW COMPOUND PRs add
    # untested entries — they have no bench history yet, so re-verify is
    # not applicable; the bench loop will measure them after merge.
    if bench_block is not None:
        ok, reason = bench_gate_still_fires(bench_block, csv_dir)
        if not ok:
            return "close", f"bench gate re-check failed: {reason}"

    return "merge", "all gates passed"


def merge_pr(pr_number: int, gh_runner: GhRunner | None = None) -> tuple[bool, str]:
    """Squash-merge and delete the branch."""
    runner = gh_runner or _default_gh_runner
    try:
        runner(["gh", "pr", "merge", str(pr_number), "--squash", "--delete-branch"])
        return True, ""
    except (RuntimeError, subprocess.SubprocessError, OSError) as e:
        return False, str(e)


def close_pr(pr_number: int, reason: str, gh_runner: GhRunner | None = None) -> tuple[bool, str]:
    """Close with an explanatory comment."""
    runner = gh_runner or _default_gh_runner
    comment = f"auto-handler closed: {reason}"
    try:
        runner(["gh", "pr", "close", str(pr_number), "--comment", comment])
        return True, ""
    except (RuntimeError, subprocess.SubprocessError, OSError) as e:
        return False, str(e)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vgc_ai.pr_auto_handler")
    p.add_argument("--workdir", type=Path, default=Path.cwd())
    p.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    p.add_argument("--min-age", type=int, default=DEFAULT_MIN_AGE_SEC)
    p.add_argument("--max-age", type=int, default=DEFAULT_MAX_AGE_SEC)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Decide per PR but do not call gh merge/close.",
    )
    args = p.parse_args(argv)

    prs = list_matching_prs()
    if not prs:
        print("auto-handler: no matching PRs", file=sys.stderr)
        return 0

    for pr in prs:
        n = int(pr["number"])
        action, reason = evaluate_pr(
            pr,
            args.csv_dir,
            args.workdir,
            min_age_sec=args.min_age,
            max_age_sec=args.max_age,
        )
        print(f"auto-handler: PR #{n} -> {action} ({reason})", file=sys.stderr)
        if args.dry_run:
            continue
        if action == "merge":
            ok, err = merge_pr(n)
            if not ok:
                print(f"auto-handler: merge failed for #{n}: {err}", file=sys.stderr)
        elif action == "close":
            ok, err = close_pr(n, reason)
            if not ok:
                print(f"auto-handler: close failed for #{n}: {err}", file=sys.stderr)
        # "wait" — do nothing this cycle; next poll will retry.
    return 0


if __name__ == "__main__":
    sys.exit(main())
