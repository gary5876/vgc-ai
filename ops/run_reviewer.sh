#!/usr/bin/env bash
# Reviewer loop driver — wakes every SLEEP_SEC and runs one decision.
#
# Run from the bench checkout on the VM, e.g.:
#   cd /home/gjw0622/vgc-ai-bench && bash ops/run_reviewer.sh
#
# Each cycle:
#   1. git pull              — sync to latest main (incl. auto-merged PRs)
#   2. uv sync               — keep deps current
#   3. python -m vgc_ai.reviewer
#                            — single decision: pause / skip / fire claude
#
# Token discipline per [[feedback_loop_token_budget]]: reviewer pauses while
# any of its open PRs are in flight, refuses if daily cap hit, only spawns
# `claude -p` when a candidate has crossed the ci95 gate. Routine wakes
# spend ZERO Claude tokens.
#
# Logs to ~/vgc-ai-logs/reviewer/reviewer-<TS>.log per cycle.

set -uo pipefail

LOG_DIR="${HOME}/vgc-ai-logs/reviewer"
mkdir -p "${LOG_DIR}"
SLEEP_SEC="${SLEEP_SEC:-600}"  # 10 min default

while true; do
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    log="${LOG_DIR}/reviewer-${ts}.log"
    {
        echo "=== reviewer wake ${ts} ==="
        git pull --quiet || true
        uv sync --quiet || true
        uv run python -m vgc_ai.reviewer || true
        echo "=== done ${ts} ==="
    } 2>&1 | tee "${log}"
    sleep "${SLEEP_SEC}"
done
