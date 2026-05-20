#!/usr/bin/env bash
# Proposer loop driver — wakes every SLEEP_SEC (default 1h) and runs one
# autonomous-ideation decision.
#
# Run from the bench checkout on the VM, e.g.:
#   cd /home/gjw0622/vgc-ai-bench && bash ops/run_proposer.sh
#
# Each cycle:
#   1. git pull       — sync to latest main (incl. auto-merged PRs)
#   2. uv sync        — keep deps current
#   3. python -m vgc_ai.proposer
#                     — single decision: pause / skip / fire claude
#
# Token discipline per [[feedback_loop_token_budget]]: proposer pauses
# while any reviewer/proposer PR is in flight, refuses if shared daily
# cap hit, only spawns `claude -p` when there's a track that could use
# a new compound. Routine wakes spend ZERO Claude tokens.
#
# Default cadence is HOURLY (3600s) — lower than the reviewer's 10 min
# because proposer fires are much more expensive (10–20K tokens vs 5K)
# and produce code that needs human-or-CI review.
#
# Logs to ~/vgc-ai-logs/proposer/proposer-<TS>.log per cycle.

set -uo pipefail

LOG_DIR="${HOME}/vgc-ai-logs/proposer"
mkdir -p "${LOG_DIR}"
SLEEP_SEC="${SLEEP_SEC:-3600}"

while true; do
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    log="${LOG_DIR}/proposer-${ts}.log"
    {
        echo "=== proposer wake ${ts} ==="
        git pull --quiet || true
        uv sync --quiet || true
        uv run python -m vgc_ai.proposer || true
        echo "=== done ${ts} ==="
    } 2>&1 | tee "${log}"
    sleep "${SLEEP_SEC}"
done
