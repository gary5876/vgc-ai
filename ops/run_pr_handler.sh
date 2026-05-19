#!/usr/bin/env bash
# PR auto-handler loop driver. Run from the bench checkout on the VM:
#   cd /home/gjw0622/vgc-ai-bench && bash ops/run_pr_handler.sh
#
# Each cycle (SLEEP_SEC seconds, default 60):
#   1. git pull           — sync to latest main
#   2. For every open PR with a BENCH GATE block in its body, the
#      python module decides merge / close / wait.
#
# Zero Claude tokens — the handler is pure-Python validation:
# ruff/mypy/pytest plus the bench-gate re-verification.
#
# The handler runs in its own dedicated checkout so the bench loop
# (running `git pull` on the same dir) doesn't race git operations.
# Recommended VM layout:
#   /home/gjw0622/vgc-ai-bench/    (bench loop pulls here)
#   /home/gjw0622/vgc-ai-handler/  (this script pulls here)
#
# Logs to ~/vgc-ai-logs/pr-handler/handler-<TS>.log per cycle.

set -uo pipefail

LOG_DIR="${HOME}/vgc-ai-logs/pr-handler"
mkdir -p "${LOG_DIR}"
SLEEP_SEC="${SLEEP_SEC:-60}"

while true; do
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    log="${LOG_DIR}/handler-${ts}.log"
    {
        echo "=== handler wake ${ts} ==="
        git pull --quiet || true
        uv sync --quiet || true
        uv run python -m vgc_ai.pr_auto_handler || true
        echo "=== done ${ts} ==="
    } 2>&1 | tee "${log}"
    sleep "${SLEEP_SEC}"
done
