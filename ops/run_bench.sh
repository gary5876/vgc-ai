#!/usr/bin/env bash
# VM bench loop driver — version-controlled successor to the on-disk
# ~/run_bench.sh script that the original `bench` tmux session was using.
#
# Run from the bench checkout on the VM, e.g.:
#   cd /home/gjw0622/vgc-ai-bench && bash ops/run_bench.sh
#
# Each cycle:
#   1. git pull            — picks up auto-merged PRs from the reviewer loop
#   2. uv sync             — installs / refreshes deps
#   3. bench.run_continuous — per-policy continuous metric (legacy schema)
#   4. bench.run_strategy_tournament battle|championship|balance
#                          — strategy-level metric the reviewer reads
#
# Logs to ~/vgc-ai-logs/bench/round-<TS>.log per cycle.
#
# `set -e` is intentionally OFF: a single failed bench (e.g. a transient
# vgc2 internal error on one battle) must not kill the loop. Each step
# is `|| true` so the loop survives partial failures and keeps producing
# rows from the steps that did succeed.

set -uo pipefail

LOG_DIR="${HOME}/vgc-ai-logs/bench"
mkdir -p "${LOG_DIR}"

# Tunable per-cycle sample sizes. Kept small so a cycle finishes within
# minutes and accumulates rows the reviewer can pool across cycles.
BATTLE_N="${BATTLE_N:-200}"
CHAMPIONSHIP_N="${CHAMPIONSHIP_N:-30}"
CONTINUOUS_N="${CONTINUOUS_N:-100}"
SLEEP_SEC="${SLEEP_SEC:-60}"

while true; do
    ts=$(date -u +%Y%m%dT%H%M%SZ)
    log="${LOG_DIR}/round-${ts}.log"
    {
        echo "=== round ${ts} ==="
        git pull --quiet || true
        uv sync --quiet || true
        echo "--- run_continuous (n=${CONTINUOUS_N}) ---"
        uv run python -m bench.run_continuous --n "${CONTINUOUS_N}" || true
        echo "--- run_strategy_tournament battle (n=${BATTLE_N}) ---"
        uv run python -m bench.run_strategy_tournament battle --n "${BATTLE_N}" || true
        echo "--- run_strategy_tournament championship (n=${CHAMPIONSHIP_N}) ---"
        uv run python -m bench.run_strategy_tournament championship --n "${CHAMPIONSHIP_N}" || true
        echo "--- run_strategy_tournament balance ---"
        uv run python -m bench.run_strategy_tournament balance || true
        echo "=== done ${ts} ==="
    } 2>&1 | tee "${log}"
    sleep "${SLEEP_SEC}"
done
