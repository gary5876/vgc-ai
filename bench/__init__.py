"""Continuous benchmarking harness for vgc-ai battle policies.

Wraps :func:`vgc_ai.eval.duel.duel` to produce structured JSON results
(`run_once`) and an append-only leaderboard CSV (`run_continuous`).

Designed to run forever in a shell loop on the VM:

    while true; do uv run python -m bench.run_continuous; sleep 60; done

Each invocation is a fresh Python process so newly-added policies in
``vgc_ai.cli.POLICIES`` are picked up at the next round without a restart.
"""
