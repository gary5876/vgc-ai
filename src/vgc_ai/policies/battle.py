"""Battle policy.

Aliases ``HeuristicDetBattlePolicy`` — the first policy to clear the bench
gate against ``GreedyBattlePolicy`` (n=2000: win_rate=0.5405,
ci95_low=0.5186, +4.05% true edge at 95% confidence; PR #9). Per-turn cost
~25 ms in doubles, well under the sub-second budget the contest harness
implies. ``GreedyBattlePolicy`` remains the registered ``greedy`` baseline
for bench comparison.

``TreeSearchBattlePolicy`` is stronger (100% vs random in our test) but
measures ~170s per battle / ~11s per turn in doubles, far above typical
competition per-turn limits — kept available via
``vgc2.agent.battle.TreeSearchBattlePolicy`` for local benchmarking only.

Next target: replace this alias with a policy that clears the gate against
``HeuristicDetBattlePolicy`` itself. Candidates queued in TASKS.md:
``policy-heuristic-eval-det-2ply`` (deeper lookahead) and
``policy-tabular-mc-canonicalize-action-idx`` (unstick the MC family).
"""

from vgc_ai.policies.heuristic_det import HeuristicDetBattlePolicy as VgcAiBattlePolicy

__all__ = ["VgcAiBattlePolicy"]
