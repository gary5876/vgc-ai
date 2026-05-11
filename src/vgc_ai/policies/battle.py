"""Battle policy.

Currently aliases `GreedyBattlePolicy` — submission-viable: ~0.04s per battle in
doubles, ~90% win rate vs `RandomBattlePolicy`. `TreeSearchBattlePolicy` is
stronger (100% vs random in our test) but measures ~170s per battle / ~11s per
turn in doubles, well over typical competition per-turn limits. Tree is kept
available via `vgc2.agent.battle.TreeSearchBattlePolicy` for local benchmarking
only.

Target: beat `GreedyBattlePolicy` consistently while staying under a sub-second
per-turn budget. Probable approach: tabular Monte Carlo (cf. AurelianTactics
2024 3rd place) or a small search with hard time limit + heuristic eval.
"""

from vgc2.agent.battle import GreedyBattlePolicy as VgcAiBattlePolicy

__all__ = ["VgcAiBattlePolicy"]
