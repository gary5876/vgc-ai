"""Registry of named battle policies for CLI and bench consumers.

Kept separate from ``vgc_ai.cli`` so ``bench/`` and other callers can import
``POLICIES`` without pulling in the argparse-flavored CLI module.
"""

from __future__ import annotations

from vgc2.agent.battle import GreedyBattlePolicy, RandomBattlePolicy, TreeSearchBattlePolicy

from vgc_ai.eval.duel import PolicyFactory

POLICIES: dict[str, PolicyFactory] = {
    "random": RandomBattlePolicy,
    "greedy": GreedyBattlePolicy,
    "tree": TreeSearchBattlePolicy,
}

__all__ = ["POLICIES"]
