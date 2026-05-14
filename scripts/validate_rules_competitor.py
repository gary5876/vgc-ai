"""Smoke-test that VgcAiDesignCompetitor runs the Rules Balance loop end-to-end.

Mirrors ``scripts/validate_competitor.py`` but for the design-competitor
surface: build ``RuleDesign`` (in-process — no ``ProxyDesignCompetitor`` /
sockets), register a ``DesignCompetitorManager`` wrapping our
``VgcAiDesignCompetitor``, run, assert a numeric score was produced.

Exit 0 = pass; non-zero = something in the rule-balance pipeline crashed.
"""

from __future__ import annotations

import sys
import time

from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.balance.rules.constraints import RuleConstraints
from vgc2.balance.rules.evaluator import evaluate_rules
from vgc2.competition import DesignCompetitorManager
from vgc2.competition.ecosystem import RuleDesign
from vgc2.competition.fixed_matches import FixedMatches

from vgc_ai.design_competitor import VgcAiDesignCompetitor

N_TEAM_PAIRS = 4  # small for a fast smoke
TIME_BUDGET_SEC = 60.0


def main() -> int:
    constraints = RuleConstraints()
    agent_pair = GreedyBattlePolicy(), GreedyBattlePolicy()
    fixed_matches = FixedMatches(agent_pair, N_TEAM_PAIRS)
    design = RuleDesign(fixed_matches, constraints, [evaluate_rules])

    competitor = VgcAiDesignCompetitor(name="vgc-ai")
    dcm = DesignCompetitorManager(competitor)
    design.register(dcm)

    t0 = time.perf_counter()
    design.run()
    elapsed = time.perf_counter() - t0

    print(
        f"validate_rules_competitor: name={competitor.name} "
        f"score={dcm.score:.4f} elapsed_sec={elapsed:.2f}"
    )

    if not isinstance(dcm.score, (int, float)):
        print(f"FAIL: score is not numeric ({type(dcm.score).__name__})", file=sys.stderr)
        return 1
    if elapsed > TIME_BUDGET_SEC:
        print(
            f"FAIL: exceeded {TIME_BUDGET_SEC}s budget (took {elapsed:.2f}s)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
