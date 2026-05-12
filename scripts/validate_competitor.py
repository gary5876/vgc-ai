"""Smoke-test that VgcAiCompetitor runs a doubles Match end-to-end.

Used as a competition-submission validation gate: builds two competitor
instances, runs ten doubles battles via ``vgc2.competition.match.Match``,
and asserts the match completes with a winner per battle. Exits 0 on success.
"""

from __future__ import annotations

import sys
import time

from vgc2.competition import CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_team

from vgc_ai.competitor import VgcAiCompetitor

N_BATTLES = 10
N_ACTIVE = 2
TIME_BUDGET_SEC = 60.0


def main() -> int:
    a = VgcAiCompetitor(name="vgc-ai-A")
    b = VgcAiCompetitor(name="vgc-ai-B")
    cm = (CompetitorManager(a), CompetitorManager(b))
    # Match._run_random executes two battles per loop iteration (one per side
    # ordering) so n_battles=5 is the smallest setting that yields >=10 actual
    # battles; a tiebreaker may push the total slightly higher.
    match = Match(cm, n_active=N_ACTIVE, n_battles=N_BATTLES // 2, gen=gen_team)

    t0 = time.perf_counter()
    match.run()
    elapsed = time.perf_counter() - t0

    wins_a, wins_b = match.wins
    total = wins_a + wins_b
    print(
        f"validate_competitor: wins_a={wins_a} wins_b={wins_b} "
        f"total_battles={total} elapsed_sec={elapsed:.2f}"
    )

    if total < N_BATTLES:
        print(
            f"FAIL: ran fewer battles ({total}) than requested ({N_BATTLES})",
            file=sys.stderr,
        )
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
