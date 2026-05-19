"""Smoke-test that VgcAiCompetitor runs a doubles Match end-to-end.

Used as a competition-submission validation gate. Runs two passes:

1. **Default rules** — ten doubles battles with the standard ``BattleRuleParam``.
2. **Random rules** — ten doubles battles with a ``gen_rule_set``-generated
   ``BattleRuleParam`` (3 attribute mutations + 5 type-chart cell mutations).
   The 2026 Battle Track generates the ruleset dynamically per tournament, so
   the submission must run cleanly under arbitrary perturbations of damage
   multipliers / STAB / status thresholds / type chart cells.

Asserts both passes complete with a winner per battle and stay within the
time budget. Exits 0 on success.
"""

from __future__ import annotations

import sys
import time

from numpy.random import default_rng
from vgc2.battle_engine.constants import BattleRuleParam
from vgc2.competition import CompetitorManager
from vgc2.competition.match import Match
from vgc2.util.generator import gen_rule_set, gen_team

from vgc_ai.competitor import VgcAiCompetitor

N_BATTLES = 10
N_ACTIVE = 2
TIME_BUDGET_SEC = 60.0
RANDOM_RULES_SEED = 42


def _run_pass(label: str, params: BattleRuleParam) -> int:
    a = VgcAiCompetitor(name="vgc-ai-A")
    b = VgcAiCompetitor(name="vgc-ai-B")
    cm = (CompetitorManager(a), CompetitorManager(b))
    # Match._run_random executes two battles per loop iteration (one per side
    # ordering) so n_battles=5 is the smallest setting that yields >=10 actual
    # battles; a tiebreaker may push the total slightly higher.
    match = Match(
        cm,
        n_active=N_ACTIVE,
        n_battles=N_BATTLES // 2,
        gen=gen_team,
        params=params,
    )

    t0 = time.perf_counter()
    match.run()
    elapsed = time.perf_counter() - t0

    wins_a, wins_b = match.wins
    total = wins_a + wins_b
    print(
        f"validate_competitor[{label}]: wins_a={wins_a} wins_b={wins_b} "
        f"total_battles={total} elapsed_sec={elapsed:.2f}"
    )

    if total < N_BATTLES:
        print(
            f"FAIL[{label}]: ran fewer battles ({total}) than requested ({N_BATTLES})",
            file=sys.stderr,
        )
        return 1
    if elapsed > TIME_BUDGET_SEC:
        print(
            f"FAIL[{label}]: exceeded {TIME_BUDGET_SEC}s budget (took {elapsed:.2f}s)",
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    rc = _run_pass("default-rules", BattleRuleParam())
    if rc != 0:
        return rc
    random_params = gen_rule_set(rng=default_rng(RANDOM_RULES_SEED))
    return _run_pass("random-rules", random_params)


if __name__ == "__main__":
    sys.exit(main())
