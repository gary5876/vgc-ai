"""Live smoke test — runs random-vs-random doubles battles against a local
Pokemon Showdown server.

Skipped unless ``VGC_AI_LIVE=1`` is set in the environment. Assumes a server
is running at ``ws://localhost:8000`` (start with
``node pokemon-showdown start --no-security`` in the smogon/pokemon-showdown
checkout).
"""

from __future__ import annotations

import os
import uuid

import pytest

from vgc_ai.agents.random_agent import make_random_doubles_player

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("VGC_AI_LIVE") != "1",
        reason="set VGC_AI_LIVE=1 with a local Showdown server running",
    ),
]


async def test_random_doubles_three_battles() -> None:
    suffix = uuid.uuid4().hex[:6]
    p1 = make_random_doubles_player(f"vgcai-itp1-{suffix}")
    p2 = make_random_doubles_player(f"vgcai-itp2-{suffix}")

    n = 3
    await p1.battle_against(p2, n_battles=n)

    assert p1.n_finished_battles == n
    assert p2.n_finished_battles == n
    assert p1.n_won_battles + p1.n_lost_battles + p1.n_tied_battles == n
