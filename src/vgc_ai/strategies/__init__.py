"""Compound-strategy registries per competition track.

A "strategy" is a named compound of vgc2 policies. The tournament driver and
reviewer loop both read these registries: the tournament benches every
compound, the reviewer decides which compound becomes the
``VgcAi*Competitor`` default based on the bench evidence.

Three registries, one per track:

- ``BATTLE_STRATEGIES`` — battle-policy candidates (single policy each).
- ``CHAMPIONSHIP_STRATEGIES`` — team-build + selection pairs that ride on
  top of the current battle default.
- ``BALANCE_STRATEGIES`` — meta-balance + rule-balance pairs.

Each registry has a paired ``*_DEFAULT`` constant naming the current
submission default. PRs that change a default also update that constant.
"""

from vgc_ai.strategies.registry import (
    BALANCE_DEFAULT,
    BALANCE_STRATEGIES,
    BATTLE_DEFAULT,
    BATTLE_STRATEGIES,
    CHAMPIONSHIP_DEFAULT,
    CHAMPIONSHIP_STRATEGIES,
    BalanceStrategy,
    BattleStrategy,
    ChampionshipStrategy,
)

__all__ = [
    "BALANCE_DEFAULT",
    "BALANCE_STRATEGIES",
    "BATTLE_DEFAULT",
    "BATTLE_STRATEGIES",
    "CHAMPIONSHIP_DEFAULT",
    "CHAMPIONSHIP_STRATEGIES",
    "BalanceStrategy",
    "BattleStrategy",
    "ChampionshipStrategy",
]
