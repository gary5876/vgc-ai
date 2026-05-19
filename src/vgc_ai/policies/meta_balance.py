"""Meta-balance policy (Balance Track).

The Balance Track (Meta + Rule merged) is new in the 2026 4th edition. A
``MetaBalancePolicy`` proposes changes to the move set and roster to shape
usage / win-rate diversity in the meta-game.

``NoOpMetaBalancePolicy`` is a legal "do nothing" submission: it returns
empty change lists. It exists to (a) stand up a submittable Balance Track
shell before any tuning work and (b) surface integration / serialisation
bugs early.

``MetaConstraints`` is still an empty stub in vgc2 v2.1.1 (the framework
TODO is visible in ``vgc2.agent.meta_balance``). Once Reis fills it in
(end-of-March 2026 roadmap), this policy will need to honour real
constraints.
"""

from __future__ import annotations

from vgc2.agent import (
    MetaBalancePolicy,
    MetaConstraints,
    MoveSet,
    MoveSetBalanceCommand,
    Roster,
    RosterBalanceCommand,
)


class NoOpMetaBalancePolicy(MetaBalancePolicy):  # type: ignore[misc]
    def decision(
        self,
        move_set: MoveSet,
        roster: Roster,
        constraints: MetaConstraints,
    ) -> tuple[MoveSetBalanceCommand, RosterBalanceCommand]:
        return [], []
