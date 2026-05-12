"""Battle policies.

Currently aliases `GreedyBattlePolicy` as the default — submission-viable:
~0.04s per battle in doubles, ~90% win rate vs `RandomBattlePolicy`.
`TreeSearchBattlePolicy` is stronger (100% vs random in our test) but measures
~170s per battle / ~11s per turn in doubles, well over typical competition
per-turn limits. Tree is kept available via
`vgc2.agent.battle.TreeSearchBattlePolicy` for local benchmarking only.

Also exposes `GreedyWithSwitchBBattlePolicy`, variant (b) of the greedy-with-
switch family: same trigger as the base greedy-switch (own active HP < 25%
with a STAB super-effective bench answer) but limited to **at most one
switch per turn** even when both actives are below the HP threshold
(rationale: tandem-switch trades momentum — giving up both attacks in a
single turn typically loses more than the one bad matchup it solves).

Target: beat `GreedyBattlePolicy` consistently while staying under a sub-second
per-turn budget. Probable approach: tabular Monte Carlo (cf. AurelianTactics
2024 3rd place) or a small search with hard time limit + heuristic eval.
"""

from __future__ import annotations

from typing import Any

from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.battle import GreedyBattlePolicy as VgcAiBattlePolicy
from vgc2.battle_engine import BattleCommand, State
from vgc2.battle_engine.damage_calculator import type_effectiveness_modifier
from vgc2.battle_engine.modifiers import Stat
from vgc2.battle_engine.pokemon import BattlingPokemon
from vgc2.battle_engine.view import TeamView

LOW_HP_FRACTION = 0.25


class GreedyWithSwitchBBattlePolicy(GreedyBattlePolicy):  # type: ignore[misc]  # vgc2 is untyped; GreedyBattlePolicy resolves as Any under --strict
    """Greedy + one switching rule, tighten variant (b).

    For each active slot, if the active mon's HP fraction is below
    `LOW_HP_FRACTION` and there exists a non-fainted bench mon with a STAB
    damaging move whose type is super-effective against any of the
    opponent's actives, switch to that bench mon. Otherwise fall back to
    greedy.

    Unlike the base variant, **at most one switch is issued per turn** even
    in doubles. If both actives qualify, only the lower-HP-fraction slot
    switches; the other attacks (rationale: tandem-switch trades momentum).
    """

    def decision(
        self,
        state: State,
        opp_view: TeamView | None = None,
    ) -> list[BattleCommand]:
        my_active = state.sides[0].team.active
        my_reserve = state.sides[0].team.reserve
        opp_active = state.sides[1].team.active

        live_reserve_idx = [i for i, r in enumerate(my_reserve) if r.hp > 0]
        greedy_cmds: Any = super().decision(state, opp_view)
        if not live_reserve_idx or greedy_cmds is None:
            return greedy_cmds  # type: ignore[no-any-return]

        cmds: list[BattleCommand] = list(greedy_cmds)
        candidates: list[tuple[float, int]] = []
        for slot, attacker in enumerate(my_active):
            if slot >= len(cmds) or attacker.hp == 0:
                continue
            max_hp = attacker.constants.stats[Stat.MAX_HP]
            if max_hp == 0:
                continue
            frac = attacker.hp / max_hp
            if frac >= LOW_HP_FRACTION:
                continue
            candidates.append((frac, slot))

        candidates.sort()
        for _, slot in candidates:
            switch_idx = self._pick_switch(my_reserve, opp_active, live_reserve_idx)
            if switch_idx is not None:
                cmds[slot] = (-1, switch_idx)
                break  # at most one switch per turn
        return cmds

    def _pick_switch(
        self,
        reserve: list[BattlingPokemon],
        opp_active: list[BattlingPokemon],
        candidates: list[int],
    ) -> int | None:
        for idx in candidates:
            mon = reserve[idx]
            for bm in mon.battling_moves:
                if bm.pp <= 0 or bm.disabled:
                    continue
                m = bm.constants
                if m.base_power <= 0:
                    continue
                if m.pkm_type not in mon.types:
                    continue
                for opp in opp_active:
                    if opp.hp == 0:
                        continue
                    if type_effectiveness_modifier(self.params, m.pkm_type, opp.types) > 1.0:
                        return idx
        return None


__all__ = ["GreedyWithSwitchBBattlePolicy", "VgcAiBattlePolicy"]
