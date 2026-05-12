"""Battle policies.

Currently aliases `GreedyBattlePolicy` as the default — submission-viable:
~0.04s per battle in doubles, ~90% win rate vs `RandomBattlePolicy`.
`TreeSearchBattlePolicy` is stronger (100% vs random in our test) but measures
~170s per battle / ~11s per turn in doubles, well over typical competition
per-turn limits. Tree is kept available via
`vgc2.agent.battle.TreeSearchBattlePolicy` for local benchmarking only.

Also exposes `GreedyWithSwitchDBattlePolicy`, variant (d) of the greedy-with-
switch family: same trigger as the base greedy-switch (own active HP < 25%)
but the candidate must have a STAB damaging move that is at least
super-effective against the **slot-aligned** opponent (the mon currently in
front of the low-HP active), not "any opponent" (rationale: switching to
threaten the wrong slot doesn't help the active that is leaving).

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


class GreedyWithSwitchDBattlePolicy(GreedyBattlePolicy):  # type: ignore[misc]  # vgc2 is untyped; GreedyBattlePolicy resolves as Any under --strict
    """Greedy + one switching rule, tighten variant (d).

    For each active slot, if the active mon's HP fraction is below
    `LOW_HP_FRACTION`, look for a non-fainted bench mon with a STAB damaging
    move whose type is super-effective against the opponent in the **same
    slot** (the one currently in front of the low-HP active). If that
    opponent slot is empty or fainted, do not switch.

    In doubles, both slots may switch in the same turn but never to the same
    bench index.
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
        used: set[int] = set()
        for slot, attacker in enumerate(my_active):
            if slot >= len(cmds) or attacker.hp == 0:
                continue
            max_hp = attacker.constants.stats[Stat.MAX_HP]
            if max_hp == 0 or attacker.hp / max_hp >= LOW_HP_FRACTION:
                continue
            if slot >= len(opp_active):
                continue
            opp = opp_active[slot]
            if opp.hp == 0:
                continue
            switch_idx = self._pick_switch(my_reserve, opp, live_reserve_idx, used)
            if switch_idx is not None:
                cmds[slot] = (-1, switch_idx)
                used.add(switch_idx)
        return cmds

    def _pick_switch(
        self,
        reserve: list[BattlingPokemon],
        opp: BattlingPokemon,
        candidates: list[int],
        used: set[int],
    ) -> int | None:
        for idx in candidates:
            if idx in used:
                continue
            mon = reserve[idx]
            for bm in mon.battling_moves:
                if bm.pp <= 0 or bm.disabled:
                    continue
                m = bm.constants
                if m.base_power <= 0:
                    continue
                if m.pkm_type not in mon.types:
                    continue
                if type_effectiveness_modifier(self.params, m.pkm_type, opp.types) > 1.0:
                    return idx
        return None


__all__ = ["GreedyWithSwitchDBattlePolicy", "VgcAiBattlePolicy"]
