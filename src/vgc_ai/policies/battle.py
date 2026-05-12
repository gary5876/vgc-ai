"""Battle policies.

Currently aliases `GreedyBattlePolicy` as the default — submission-viable:
~0.04s per battle in doubles, ~90% win rate vs `RandomBattlePolicy`.
`TreeSearchBattlePolicy` is stronger (100% vs random in our test) but measures
~170s per battle / ~11s per turn in doubles, well over typical competition
per-turn limits. Tree is kept available via
`vgc2.agent.battle.TreeSearchBattlePolicy` for local benchmarking only.

Also exposes `GreedyWithSwitchABattlePolicy`, variant (a) of the greedy-with-
switch family: same as the base greedy-switch but with an additional
full-HP requirement on the bench candidate (rationale: avoid switching into
a damaged mon that immediately faints).

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


class GreedyWithSwitchABattlePolicy(GreedyBattlePolicy):  # type: ignore[misc]  # vgc2 is untyped; GreedyBattlePolicy resolves as Any under --strict
    """Greedy + one switching rule, tighten variant (a).

    For each active slot, if the active mon's HP fraction is below
    `LOW_HP_FRACTION` and there exists a **full-HP**, non-fainted bench mon
    with a STAB damaging move whose type is super-effective against any of
    the opponent's actives, switch to that bench mon. Otherwise fall back to
    greedy.

    The full-HP requirement (rationale: avoid switching into a damaged mon
    that immediately faints) is the only difference from the base
    greedy-switch variant.

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

        live_full_hp_reserve_idx = [
            i for i, r in enumerate(my_reserve) if r.hp > 0 and _is_full_hp(r)
        ]
        greedy_cmds: Any = super().decision(state, opp_view)
        if not live_full_hp_reserve_idx or greedy_cmds is None:
            return greedy_cmds  # type: ignore[no-any-return]

        cmds: list[BattleCommand] = list(greedy_cmds)
        used: set[int] = set()
        for slot, attacker in enumerate(my_active):
            if slot >= len(cmds) or attacker.hp == 0:
                continue
            max_hp = attacker.constants.stats[Stat.MAX_HP]
            if max_hp == 0 or attacker.hp / max_hp >= LOW_HP_FRACTION:
                continue
            switch_idx = self._pick_switch(my_reserve, opp_active, live_full_hp_reserve_idx, used)
            if switch_idx is not None:
                cmds[slot] = (-1, switch_idx)
                used.add(switch_idx)
        return cmds

    def _pick_switch(
        self,
        reserve: list[BattlingPokemon],
        opp_active: list[BattlingPokemon],
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
                for opp in opp_active:
                    if opp.hp == 0:
                        continue
                    if type_effectiveness_modifier(self.params, m.pkm_type, opp.types) > 1.0:
                        return idx
        return None


def _is_full_hp(mon: BattlingPokemon) -> bool:
    max_hp: int = mon.constants.stats[Stat.MAX_HP]
    hp: int = mon.hp
    return max_hp > 0 and hp >= max_hp


__all__ = ["GreedyWithSwitchABattlePolicy", "VgcAiBattlePolicy"]
