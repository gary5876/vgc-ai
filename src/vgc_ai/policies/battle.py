"""Battle policies.

Currently aliases `GreedyBattlePolicy` as the default — submission-viable:
~0.04s per battle in doubles, ~90% win rate vs `RandomBattlePolicy`.
`TreeSearchBattlePolicy` is stronger (100% vs random in our test) but measures
~170s per battle / ~11s per turn in doubles, well over typical competition
per-turn limits. Tree is kept available via
`vgc2.agent.battle.TreeSearchBattlePolicy` for local benchmarking only.

Also exposes `GreedyWithSwitchCBattlePolicy`, variant (c) of the greedy-with-
switch family: same trigger as the base greedy-switch (own active HP < 25%
with a STAB super-effective bench answer) but the bench picker scores
candidates by `max(type_effectiveness x stab_modifier)` over their damaging
moves and switches to the highest-scoring eligible candidate instead of the
first match (rationale: not all super-effective is equal — a 4x STAB hit
beats a 2x STAB hit).

Target: beat `GreedyBattlePolicy` consistently while staying under a sub-second
per-turn budget. Probable approach: tabular Monte Carlo (cf. AurelianTactics
2024 3rd place) or a small search with hard time limit + heuristic eval.
"""

from __future__ import annotations

from typing import Any

from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.agent.battle import GreedyBattlePolicy as VgcAiBattlePolicy
from vgc2.battle_engine import BattleCommand, State
from vgc2.battle_engine.damage_calculator import (
    stab_modifier,
    type_effectiveness_modifier,
)
from vgc2.battle_engine.modifiers import Stat
from vgc2.battle_engine.pokemon import BattlingPokemon
from vgc2.battle_engine.view import TeamView

LOW_HP_FRACTION = 0.25


class GreedyWithSwitchCBattlePolicy(GreedyBattlePolicy):  # type: ignore[misc]  # vgc2 is untyped; GreedyBattlePolicy resolves as Any under --strict
    """Greedy + one switching rule, tighten variant (c).

    Trigger is identical to the base greedy-switch: for each active slot, if
    the active mon's HP fraction is below `LOW_HP_FRACTION`, look for a
    bench mon that can answer with a STAB damaging move that is at least
    super-effective against some live opponent active. The difference is
    the **picker**: among bench mons that pass the STAB super-effective
    gate, switch to the one whose best damaging move scores highest under
    `max(type_effectiveness * stab_modifier)` over all its damaging moves
    against live opponent actives.

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
            switch_idx = self._pick_switch(my_reserve, opp_active, live_reserve_idx, used)
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
        best_idx: int | None = None
        best_score = -1.0
        for idx in candidates:
            if idx in used:
                continue
            mon = reserve[idx]
            if not self._has_stab_super_effective(mon, opp_active):
                continue
            score = self._candidate_score(mon, opp_active)
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def _has_stab_super_effective(
        self,
        mon: BattlingPokemon,
        opp_active: list[BattlingPokemon],
    ) -> bool:
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
                    return True
        return False

    def _candidate_score(
        self,
        mon: BattlingPokemon,
        opp_active: list[BattlingPokemon],
    ) -> float:
        best = 0.0
        for bm in mon.battling_moves:
            if bm.pp <= 0 or bm.disabled:
                continue
            m = bm.constants
            if m.base_power <= 0:
                continue
            stab = stab_modifier(self.params, mon, m)
            for opp in opp_active:
                if opp.hp == 0:
                    continue
                eff = type_effectiveness_modifier(self.params, m.pkm_type, opp.types)
                score = eff * stab
                if score > best:
                    best = score
        return best


__all__ = ["GreedyWithSwitchCBattlePolicy", "VgcAiBattlePolicy"]
