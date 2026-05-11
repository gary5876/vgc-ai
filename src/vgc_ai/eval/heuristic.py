"""Heuristic evaluation of a battle state from side 0's perspective.

Returns a single float — positive favors side 0, negative favors side 1.
Components are kept cheap so the eval can be called inside a per-action
forward-search loop and still fit a sub-second-per-turn budget.

Weights are intentional and reflect the dominant signals in vgc2 doubles:

- ``HP_WEIGHT`` (1.0): scaled-HP differential over the whole party. The
  baseline currency — each Pokemon's ``hp / max_hp`` contributes 1.
- ``FAINT_WEIGHT`` (3.0): bonus per opposing faint vs penalty per own
  faint. Faints are far more decisive than equal-magnitude HP damage
  because they remove a future action; matches the 3x asymmetry in
  vgc2's own ``eval_state``.
- ``MATCHUP_WEIGHT`` (0.5): per-attacker best damaging-move type
  effectiveness on the opposing active line, minus the same for the
  opponent. Discourages keeping a 2x-weak Pokemon in front.
- ``ACTIVE_HP_WEIGHT`` (0.5): mean ``hp / max_hp`` of active Pokemon.
  Without this, the eval is indifferent to keeping the bench healthy vs
  the frontline. The frontline is what attacks next turn, so a small
  bias toward it improves switch decisions.
"""

from __future__ import annotations

from vgc2.battle_engine import BattleRuleParam, State
from vgc2.battle_engine.damage_calculator import type_effectiveness_modifier
from vgc2.battle_engine.modifiers import Stat
from vgc2.battle_engine.pokemon import BattlingPokemon
from vgc2.battle_engine.team import BattlingTeam

HP_WEIGHT = 1.0
FAINT_WEIGHT = 3.0
MATCHUP_WEIGHT = 0.5
ACTIVE_HP_WEIGHT = 0.5


def _hp_ratio(pkm: BattlingPokemon) -> float:
    max_hp = pkm.constants.stats[Stat.MAX_HP]
    return pkm.hp / max_hp if max_hp > 0 else 0.0


def _team_hp_sum(team: BattlingTeam) -> float:
    return sum(_hp_ratio(p) for p in team.active + team.reserve)


def _team_fainted_count(team: BattlingTeam) -> int:
    return sum(1 for p in team.active + team.reserve if p.hp == 0)


def _matchup_score(
    params: BattleRuleParam,
    attackers: list[BattlingPokemon],
    defenders: list[BattlingPokemon],
) -> float:
    """Sum over attackers of (best type multiplier across damaging moves) - 1.

    Neutral hits contribute 0, super-effective positive, resisted negative.
    Status-only moves (``base_power == 0``) and unusable moves are skipped.
    """
    if not defenders:
        return 0.0
    score = 0.0
    for atk in attackers:
        if atk.hp == 0:
            continue
        best = 1.0
        any_move = False
        for move in atk.battling_moves:
            if move.pp == 0 or move.disabled:
                continue
            if move.constants.base_power == 0:
                continue
            any_move = True
            for d in defenders:
                if d.hp == 0:
                    continue
                m = type_effectiveness_modifier(params, move.constants.pkm_type, d.types)
                if m > best:
                    best = m
        if any_move:
            score += best - 1.0
    return score


def _active_hp_mean(team: BattlingTeam) -> float:
    if not team.active:
        return 0.0
    return sum(_hp_ratio(p) for p in team.active) / len(team.active)


def evaluate(state: State, params: BattleRuleParam | None = None) -> float:
    """Score the state from side 0's perspective. Higher is better for side 0."""
    p = params or BattleRuleParam()
    t0, t1 = state.sides[0].team, state.sides[1].team

    hp_diff = _team_hp_sum(t0) - _team_hp_sum(t1)
    faint_diff = _team_fainted_count(t1) - _team_fainted_count(t0)
    matchup_diff = _matchup_score(p, t0.active, t1.active) - _matchup_score(p, t1.active, t0.active)
    active_hp_diff = _active_hp_mean(t0) - _active_hp_mean(t1)

    return (
        HP_WEIGHT * hp_diff
        + FAINT_WEIGHT * faint_diff
        + MATCHUP_WEIGHT * matchup_diff
        + ACTIVE_HP_WEIGHT * active_hp_diff
    )
