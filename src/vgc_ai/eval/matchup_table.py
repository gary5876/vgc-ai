"""Roster x Roster 1v1 matchup table built from actual battle simulation.

For each ordered pair (i, j) of distinct species in the roster, run
``n_battles_per_pair`` singleton-vs-singleton battles (one-mon teams,
``n_active=1``) with both sides playing ``GreedyBattlePolicy``. Return an
``N x N`` matrix where ``M[i][j]`` is species i's win rate against j over
those battles. Diagonal is 0.5 by convention.

Singleton battles are an approximation of the doubles-actually-played
contest format — they ignore positioning and double-targeting — but they
isolate raw 1v1 typing + stat + move synergy, which is the only signal the
team-build pre-game has. The table is computed once per roster and cached.

Greedy was chosen as the proxy battle policy because (a) it's the
plausible-opponent baseline, (b) per-battle cost is ~25 ms, so a 30-species
roster computes in ~3-4 minutes (n_battles=5 * 30*29 ordered pairs * ~25ms
* singleton speedup), which is a one-time precompute per championship.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from vgc2.agent import BattlePolicy
from vgc2.agent.battle import GreedyBattlePolicy
from vgc2.balance.meta import Roster
from vgc2.battle_engine import BattleEngine, BattleRuleParam, State
from vgc2.battle_engine.game_state import get_battle_teams
from vgc2.battle_engine.modifiers import Nature
from vgc2.battle_engine.pokemon import Pokemon, PokemonSpecies
from vgc2.battle_engine.team import Team
from vgc2.battle_engine.view import StateView, TeamView
from vgc2.competition.match import label_teams, run_battle

_DEFAULT_EVS: tuple[int, int, int, int, int, int] = (85, 85, 85, 85, 85, 85)
_DEFAULT_IVS: tuple[int, int, int, int, int, int] = (31, 31, 31, 31, 31, 31)
_DEFAULT_NATURE: Nature = Nature.SERIOUS
_STAB_MULTIPLIER: float = 1.5


def _top_moves(species: PokemonSpecies, max_pkm_moves: int) -> list[int]:
    """Pick top ``max_pkm_moves`` move indices by ``base_power * STAB``.

    Duplicates the heuristic in ``teambuild._move_priority`` but inlined here
    to keep this module standalone — ``teambuild`` should be the one importing
    matchup-table machinery, not vice versa.
    """
    species_types = set(species.types)

    def score(idx: int) -> float:
        m = species.moves[idx]
        stab = _STAB_MULTIPLIER if m.pkm_type in species_types else 1.0
        return float(m.base_power) * stab

    indices = sorted(range(len(species.moves)), key=lambda i: (-score(i), i))
    return indices[:max_pkm_moves]


def _build_singleton_team(species: PokemonSpecies, max_pkm_moves: int) -> Team:
    move_idx = _top_moves(species, max_pkm_moves)
    pkm = Pokemon(
        species=species,
        move_indexes=move_idx,
        level=100,
        evs=_DEFAULT_EVS,
        ivs=_DEFAULT_IVS,
        nature=_DEFAULT_NATURE,
    )
    return Team([pkm])


def _one_battle(
    species_a: PokemonSpecies,
    species_b: PokemonSpecies,
    max_pkm_moves: int,
    params: BattleRuleParam,
    policy_factory: type[BattlePolicy],
) -> int:
    """Run a single 1v1 battle. Returns 0 if A wins, 1 if B wins."""
    team_a = _build_singleton_team(species_a, max_pkm_moves)
    team_b = _build_singleton_team(species_b, max_pkm_moves)
    teams = (team_a, team_b)
    label_teams(teams)
    team_view = TeamView(team_a), TeamView(team_b)
    state = State(get_battle_teams(teams, n_active=1))
    state_view = (
        StateView(state, 0, team_view),
        StateView(state, 1, team_view),
    )
    engine = BattleEngine(state, debug=False)

    a = policy_factory()
    b = policy_factory()
    a.set_params(params)
    b.set_params(params)
    winner: int = run_battle(engine, (a, b), team_view, state_view, client=None)
    return winner


def _pair_winrate(
    species_a: PokemonSpecies,
    species_b: PokemonSpecies,
    n_battles_per_pair: int,
    max_pkm_moves: int,
    params: BattleRuleParam,
    policy_factory: type[BattlePolicy],
) -> float:
    wins_a = 0
    decided = 0
    for _ in range(n_battles_per_pair):
        winner = _one_battle(species_a, species_b, max_pkm_moves, params, policy_factory)
        if winner == 0:
            wins_a += 1
            decided += 1
        elif winner == 1:
            decided += 1
        # ties (no winner) are dropped
    if decided == 0:
        return 0.5
    return wins_a / decided


def build_matchup_table(
    roster: Roster,
    *,
    n_battles_per_pair: int = 5,
    max_pkm_moves: int = 4,
    params: BattleRuleParam | None = None,
    policy_factory: type[BattlePolicy] = GreedyBattlePolicy,
) -> npt.NDArray[np.float64]:
    """Compute an ``N x N`` matchup matrix from singleton-vs-singleton battles.

    Symmetric pairs are computed once: ``M[j][i] = 1 - M[i][j]``. The diagonal
    is 0.5 by convention (no mirror match is informative).
    """
    p = params or BattleRuleParam()
    n = len(roster)
    table = np.full((n, n), 0.5, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            wr = _pair_winrate(
                roster[i], roster[j], n_battles_per_pair, max_pkm_moves, p, policy_factory
            )
            table[i][j] = wr
            table[j][i] = 1.0 - wr
    return table


def roster_cache_key(roster: Roster) -> tuple[int, ...]:
    """Stable identity for the roster — used to invalidate the cache when the
    contest hands us a new one.

    Uses ``id(species)`` per slot so equality is object identity, not value
    equality. That's intentional: within one ``Championship`` the roster
    object is reused across epochs, but a fresh ``Championship`` builds a new
    list of species objects, even if they look superficially similar.
    """
    return tuple(id(s) for s in roster)


# Module-level cache shared across policies (team-build + selection). Keyed on
# roster identity AND build parameters; both team-build and selection consult
# this rather than each holding their own table. The cache is intentionally
# unbounded — a fresh Championship instance produces fresh species objects, so
# the prior key's entries become unreachable and Python eventually collects
# them.
_MATCHUP_TABLE_CACHE: dict[tuple[int, ...], npt.NDArray[np.float64]] = {}


def get_or_build_matchup_table(
    roster: Roster,
    *,
    n_battles_per_pair: int = 5,
    max_pkm_moves: int = 4,
    params: BattleRuleParam | None = None,
    policy_factory: type[BattlePolicy] = GreedyBattlePolicy,
) -> npt.NDArray[np.float64]:
    """Return the cached matchup table for ``roster``, building it on first call.

    The cache key bundles the roster identity, ``n_battles_per_pair``, and
    ``max_pkm_moves`` so tables built with different precision / move counts
    don't collide.
    """
    key = (*roster_cache_key(roster), n_battles_per_pair, max_pkm_moves)
    if key not in _MATCHUP_TABLE_CACHE:
        _MATCHUP_TABLE_CACHE[key] = build_matchup_table(
            roster,
            n_battles_per_pair=n_battles_per_pair,
            max_pkm_moves=max_pkm_moves,
            params=params,
            policy_factory=policy_factory,
        )
    return _MATCHUP_TABLE_CACHE[key]


def clear_matchup_table_cache() -> None:
    """Reset the module-level cache. Test-only helper; production code should
    not need to invalidate."""
    _MATCHUP_TABLE_CACHE.clear()
