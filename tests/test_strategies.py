"""Strategy-registry shape and factory tests.

The registry is data, not logic — but if a factory raises on import or
returns the wrong policy type, every downstream consumer (tournament
driver, reviewer loop) fails far from the source. These tests fail loudly
right at the import boundary instead.
"""

from __future__ import annotations

from vgc2.agent import (
    BattlePolicy,
    MetaBalancePolicy,
    RuleBalancePolicy,
    SelectionPolicy,
    TeamBuildPolicy,
)

from vgc_ai.strategies import (
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


def test_battle_registry_nonempty_and_keyed_by_name() -> None:
    assert BATTLE_STRATEGIES, "battle registry is empty"
    for key, strat in BATTLE_STRATEGIES.items():
        assert isinstance(strat, BattleStrategy)
        assert strat.name == key, f"registry key {key!r} != strategy.name {strat.name!r}"


def test_championship_registry_nonempty_and_keyed_by_name() -> None:
    assert CHAMPIONSHIP_STRATEGIES, "championship registry is empty"
    for key, strat in CHAMPIONSHIP_STRATEGIES.items():
        assert isinstance(strat, ChampionshipStrategy)
        assert strat.name == key


def test_balance_registry_nonempty_and_keyed_by_name() -> None:
    assert BALANCE_STRATEGIES, "balance registry is empty"
    for key, strat in BALANCE_STRATEGIES.items():
        assert isinstance(strat, BalanceStrategy)
        assert strat.name == key


def test_defaults_resolve_to_registered_strategies() -> None:
    assert BATTLE_DEFAULT in BATTLE_STRATEGIES
    assert CHAMPIONSHIP_DEFAULT in CHAMPIONSHIP_STRATEGIES
    assert BALANCE_DEFAULT in BALANCE_STRATEGIES


def test_battle_factories_produce_battle_policies() -> None:
    for strat in BATTLE_STRATEGIES.values():
        policy = strat.battle_policy()
        assert isinstance(policy, BattlePolicy), (
            f"{strat.name} factory returned {type(policy).__name__}, not BattlePolicy"
        )


def test_championship_factories_produce_paired_policies() -> None:
    for strat in CHAMPIONSHIP_STRATEGIES.values():
        tb = strat.team_build_policy()
        sel = strat.selection_policy()
        assert isinstance(tb, TeamBuildPolicy), (
            f"{strat.name} team_build_policy returned {type(tb).__name__}"
        )
        assert isinstance(sel, SelectionPolicy), (
            f"{strat.name} selection_policy returned {type(sel).__name__}"
        )


def test_balance_factories_produce_paired_policies() -> None:
    for strat in BALANCE_STRATEGIES.values():
        mb = strat.meta_balance_policy()
        rb = strat.rule_balance_policy()
        assert isinstance(mb, MetaBalancePolicy), (
            f"{strat.name} meta_balance_policy returned {type(mb).__name__}"
        )
        assert isinstance(rb, RuleBalancePolicy), (
            f"{strat.name} rule_balance_policy returned {type(rb).__name__}"
        )


def test_factories_return_fresh_instances() -> None:
    # Several real policies hold per-instance caches; sharing one across
    # competitors would entangle their decisions. The factory contract
    # guarantees a fresh object per call.
    strat = BATTLE_STRATEGIES[BATTLE_DEFAULT]
    assert strat.battle_policy() is not strat.battle_policy()

    cstrat = CHAMPIONSHIP_STRATEGIES[CHAMPIONSHIP_DEFAULT]
    assert cstrat.team_build_policy() is not cstrat.team_build_policy()
    assert cstrat.selection_policy() is not cstrat.selection_policy()
