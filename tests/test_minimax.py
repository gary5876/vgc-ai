"""Unit tests for ``solve_row_minimax_policy``.

Standard zero-sum 2-player matrix games with known equilibria, plus the
documented degenerate cases (empty / single-row tables).
"""

from __future__ import annotations

import numpy as np

from vgc_ai.eval.minimax import solve_row_minimax_policy


def test_empty_matrix_returns_empty() -> None:
    table = np.zeros((0, 0), dtype=np.float64)
    p = solve_row_minimax_policy(table)
    assert p.shape == (0,)


def test_single_row_returns_one() -> None:
    table = np.array([[1.0, -1.0, 0.5]], dtype=np.float64)
    p = solve_row_minimax_policy(table)
    np.testing.assert_allclose(p, [1.0])


def test_no_columns_returns_uniform() -> None:
    table = np.zeros((3, 0), dtype=np.float64)
    p = solve_row_minimax_policy(table)
    np.testing.assert_allclose(p, [1 / 3, 1 / 3, 1 / 3])


def test_dominant_strategy_returns_pure() -> None:
    # Row 0 dominates row 1 against every column: LP should pick row 0
    # outright.
    table = np.array([[2.0, 1.0], [-1.0, -1.0]], dtype=np.float64)
    p = solve_row_minimax_policy(table)
    np.testing.assert_allclose(p, [1.0, 0.0], atol=1e-6)


def test_matching_pennies_returns_uniform() -> None:
    # Classic symmetric matching-pennies has unique Nash p = (0.5, 0.5)
    # for both players.
    table = np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=np.float64)
    p = solve_row_minimax_policy(table)
    np.testing.assert_allclose(p, [0.5, 0.5], atol=1e-6)


def test_rock_paper_scissors_returns_uniform() -> None:
    # RPS Nash equilibrium is uniform over all three actions.
    table = np.array(
        [
            [0.0, -1.0, 1.0],
            [1.0, 0.0, -1.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    p = solve_row_minimax_policy(table)
    np.testing.assert_allclose(p, [1 / 3, 1 / 3, 1 / 3], atol=1e-6)


def test_output_is_probability_distribution() -> None:
    # Arbitrary 4x3 matrix.
    table = np.array(
        [
            [0.2, -0.3, 0.5],
            [-0.1, 0.4, 0.0],
            [0.3, 0.1, -0.2],
            [-0.4, 0.0, 0.3],
        ],
        dtype=np.float64,
    )
    p = solve_row_minimax_policy(table)
    assert p.shape == (4,)
    assert p.sum() == 1.0 or abs(p.sum() - 1.0) < 1e-9
    assert (p >= 0.0).all()


def test_non_square_matrix_works() -> None:
    # 2 rows, 4 columns: LP must accept the rectangular shape.
    table = np.array([[1.0, 0.0, -1.0, 0.5], [-0.5, 1.0, 0.0, -0.5]], dtype=np.float64)
    p = solve_row_minimax_policy(table)
    assert p.shape == (2,)
    assert abs(p.sum() - 1.0) < 1e-9


def test_value_at_equilibrium_is_min_over_columns() -> None:
    # Verify the LP found a real equilibrium: for the optimal p, the
    # worst column expected payoff (the LP value v) must be at least
    # as good as any pure strategy worst case.
    table = np.array(
        [
            [0.6, -0.2, 0.1],
            [-0.1, 0.5, 0.3],
            [0.2, 0.1, -0.3],
        ],
        dtype=np.float64,
    )
    p = solve_row_minimax_policy(table)
    v = float((p @ table).min())
    worst_pure = float(table.min(axis=1).max())
    assert v + 1e-6 >= worst_pure
