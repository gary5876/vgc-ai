"""LP-minimax solver for zero-sum 2-player matrix games.

Generalises ``teambuild._solve_minimax_policy`` to non-square payoff
matrices so the same routine can drive battle-time selection (where the
row count = our team size and the column count = opponent team size, and
they need not match) as well as team-build (square roster x roster).

For an ``m x n`` payoff matrix ``M`` where ``M[i, j]`` is the row
player's payoff when they play action ``i`` and the column player plays
action ``j``, ``solve_row_minimax_policy`` returns the row player's
max-min Nash equilibrium mixing distribution ``p`` of length ``m``.

LP formulation (standard zero-sum minimax):

    variables x = [v, p_0, ..., p_{m-1}]
    minimize   -v                                (maximize v)
    subject to v - sum_i p_i * M[i, j] <= 0      for each column j
               sum_i p_i = 1
               p_i >= 0,  v unbounded

Falls back to uniform if the LP fails or degenerates.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def solve_row_minimax_policy(
    table: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Row player's max-min Nash policy over a zero-sum payoff matrix.

    ``table[i, j]`` is the row player's payoff when row ``i`` is played
    against column ``j``. Returns a length-``m`` distribution over rows
    (``m == table.shape[0]``) whose worst-case expected payoff is
    maximal.

    Degenerate cases:

    - ``m == 0``: returns the empty array.
    - ``n == 0`` (no opp actions): every row distribution is trivially
      optimal; returns uniform.
    - ``m == 1``: only one row; returns ``[1.0]``.
    - LP solver failure / all-zero solution: returns uniform.
    """
    from scipy.optimize import linprog

    m = int(table.shape[0])
    if m == 0:
        return np.zeros(0, dtype=np.float64)
    if m == 1:
        return np.ones(1, dtype=np.float64)
    n = int(table.shape[1])
    if n == 0:
        return np.full(m, 1.0 / m, dtype=np.float64)

    c = np.zeros(m + 1, dtype=np.float64)
    c[0] = -1.0

    a_ub = np.zeros((n, m + 1), dtype=np.float64)
    a_ub[:, 0] = 1.0
    a_ub[:, 1:] = -table.T
    b_ub = np.zeros(n, dtype=np.float64)

    a_eq = np.zeros((1, m + 1), dtype=np.float64)
    a_eq[0, 1:] = 1.0
    b_eq = np.array([1.0], dtype=np.float64)

    bounds: list[tuple[float | None, float | None]] = [(None, None)] + [(0.0, None)] * m

    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds)
    if not result.success:
        return np.full(m, 1.0 / m, dtype=np.float64)
    p: npt.NDArray[np.float64] = np.asarray(result.x[1:], dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = float(p.sum())
    if s <= 0.0:
        return np.full(m, 1.0 / m, dtype=np.float64)
    return p / s


__all__ = ["solve_row_minimax_policy"]
