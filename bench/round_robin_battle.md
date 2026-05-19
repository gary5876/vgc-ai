| vs | greedy | heuristic_det | random | tabular_mc |
|---|---|---|---|---|
| greedy | — | 0.490 | 0.955 | 0.515 |
| heuristic_det | 0.510 | — | 0.925 | 0.575 |
| random | 0.045 | 0.075 | — | 0.020 |
| tabular_mc | 0.485 | 0.425 | 0.980 | — |

Run: n=200, team_size=4, n_active=2, 2026-05-19.

Note: `tabular_mc` is **untrained** here. Empty Q-table → delegates every
decision to `GreedyBattlePolicy` (per AurelianTactics 2024 inference logic).
Previous row (random fallback, pre-fix) showed 0.035 vs greedy / 0.500 vs
random; that was a single-line inference bug masquerading as a training
problem across five prior iterations.
