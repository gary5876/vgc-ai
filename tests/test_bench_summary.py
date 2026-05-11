"""Tests for ``bench.summary``: aggregation + Markdown rendering + CLI edges."""

from __future__ import annotations

import csv
from pathlib import Path

from bench.summary import (
    BASELINES,
    aggregate,
    load_rows,
    main,
    render,
    summarize,
)

CSV_HEADER = [
    "timestamp",
    "policy_a",
    "policy_b",
    "n_battles",
    "wins_a",
    "wins_b",
    "ties",
    "win_rate_a",
    "ci95_low",
    "ci95_high",
    "elapsed_sec",
    "avg_battle_ms",
    "avg_turn_ms_a",
    "avg_turn_ms_b",
]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_HEADER)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in CSV_HEADER})


def test_load_rows_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_rows(tmp_path / "absent.csv", 50) == []


def test_summarize_empty_leaderboard_prints_no_data(tmp_path: Path) -> None:
    csv_path = tmp_path / "leaderboard.csv"
    _write(csv_path, [])
    assert summarize(csv_path, 50) == "no data yet"


def test_summarize_missing_file_returns_no_data(tmp_path: Path) -> None:
    assert summarize(tmp_path / "nope.csv", 50) == "no data yet"


def test_aggregate_sums_both_orientations() -> None:
    """A row ``heuristic vs greedy`` and ``greedy vs heuristic`` must combine."""
    rows: list[dict[str, str]] = [
        {"policy_a": "heuristic", "policy_b": "greedy", "wins_a": "12", "wins_b": "8"},
        {"policy_a": "greedy", "policy_b": "heuristic", "wins_a": "9", "wins_b": "11"},
    ]
    policies, totals = aggregate(rows)
    assert "heuristic" in policies and "greedy" in policies
    # heuristic vs greedy: 12 (as A) + 11 (as B) = 23 wins in 40 decided games
    assert totals[("heuristic", "greedy")] == (23, 40)
    # greedy vs heuristic is not aggregated because heuristic is not a baseline


def test_aggregate_skips_zero_decided_rows() -> None:
    rows = [{"policy_a": "greedy", "policy_b": "random", "wins_a": "0", "wins_b": "0"}]
    policies, totals = aggregate(rows)
    assert policies == [] and totals == {}


def test_last_n_slices_most_recent_rows(tmp_path: Path) -> None:
    csv_path = tmp_path / "leaderboard.csv"
    rows = [
        {"policy_a": "greedy", "policy_b": "random", "wins_a": "20", "wins_b": "0"}
        for _ in range(10)
    ] + [{"policy_a": "greedy", "policy_b": "random", "wins_a": "0", "wins_b": "20"}]
    _write(csv_path, rows)  # type: ignore[arg-type]
    last1 = load_rows(csv_path, 1)
    assert len(last1) == 1 and last1[0]["wins_b"] == "20"
    last3 = load_rows(csv_path, 3)
    assert len(last3) == 3 and last3[-1]["wins_b"] == "20"


def test_render_produces_table_with_baseline_columns() -> None:
    rows: list[dict[str, str]] = [
        {"policy_a": "greedy", "policy_b": "random", "wins_a": "18", "wins_b": "2"},
        {"policy_a": "heuristic", "policy_b": "greedy", "wins_a": "11", "wins_b": "9"},
    ]
    policies, totals = aggregate(rows)
    table = render(policies, totals)
    lines = table.splitlines()
    # Header row + separator row + one row per policy
    assert len(lines) == 2 + len(policies)
    assert lines[0].startswith("| policy ")
    for baseline in BASELINES:
        assert f"vs {baseline}" in lines[0]
    # heuristic listed before greedy (challengers first)
    pol_order = [line.split("|")[1].strip() for line in lines[2:]]
    assert pol_order.index("heuristic") < pol_order.index("greedy")
    # Diagonal cell (greedy vs greedy) is em-dash
    greedy_row = next(line for line in lines[2:] if line.split("|")[1].strip() == "greedy")
    assert "—" in greedy_row


def test_render_diagonal_em_dash_when_no_data() -> None:
    table = render(["heuristic", "greedy"], {})
    # Every body cell has no data → em-dash everywhere
    assert table.count("—") >= 2 * len(BASELINES)


def test_cli_default_path_missing_prints_no_data(
    tmp_path: Path, capsys: object, monkeypatch: object
) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[attr-defined]
    rc = main([])
    assert rc == 0
    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "no data yet" in captured.out


def test_cli_renders_table(tmp_path: Path, capsys: object) -> None:
    csv_path = tmp_path / "leaderboard.csv"
    _write(
        csv_path,
        [
            {"policy_a": "greedy", "policy_b": "random", "wins_a": "18", "wins_b": "2"},
            {"policy_a": "heuristic", "policy_b": "greedy", "wins_a": "11", "wins_b": "9"},
        ],
    )
    rc = main(["--path", str(csv_path), "--last", "10"])
    assert rc == 0
    out = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "vs greedy" in out and "vs random" in out
    assert "heuristic" in out
