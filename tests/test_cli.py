"""Unit tests for CLI argument parsing — no Showdown needed."""

import pytest

from vgc_ai.cli import build_parser


def test_play_defaults() -> None:
    parser = build_parser()
    args = parser.parse_args(["play"])
    assert args.command == "play"
    assert args.vs == "random"
    assert args.n == 10
    assert args.format == "gen9randomdoublesbattle"


def test_play_custom_n() -> None:
    parser = build_parser()
    args = parser.parse_args(["play", "--n", "3"])
    assert args.n == 3


def test_play_custom_format() -> None:
    parser = build_parser()
    args = parser.parse_args(["play", "--format", "gen9vgc2026regi"])
    assert args.format == "gen9vgc2026regi"


def test_unknown_subcommand_errors() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["nope"])
