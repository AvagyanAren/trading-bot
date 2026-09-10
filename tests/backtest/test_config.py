"""Backtest YAML loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.backtest.config import BacktestConfigError, load_backtest_config


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "backtest.yaml"
    path.write_text(body, encoding="utf-8")
    return path


VALID = """
backtest:
  starting_capital: 20
risk:
  risk_per_trade: 0.01
  stop_loss: 0.01
  take_profit: 0.02
execution:
  entry_model: next_open
  slippage: 0.0005
  same_candle_priority: stop_loss
fees:
  entry_rate: 0.001
  exit_rate: 0.001
"""


def test_load_valid_config(tmp_path: Path):
    config = load_backtest_config(_write(tmp_path, VALID))
    assert config.starting_capital == 20.0
    assert config.entry_model == "next_open"
    assert config.same_candle_priority == "stop_loss"
    assert config.slippage == 0.0005


def test_rejects_unknown_entry_model(tmp_path: Path):
    body = VALID.replace("next_open", "next_close")
    with pytest.raises(BacktestConfigError, match="entry_model"):
        load_backtest_config(_write(tmp_path, body))


def test_rejects_max_holding_candles(tmp_path: Path):
    body = VALID + "\nmax_holding_candles: 12\n"
    with pytest.raises(BacktestConfigError, match="max_holding_candles"):
        load_backtest_config(_write(tmp_path, body))


def test_rejects_non_positive_capital(tmp_path: Path):
    body = VALID.replace("starting_capital: 20", "starting_capital: 0")
    with pytest.raises(BacktestConfigError, match="starting_capital"):
        load_backtest_config(_write(tmp_path, body))
