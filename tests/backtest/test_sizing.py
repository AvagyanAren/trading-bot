"""Position sizing tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.backtest.fills import long_entry_price
from src.backtest.sizing import size_long_entry

from .conftest import make_backtest_config


def test_twenty_dollar_cash_cap_binds(tmp_path: Path):
    config = make_backtest_config(tmp_path)
    sized = size_long_entry(20.0, 100.0, config)
    assert sized.affordable is True
    entry = long_entry_price(100.0, config.slippage)
    theoretical = 20.0 / entry
    cash_cap = 20.0 / (entry * (1.0 + config.entry_rate))
    assert sized.quantity == pytest.approx(min(theoretical, cash_cap))
    assert sized.quantity == pytest.approx(cash_cap)
    assert sized.position_value + sized.entry_fee == pytest.approx(20.0)
    assert sized.risk_amount == pytest.approx(sized.quantity * entry * 0.01)
    assert sized.risk_amount < 0.20


def test_size_shrinks_with_equity(tmp_path: Path):
    config = make_backtest_config(tmp_path)
    full = size_long_entry(20.0, 100.0, config)
    half = size_long_entry(10.0, 100.0, config)
    assert half.quantity < full.quantity
    assert half.quantity == pytest.approx(full.quantity * 0.5)


def test_zero_open_is_not_affordable(tmp_path: Path):
    config = make_backtest_config(tmp_path)
    sized = size_long_entry(20.0, 0.0, config)
    assert sized.affordable is False
    assert sized.quantity == 0.0


def test_uses_execution_open_not_a_frozen_price(tmp_path: Path):
    config = make_backtest_config(tmp_path)
    at_100 = size_long_entry(20.0, 100.0, config)
    at_200 = size_long_entry(20.0, 200.0, config)
    assert at_200.entry_reference_price == 200.0
    assert at_200.quantity < at_100.quantity
