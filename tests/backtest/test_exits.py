"""SL/TP hit detection, inclusive equality, same-candle SL-first."""

from __future__ import annotations

from src.backtest.exits import ExitHit, evaluate_ohlcv_exit


def test_sl_only():
    assert evaluate_ohlcv_exit(101.0, 99.0, 99.5, 102.0) is ExitHit.SL


def test_tp_only():
    assert evaluate_ohlcv_exit(103.0, 100.0, 99.0, 102.0) is ExitHit.TP


def test_neither():
    assert evaluate_ohlcv_exit(101.0, 100.0, 99.0, 102.0) is ExitHit.NONE


def test_both_assume_sl_first():
    assert evaluate_ohlcv_exit(103.0, 98.0, 99.0, 102.0) is ExitHit.SL


def test_equality_triggers():
    assert evaluate_ohlcv_exit(101.0, 99.0, 99.0, 102.0) is ExitHit.SL
    assert evaluate_ohlcv_exit(102.0, 100.0, 99.0, 102.0) is ExitHit.TP
    assert evaluate_ohlcv_exit(102.0, 99.0, 99.0, 102.0) is ExitHit.SL
