"""OHLCV stop/target hit detection, including same-candle SL-first."""

from __future__ import annotations

from enum import Enum

from .config import SAME_CANDLE_STOP_LOSS


class ExitHit(str, Enum):
    """Intrabar exit decision. TIME_EXIT is intentionally absent."""

    NONE = "NONE"
    SL = "SL"
    TP = "TP"


def evaluate_ohlcv_exit(
    high: float,
    low: float,
    stop_loss: float,
    take_profit: float,
    *,
    same_candle_priority: str = SAME_CANDLE_STOP_LOSS,
) -> ExitHit:
    """Inclusive touch. If both SL and TP are reachable, assume SL first.

    Does not read candle close. Does not infer intrabar order from close.
    """
    sl_hit = low <= stop_loss
    tp_hit = high >= take_profit
    if sl_hit and tp_hit:
        if same_candle_priority != SAME_CANDLE_STOP_LOSS:
            raise ValueError(
                f"Unsupported same_candle_priority {same_candle_priority!r}"
            )
        return ExitHit.SL
    if sl_hit:
        return ExitHit.SL
    if tp_hit:
        return ExitHit.TP
    return ExitHit.NONE
