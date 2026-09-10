"""Deterministic LONG-only entry signals on the v0.2 enriched store.

v0.3 answers: given the information known at the close of this candle, does
the configured hypothesis generate a LONG entry signal?

It does not calculate indicators, size positions, apply stop loss or take
profit, simulate fills, talk to Binance, or compute P&L.
"""
