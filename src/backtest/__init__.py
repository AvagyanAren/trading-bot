"""Deterministic historical execution simulator for v0.4.

v0.4 answers: if Strategy #1 had been traded sequentially on the historical
dataset under the configured execution and risk assumptions, which trades
would have occurred?

It does not recalculate indicators, does not change the v0.3 signal engine,
does not talk to Binance, and does not claim that the strategy is profitable.
"""
