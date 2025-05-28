"""Backtesting helpers (data loaders, metrics & evaluation)."""

from .data import load_ohlc, add_feeds_to_cerebro
from .metrics import (
    daily_returns,
    cagr,
    sharpe,
    max_drawdown,
    calmar,
)
from .evaluate import evaluate, evaluate_sync

__all__ = [
    "load_ohlc",
    "add_feeds_to_cerebro",
    "evaluate",
    "evaluate_sync",
    "daily_returns",
    "cagr",
    "sharpe",
    "max_drawdown",
    "calmar",
]
