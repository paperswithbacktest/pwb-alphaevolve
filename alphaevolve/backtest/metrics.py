"""
Common performance-metric helpers (numpy-friendly, no external deps).
"""

import numpy as np
import pandas as pd


def _to_np(arr):
    if isinstance(arr, (pd.Series, pd.DataFrame)):
        arr = arr.values
    return np.asarray(arr, dtype=float)


# ------------------------------------------------------------------ #
# BASIC METRICS
# ------------------------------------------------------------------ #
def daily_returns(equity_curve: pd.Series) -> np.ndarray:
    return _to_np(equity_curve.pct_change().dropna())


def cagr(equity_curve: pd.Series, periods_per_year: int = 252) -> float:
    arr = _to_np(equity_curve)
    n_years = len(arr) / periods_per_year
    return (arr[-1] / arr[0]) ** (1 / n_years) - 1


def sharpe(returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252) -> float:
    excess = returns - rf / periods_per_year
    if excess.std(ddof=1) == 0:
        return 0.0
    return np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1)


def max_drawdown(equity_curve: pd.Series) -> float:
    """Return *percentage* max drawdown (negative value)."""
    cummax = equity_curve.cummax()
    dd = (equity_curve - cummax) / cummax
    return dd.min()


def calmar(cagr_: float, mdd: float) -> float:
    return cagr_ / abs(mdd) if mdd != 0 else 0
