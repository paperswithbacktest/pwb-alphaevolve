"""
Synchronous + async helpers to evaluate a *code string* defining a Backtrader
strategy.  The entry symbol must expose either:

    1. a subclass of `bt.Strategy` named `Strategy`, **or**
    2. a variable `STRATEGY_CLASS` pointing to a bt.Strategy subclass.

Returned KPI dict is JSON-serialisable for Mongo storage.
"""

import asyncio, inspect, importlib.util, sys, tempfile, types
from functools import partial
from pathlib import Path
from typing import Any, Sequence, Dict
import backtrader as bt
import pandas as pd

from examples import config as example_config
from .data import load_ohlc, add_feeds_to_cerebro
from . import metrics as mt


# ------------------------------------------------------------------ #
# INTERNAL HELPERS
# ------------------------------------------------------------------ #
def _load_module_from_code(code: str, name: str | None = None) -> types.ModuleType:
    """Create a temporary module from source code string."""
    name = name or f"strategy_{hash(code)}"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
        tmp.write(code)
        tmp_path = Path(tmp.name)

    spec = importlib.util.spec_from_file_location(name, tmp_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    tmp_path.unlink(missing_ok=True)
    return mod


def _find_strategy(mod: types.ModuleType) -> type[bt.Strategy]:
    for attr in ("Strategy", "STRATEGY_CLASS"):
        if hasattr(mod, attr):
            cls = getattr(mod, attr)
            if inspect.isclass(cls) and issubclass(cls, bt.Strategy):
                return cls
    # fallback: first subclass of bt.Strategy in module
    for v in mod.__dict__.values():
        if (
            inspect.isclass(v)
            and issubclass(v, bt.Strategy)
            and not v.__name__ == "BaseLoggingStrategy"
        ):
            return v
    raise ValueError("No compatible Strategy class found in code snippet.")


def _run_backtest(
    strategy_cls: type[bt.Strategy], symbols: Sequence[str] = example_config.DEFAULT_SYMBOLS
) -> Dict[str, Any]:
    df = load_ohlc(tuple(symbols), start=example_config.START_DATE)
    cerebro = bt.Cerebro()
    add_feeds_to_cerebro(df, cerebro)
    cerebro.addstrategy(strategy_cls)
    cerebro.broker.set_cash(100_000)

    # execute
    strat = cerebro.run(maxcpus=1)[0]  # serial for determinism

    # metrics
    curve = pd.Series(
        [pt["value"] for pt in strat.equity_curve],
        index=[pt["date"] for pt in strat.equity_curve],
        name="equity",
    )
    rets = mt.daily_returns(curve)
    kpis = {
        "total_return": curve.iloc[-1] / curve.iloc[0] - 1,
        "cagr": mt.cagr(curve),
        "sharpe": mt.sharpe(rets),
        "max_drawdown": float(mt.max_drawdown(curve)),
        "calmar": mt.calmar(mt.cagr(curve), mt.max_drawdown(curve)),
        "n_days": int(curve.size),
    }
    return kpis


# ------------------------------------------------------------------ #
# PUBLIC API
# ------------------------------------------------------------------ #
def evaluate_sync(
    code: str, *, symbols: Sequence[str] = example_config.DEFAULT_SYMBOLS
) -> Dict[str, Any]:
    """Blocking evaluation; raises on errors (handled by controller)."""
    mod = _load_module_from_code(code)
    strat_cls = _find_strategy(mod)
    return _run_backtest(strat_cls, symbols=symbols)


async def evaluate(
    code: str, *, symbols: Sequence[str] = example_config.DEFAULT_SYMBOLS
) -> Dict[str, Any]:
    """
    Async wrapper so the evolution controller can `await`.
    Runs the sync back-test in a thread to avoid event-loop blocking.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, partial(evaluate_sync, code, symbols=symbols)
    )
