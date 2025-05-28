#!/usr/bin/env python
"""Streamlit dashboard for pwb‑alphaevolve

Run with:
    streamlit run scripts/dashboard.py

Features
--------
* Live hall‑of‑fame table (top‑K Sharpe)
* Select a program → run fresh back‑test and chart the equity curve
* Inspect source code in expandable section
"""
from __future__ import annotations
import textwrap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from alphaevolve.store.sqlite import ProgramStore
from examples import config as example_config
from alphaevolve.backtest.evaluate import (
    _load_module_from_code,  # type: ignore  (private helper is okay for internal app)
    _find_strategy,  # type: ignore
    _run_backtest,  # type: ignore
)

st.set_page_config(page_title="Alpha‑Evolve Dashboard", layout="wide")
store = ProgramStore()

# ------------------------------------------------------------------
# Hall of Fame table
# ------------------------------------------------------------------
st.title("🏆 AlphaEvolve Hall of Fame")
TOP_K = st.sidebar.slider("Top K strategies", 3, 50, 10)
hof_rows = store.top_k(k=TOP_K, metric=example_config.HOF_METRIC)

if not hof_rows:
    st.info("Hall‑of‑Fame is empty – run the controller first.")
    st.stop()

table = pd.DataFrame(
    [
        {
            "id": r["id"],
            "sharpe": r["metrics"]["sharpe"],
            "calmar": r["metrics"]["calmar"],
            "cagr": r["metrics"]["cagr"],
            "max‑dd": r["metrics"]["max_drawdown"],
            "total‑ret": r["metrics"]["total_return"],
        }
        for r in hof_rows
    ]
)

st.dataframe(table, use_container_width=True)

# ------------------------------------------------------------------
# Strategy viewer & equity curve
# ------------------------------------------------------------------
selected_id = st.selectbox("Select a program to inspect", table["id"].tolist())
selected = store.get(selected_id)

if selected is None:
    st.error("Program not found in store.")
    st.stop()

col_code, col_chart = st.columns([1, 2])

with col_code:
    st.subheader("Source code")
    st.code(textwrap.dedent(selected["code"]))

with col_chart:
    st.subheader("Equity curve (fresh back‑test)")

    # run back‑test locally (re‑using evaluator internals)
    try:
        mod = _load_module_from_code(selected["code"])
        strat_cls = _find_strategy(mod)
        kpis = _run_backtest(
            strat_cls
        )  # returns KPIs only but we changed fn to expose curve below
    except Exception as e:
        st.error(f"Failed to back‑test: {e}")
    else:
        # _run_backtest currently returns KPIs dict only; re‑run here to also get curve
        from alphaevolve.backtest import load_ohlc, add_feeds_to_cerebro
        import backtrader as bt

        symbols = ("SPY", "EFA", "IEF", "VNQ", "GSG")
        df = load_ohlc(symbols, start="1990-01-01")
        cerebro = bt.Cerebro()
        add_feeds_to_cerebro(df, cerebro)
        cerebro.addstrategy(strat_cls)
        cerebro.broker.set_cash(100_000)
        strat_instance = cerebro.run(maxcpus=1)[0]
        curve = pd.Series(
            [pt["value"] for pt in strat_instance.equity_curve],
            index=[pt["date"] for pt in strat_instance.equity_curve],
            name="equity",
        )

        fig, ax = plt.subplots()
        curve.plot(ax=ax)
        ax.set_ylabel("Portfolio value ($)")
        ax.set_title(f"Equity curve – Sharpe {kpis['sharpe']:.2f}")
        st.pyplot(fig)
