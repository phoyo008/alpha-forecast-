"""Streamlit dashboard for alpha-forecast.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from alpha_forecast.backtest import walk_forward_backtest
from alpha_forecast.data import load_prices
from alpha_forecast.evaluation import strategy_performance
from alpha_forecast.features import build_features
from alpha_forecast.models import MODEL_REGISTRY
from alpha_forecast.pipeline import run_comparison

st.set_page_config(page_title="alpha-forecast", layout="wide")

st.title("📈 alpha-forecast")
st.caption("Multi-model equity return forecasting & walk-forward backtesting — built on OpenBB.")

with st.sidebar:
    st.header("Configuration")
    symbol = st.text_input("Ticker", value="AAPL").upper().strip()
    horizon = st.slider("Forecast horizon (days)", 1, 20, 1)
    cost_bps = st.slider("Transaction cost (bps)", 0.0, 10.0, 1.0, 0.5)
    initial_train = st.slider("Initial training window (days)", 126, 756, 252, 21)
    step = st.slider("Retrain step (days)", 1, 63, 21)
    models = st.multiselect(
        "Models", list(MODEL_REGISTRY), default=list(MODEL_REGISTRY)
    )
    run = st.button("Run backtest", type="primary")

if run and models:
    with st.spinner("Loading data and running walk-forward backtest..."):
        board = run_comparison(
            symbol,
            horizon=horizon,
            models=models,
            initial_train=initial_train,
            step=step,
            cost_bps=cost_bps,
        )

    st.subheader(f"Leaderboard — {symbol} (horizon = {horizon}d)")
    st.dataframe(
        board.style.format(
            {
                "rmse": "{:.5f}",
                "r2": "{:.2%}",
                "dir_acc": "{:.2%}",
                "sharpe": "{:.2f}",
                "ann_return": "{:.2%}",
                "max_dd": "{:.2%}",
            }
        ),
        use_container_width=True,
    )

    best = board.iloc[0]["model"]
    naive_sharpe = board.loc[board["model"] == "naive", "sharpe"]
    msg = f"**Best model: `{best}`** (Sharpe {board.iloc[0]['sharpe']:.2f})."
    if not naive_sharpe.empty and board.iloc[0]["sharpe"] <= naive_sharpe.iloc[0]:
        msg += " ⚠️ It does **not** beat the naive baseline — treat as no real edge."
    st.markdown(msg)

    # Equity curves
    st.subheader("Out-of-sample equity curves")
    prices = load_prices(symbol, horizon=None) if False else load_prices(symbol)
    X, y = build_features(prices, horizon=horizon)
    curves = pd.DataFrame(index=None)
    equity_frames = {}
    for name in models:
        res = walk_forward_backtest(
            MODEL_REGISTRY[name], X, y, initial_train=initial_train, step=step
        )
        import numpy as np

        pos = np.sign(res.predictions).replace(0, 1.0)
        net = pos * res.actuals
        equity_frames[name] = (1 + net).cumprod()
    equity = pd.DataFrame(equity_frames)
    st.line_chart(equity)

    with st.expander("How to read this"):
        st.markdown(
            "- **dir_acc**: how often the model gets the direction right (>50% is the bar).\n"
            "- **sharpe**: risk-adjusted return of trading on the signal, net of costs.\n"
            "- A model is only interesting if it beats the **naive** baseline.\n"
            "- All predictions are strictly out-of-sample via expanding-window walk-forward."
        )
else:
    st.info("Configure parameters in the sidebar and click **Run backtest**.")
    st.markdown(
        "This dashboard runs a full data → features → model → walk-forward backtest "
        "pipeline. When run offline it uses a reproducible synthetic series so it "
        "always works; connect OpenBB or yfinance for live market data."
    )
