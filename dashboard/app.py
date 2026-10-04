"""Streamlit dashboard for alpha-forecast.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from alpha_forecast.config import blockchair_api_key, holdings_path
from alpha_forecast.data.crypto import chain_for_symbol, is_crypto_symbol, network_stats
from alpha_forecast.data.loader import load_prices
from alpha_forecast.data.quotes import describe_sources, latest_quotes
from alpha_forecast.evaluation import log_equity
from alpha_forecast.models import MODEL_REGISTRY
from alpha_forecast.pipeline import Comparison, compare_models
from alpha_forecast.portfolio.backtest import SCHEMES, portfolio_backtest
from alpha_forecast.portfolio.holdings import (
    aggregate_by_symbol,
    load_holdings,
    quote_symbol,
    value_holdings,
)
from alpha_forecast.regime import rolling_regime_probabilities
from alpha_forecast.risk import kupiec_test, rolling_var_forecast

st.set_page_config(page_title="alpha-forecast", layout="wide")

PCT = "{:.2%}"
BOARD_FORMAT = {
    "rmse": "{:.5f}",
    "r2": PCT,
    "dir_acc": PCT,
    "sharpe": "{:.2f}",
    "sharpe_lo": "{:.2f}",
    "sharpe_hi": "{:.2f}",
    "ann_return": PCT,
    "max_dd": PCT,
    "dm_pvalue": "{:.3f}",
}


# --- cached computations ------------------------------------------------------

@st.cache_data(show_spinner=False)
def run_single(
    symbol: str, start: str, end: str, horizon: int, models: tuple[str, ...],
    initial_train: int, step: int, cost_bps: float, factors: bool, macro: bool,
    onchain: bool = False,
) -> Comparison:
    return compare_models(
        symbol, start=start, end=end, horizon=horizon, models=list(models),
        initial_train=initial_train, step=step, cost_bps=cost_bps,
        use_factors=factors, use_macro=macro, use_onchain=onchain,
    )


@st.cache_data(ttl=120, show_spinner=False)
def run_network(chain: str) -> dict | None:
    return network_stats(chain)


@st.cache_data(show_spinner=False)
def run_risk(symbol: str, start: str, end: str, alpha: float, initial_train: int) -> dict:
    prices = load_prices(symbol, start, end)
    rets = np.log(prices["close"]).diff().dropna()
    out = {"returns": rets}
    for method in ("garch", "historical"):
        var = rolling_var_forecast(rets, alpha=alpha, method=method, initial_train=initial_train)
        out[method] = var
        out[f"{method}_kupiec"] = kupiec_test(rets, var["var"], alpha)
    return out


@st.cache_data(show_spinner=False)
def run_regimes(symbol: str, start: str, end: str, n_states: int, initial_train: int):
    prices = load_prices(symbol, start, end)
    rets = np.log(prices["close"]).diff().dropna()
    probs = rolling_regime_probabilities(rets, n_states=n_states, initial_train=initial_train)
    return prices["close"], rets, probs


@st.cache_data(ttl=60, show_spinner=False)
def cached_quotes(symbols_key: tuple[str, ...]) -> pd.DataFrame:
    return latest_quotes(symbols_key)


@st.cache_data(show_spinner=False)
def run_portfolio(
    symbols: tuple[str, ...], scheme: str, model: str, start: str, end: str,
    horizon: int, initial_train: int, rebalance: int, cost_bps: float,
) -> dict:
    return portfolio_backtest(
        list(symbols), scheme=scheme, model=model, start=start, end=end, horizon=horizon,
        initial_train=initial_train, rebalance=rebalance, cost_bps=cost_bps,
    )


# --- chart helpers ------------------------------------------------------------

def equity_and_drawdown(curves: dict[str, pd.Series], title: str) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], vertical_spacing=0.04
    )
    palette = qualitative.Plotly
    for i, (name, eq) in enumerate(curves.items()):
        dash = "dot" if name == "buy & hold" else None
        color = palette[i % len(palette)]
        fig.add_trace(
            go.Scatter(x=eq.index, y=eq, name=name, legendgroup=name,
                       line={"dash": dash, "color": color}),
            row=1, col=1,
        )
        dd = eq / eq.cummax() - 1
        fig.add_trace(
            go.Scatter(x=dd.index, y=dd, name=name, showlegend=False, legendgroup=name,
                       line={"dash": dash, "width": 1, "color": color}),
            row=2, col=1,
        )
    fig.update_yaxes(title_text="Growth of $1", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(
        title=title, height=560, hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    return fig


def sharpe_ci_chart(board: pd.DataFrame) -> go.Figure:
    b = board.sort_values("sharpe")
    fig = go.Figure(
        go.Bar(
            x=b["sharpe"], y=b["model"], orientation="h",
            error_x={
                "type": "data", "symmetric": False,
                "array": b["sharpe_hi"] - b["sharpe"],
                "arrayminus": b["sharpe"] - b["sharpe_lo"],
            },
        )
    )
    fig.add_vline(x=0, line_dash="dash", opacity=0.5)
    fig.update_layout(
        title="Sharpe ratio with 95% block-bootstrap interval",
        xaxis_title="Sharpe (net of costs)", height=320,
    )
    return fig


def verdict(board: pd.DataFrame) -> str:
    best = board.iloc[0]
    msg = f"**Best by Sharpe: `{best['model']}`** ({best['sharpe']:.2f})."
    if best["model"] == "naive":
        return msg + " The naive baseline wins: no model adds value here."
    if best["sharpe_lo"] <= 0:
        msg += " Its Sharpe interval includes zero, so the edge is not statistically reliable."
    if not best["dm_pvalue"] < 0.05:
        msg += (
            f" Its forecast errors are not significantly lower than naive "
            f"(DM p = {best['dm_pvalue']:.2f})."
        )
    if best["sharpe_lo"] > 0 and best["dm_pvalue"] < 0.05:
        msg += " Both tests pass: evidence of genuine out-of-sample skill."
    return msg


# --- sidebar ------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")
    mode = st.radio("Mode", ["Single asset", "Portfolio", "Holdings"], horizontal=True)
    today = date.today()
    start_d = st.date_input("Start", today - timedelta(days=365 * 4))
    end_d = st.date_input("End", today)
    horizon = st.slider("Forecast horizon (days)", 1, 20, 1)
    cost_bps = st.slider("Transaction cost (bps per unit traded)", 0.0, 20.0, 1.0, 0.5)
    initial_train = st.slider("Initial training window (days)", 126, 756, 252, 21)

    onchain = False
    if mode == "Single asset":
        symbol = st.text_input("Ticker", value="AAPL",
                               help="Stocks/ETFs (AAPL) or crypto pairs (BTC-USD)").upper().strip()
        step = st.slider("Retrain step (days)", 5, 63, 21)
        models = st.multiselect("Models", list(MODEL_REGISTRY), default=list(MODEL_REGISTRY))
        factors = st.toggle("Fama-French factor features", value=False)
        macro = st.toggle("FRED macro features (needs FRED_API_KEY)", value=False)
        if is_crypto_symbol(symbol):
            onchain = st.toggle(
                "Blockchair on-chain features (needs BLOCKCHAIR_API_KEY)",
                value=blockchair_api_key() is not None,
                disabled=chain_for_symbol(symbol) is None,
                help="Daily transactions, fees, transferred value and difficulty, lagged one day.",
            )
        var_alpha = st.select_slider("VaR tail probability", [0.01, 0.025, 0.05], value=0.01)
        n_states = st.slider("HMM regimes", 2, 3, 2)
    else:
        if mode == "Portfolio":
            tickers = st.text_input("Tickers (comma separated)", value="AAPL, MSFT, GOOG, AMZN")
            symbols = tuple(t.strip().upper() for t in tickers.split(",") if t.strip())
        else:
            holdings = load_holdings()
            classes = sorted(holdings["asset_class"].unique()) if not holdings.empty else []
            picked = st.multiselect("Asset classes", classes, default=classes)
            holdings = holdings[holdings["asset_class"].isin(picked)].reset_index(drop=True)
            symbols = ()
        scheme = st.selectbox("Allocation scheme", list(SCHEMES), index=list(SCHEMES).index(
            "black_litterman"))
        port_model = st.selectbox("Forecast model", list(MODEL_REGISTRY), index=list(
            MODEL_REGISTRY).index("gbm"))
        rebalance = st.slider("Rebalance every (days)", 1, 63, 21)

    if st.button("Run backtest", type="primary"):
        st.session_state["ran"] = mode

st.title("alpha-forecast")
st.caption(
    "Multi-model equity return forecasting with walk-forward backtesting, "
    "significance tests, risk and regime analysis."
)

start, end = start_d.isoformat(), end_d.isoformat()

# --- holdings (shown before any backtest runs) ----------------------------------

if mode == "Holdings":
    if holdings.empty:
        st.info(
            f"No holdings file at `{holdings_path()}`. Ask the agent connected to your "
            "Robinhood account to export positions there (columns: account, asset_class, "
            "symbol, quantity, avg_cost), or set `ALPHA_FORECAST_HOLDINGS` to another path."
        )
        st.stop()
    wanted = sorted({
        quote_symbol(s, a)
        for s, a in zip(holdings["symbol"], holdings["asset_class"], strict=True)
    })
    quotes = cached_quotes(tuple(wanted))
    valued = value_holdings(holdings, quotes)
    total_value = float(valued["market_value"].sum(skipna=True))
    with_cost = valued.dropna(subset=["cost_basis", "market_value"])
    total_cost = float(with_cost["cost_basis"].sum())
    pnl_value = float(with_cost["market_value"].sum())
    priced = valued["price"].notna()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market value", f"${total_value:,.2f}")
    c2.metric("Open P&L", f"${pnl_value - total_cost:+,.2f}",
              f"{(pnl_value / total_cost - 1):+.1%}" if total_cost > 0 else None)
    c3.metric("Positions", f"{len(valued)}", f"{int(priced.sum())} priced")
    c4.metric("Accounts", f"{valued['account'].nunique()}")
    if not valued["avg_cost"].notna().all():
        st.caption("P&L excludes positions whose cost basis the broker does not report.")
    st.dataframe(
        valued.drop(columns=["quote_symbol"]).style.format({
            "quantity": "{:,.6f}", "avg_cost": "{:,.2f}", "price": "{:,.2f}",
            "market_value": "${:,.2f}", "cost_basis": "${:,.2f}", "pnl": "${:+,.2f}",
            "pnl_pct": "{:+.1%}", "weight": PCT,
        }, na_rep="-"),
        hide_index=True,
    )
    by_symbol = aggregate_by_symbol(valued)
    if not by_symbol.empty and by_symbol["market_value"].notna().any():
        pie = go.Figure(go.Pie(labels=by_symbol["quote_symbol"], values=by_symbol["market_value"],
                               hole=0.45, textinfo="label+percent"))
        pie.update_layout(title="Allocation by symbol (all accounts)", height=420,
                          showlegend=False)
        st.plotly_chart(pie)
    universe = tuple(by_symbol["quote_symbol"])
    st.caption(f"Holdings read from `{holdings_path()}` · prices: "
               f"{describe_sources(quotes) if not quotes.empty else 'unavailable'}.")
    symbols = universe

quote_symbols = (symbol,) if mode == "Single asset" else symbols
if mode != "Holdings":
    quotes = cached_quotes(quote_symbols)
    if not quotes.empty:
        bits = []
        for row in quotes.itertuples(index=False):
            move = "" if pd.isna(row.change) else f" ({row.change:+.2%})"
            bits.append(f"**{row.symbol}** {row.price:,.2f}{move}")
        st.caption("Latest · " + " · ".join(bits) + " · " + describe_sources(quotes))

ran = st.session_state.get("ran") == mode

if not ran:
    if mode == "Holdings":
        st.info("Click **Run backtest** to backtest an allocation over these holdings.")
    else:
        st.info("Configure parameters in the sidebar and click **Run backtest**.")
        st.markdown(
            "Every forecast is strictly out-of-sample (expanding-window walk-forward, with "
            "overlapping targets purged for multi-day horizons). Prices come from OpenBB or "
            "Yahoo Finance; offline runs use a reproducible synthetic series. Crypto pairs "
            "such as `BTC-USD` are supported, with on-chain features from Blockchair."
        )
    st.stop()

# --- single asset ---------------------------------------------------------------

if mode == "Single asset":
    if not models:
        st.warning("Select at least one model.")
        st.stop()

    with st.spinner("Running walk-forward backtests..."):
        comp = run_single(symbol, start, end, horizon, tuple(models), initial_train, step,
                          cost_bps, factors, macro, onchain)

    if comp.prices.attrs.get("source") == "synthetic":
        st.warning(f"No market data provider reachable: showing **synthetic** data for {symbol}.")
    crypto = is_crypto_symbol(symbol)
    chain = chain_for_symbol(symbol) if crypto else None
    if onchain and not any(c.startswith("ext_onchain_") for c in comp.extra_features):
        st.warning("On-chain features were requested but not loaded: check BLOCKCHAIR_API_KEY "
                   "and that the chain is supported.")

    board = comp.board
    tab_names = ["Leaderboard", "Equity & drawdown", "Forecast diagnostics", "Risk (VaR)",
                 "Regimes"]
    if chain is not None:
        tab_names.append("Network")
    tabs = st.tabs(tab_names)
    tab_board, tab_equity, tab_diag, tab_risk, tab_regime = tabs[:5]

    with tab_board:
        best = board.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", best["model"])
        c2.metric("Sharpe", f"{best['sharpe']:.2f}",
                  help=f"Annualised over {comp.days_per_year:.0f} days, net of costs, "
                       "non-overlapping holding periods")
        c3.metric("Directional accuracy", f"{best['dir_acc']:.1%}")
        c4.metric("DM p-value vs naive", "n/a" if pd.isna(best["dm_pvalue"])
                  else f"{best['dm_pvalue']:.3f}")
        st.markdown(verdict(board))
        st.dataframe(board.style.format(BOARD_FORMAT, na_rep="-"), hide_index=True)
        st.plotly_chart(sharpe_ci_chart(board))
        with st.expander("How to read this"):
            st.markdown(
                "- **dir_acc**: share of correct direction calls (a zero forecast counts as long).\n"
                "- **sharpe / sharpe_lo / sharpe_hi**: Sharpe of trading the forecast sign, with a "
                "95% circular block-bootstrap interval.\n"
                "- **dm_pvalue**: one-sided Diebold-Mariano test that the model's squared error is "
                "lower than the naive (zero-return) forecast.\n"
                "- For horizons above one day, the strategy trades every *h*-th forecast and holds "
                "for *h* days, so returns never overlap."
            )

    with tab_equity:
        curves = {name: log_equity(net) for name, net in comp.strategy.items()}
        any_net = next(iter(comp.strategy.values()))
        first = next(iter(comp.results.values()))
        curves["buy & hold"] = log_equity(first.actuals.loc[any_net.index])
        st.plotly_chart(equity_and_drawdown(curves, f"Out-of-sample equity, {cost_bps} bps costs"))

    with tab_diag:
        pick = st.selectbox("Model", list(comp.results), key="diag_model")
        res = comp.results[pick]
        left, right = st.columns(2)
        scatter = go.Figure(go.Scattergl(
            x=res.actuals, y=res.predictions, mode="markers", marker={"size": 4, "opacity": 0.5}
        ))
        lim = float(max(res.actuals.abs().max(), res.predictions.abs().max()))
        scatter.add_shape(type="line", x0=-lim, y0=-lim, x1=lim, y1=lim, line={"dash": "dash"})
        scatter.update_layout(title="Predicted vs actual forward return",
                              xaxis_title="Actual", yaxis_title="Predicted", height=420)
        left.plotly_chart(scatter)

        hits = (np.sign(res.predictions).replace(0, 1.0) == np.sign(res.actuals)).astype(float)
        rolling = hits.rolling(63).mean()
        hit_fig = go.Figure(go.Scatter(x=rolling.index, y=rolling, name="63-day hit rate"))
        hit_fig.add_hline(y=0.5, line_dash="dash", opacity=0.6)
        hit_fig.update_layout(title="Rolling directional accuracy (63 days)",
                              yaxis_tickformat=".0%", height=420)
        right.plotly_chart(hit_fig)

    with tab_risk:
        with st.spinner("Forecasting volatility and VaR..."):
            risk = run_risk(symbol, start, end, var_alpha, initial_train)
        rets = risk["returns"]
        fig = go.Figure()
        oos = rets.loc[risk["garch"].index]
        fig.add_trace(go.Scatter(x=oos.index, y=oos, mode="lines", name="daily return",
                                 line={"width": 1}, opacity=0.6))
        for method in ("garch", "historical"):
            var = risk[method]["var"]
            fig.add_trace(go.Scatter(x=var.index, y=-var, name=f"-VaR ({method})"))
        breaches = oos[oos < -risk["garch"]["var"].reindex(oos.index)]
        fig.add_trace(go.Scatter(x=breaches.index, y=breaches, mode="markers",
                                 name="GARCH breach", marker={"color": "red", "size": 7}))
        fig.update_layout(title=f"One-day {1 - var_alpha:.1%} VaR, out-of-sample",
                          yaxis_tickformat=".1%", height=480, hovermode="x unified")
        st.plotly_chart(fig)

        kupiec = pd.DataFrame({m: risk[f"{m}_kupiec"] for m in ("garch", "historical")}).T
        st.markdown(
            "**Kupiec coverage test**: a p-value below 0.05 means the breach rate is "
            "inconsistent with the target, i.e. the model misstates risk."
        )
        st.dataframe(
            kupiec[["n", "breaches", "breach_rate", "expected_rate", "p_value"]].style.format(
                {"n": "{:.0f}", "breaches": "{:.0f}", "breach_rate": PCT,
                 "expected_rate": PCT, "p_value": "{:.3f}"}
            )
        )

    with tab_regime:
        with st.spinner("Fitting walk-forward HMM..."):
            close, rets, probs = run_regimes(symbol, start, end, n_states, initial_train)
        turbulent = probs.columns[-1]
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4],
                            vertical_spacing=0.05)
        fig.add_trace(go.Scatter(x=close.index, y=close, name="close"), row=1, col=1)
        fig.add_trace(go.Scatter(x=probs.index, y=probs[turbulent], fill="tozeroy",
                                 name="P(most volatile regime)"), row=2, col=1)
        fig.update_yaxes(range=[0, 1], tickformat=".0%", row=2, col=1)
        fig.update_layout(title="Filtered (causal) regime probability", height=520)
        st.plotly_chart(fig)

        state = probs.idxmax(axis=1)
        r = rets.reindex(state.index)
        stats = pd.DataFrame({
            "share_of_days": state.value_counts(normalize=True),
            "ann_return": r.groupby(state).mean() * comp.days_per_year,
            "ann_vol": r.groupby(state).std() * np.sqrt(comp.days_per_year),
        }).sort_index()
        st.dataframe(stats.style.format(PCT))
        st.caption(
            "Regimes are re-estimated every 21 days on data observed so far and filtered forward, "
            "so these probabilities were available in real time. The `regime` model in the "
            "leaderboard uses the same machinery to switch between per-regime forecasts."
        )

    if chain is not None:
        with tabs[5]:
            stats_now = run_network(chain)
            if stats_now is None:
                st.info("Network snapshot unavailable: set BLOCKCHAIR_API_KEY to enable "
                        "Blockchair on-chain data.")
            else:
                def _num(key: str) -> float | None:
                    v = stats_now.get(key)
                    try:
                        return float(v) if v is not None else None
                    except (TypeError, ValueError):
                        return None

                def _fmt(v: float | None, spec: str, scale: float = 1.0) -> str:
                    return "n/a" if v is None else format(v * scale, spec)

                price = _num("market_price_usd")
                chg = _num("market_price_usd_change_24h_percentage")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Price (USD)", _fmt(price, ",.0f"),
                          None if chg is None else f"{chg:+.2f}% 24h")
                c2.metric("Transactions 24h", _fmt(_num("transactions_24h"), ",.0f"))
                c3.metric("Mempool txs", _fmt(_num("mempool_transactions"), ",.0f"))
                c4.metric("Blocks 24h", _fmt(_num("blocks_24h"), ",.0f"))
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("Market cap", "n/a" if _num("market_cap_usd") is None
                          else f"${_num('market_cap_usd') / 1e9:,.1f}B")
                c6.metric("Hashrate 24h", "n/a" if _num("hashrate_24h") is None
                          else f"{_num('hashrate_24h') / 1e18:,.0f} EH/s")
                c7.metric("Avg fee 24h", "n/a" if _num("average_transaction_fee_usd_24h") is None
                          else f"${_num('average_transaction_fee_usd_24h'):,.2f}")
                c8.metric("Dominance", _fmt(_num("market_dominance_percentage"), ".1f") + "%")
                st.caption(
                    f"Blockchair `/{chain}/stats`, refreshed every 2 minutes. Best block "
                    f"{stats_now.get('best_block_height', 'n/a')} at "
                    f"{stats_now.get('best_block_time', 'n/a')} UTC."
                )
                if onchain and comp.extra_features:
                    oc = [c.removeprefix("ext_onchain_") for c in comp.extra_features
                          if c.startswith("ext_onchain_")]
                    if oc:
                        st.markdown("On-chain features used by the models (lagged one day): "
                                    + ", ".join(f"`{c}`" for c in oc))

# --- portfolio / holdings ---------------------------------------------------------

else:
    if len(symbols) < 2:
        st.warning("Enter at least two tickers." if mode == "Portfolio"
                   else "At least two distinct holdings are needed for a portfolio backtest.")
        st.stop()
    with st.spinner("Forecasting each asset and backtesting the portfolio..."):
        out = run_portfolio(symbols, scheme, port_model, start, end, horizon, initial_train,
                            rebalance, cost_bps)
        bench = run_portfolio(symbols, "equal", port_model, start, end, horizon, initial_train,
                              rebalance, cost_bps)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Sharpe", f"{out['sharpe']:.2f}", f"{out['sharpe'] - bench['sharpe']:+.2f} vs 1/N")
    c2.metric("Ann. return", f"{out['ann_return']:.1%}")
    c3.metric("Ann. vol", f"{out['ann_vol']:.1%}")
    c4.metric("Max drawdown", f"{out['max_drawdown']:.1%}")
    c5.metric("Ann. turnover", f"{out['ann_turnover']:.2f}x")

    curves = {scheme: out["equity"]}
    if scheme != "equal":
        curves["equal weight (1/N)"] = bench["equity"]
    st.plotly_chart(equity_and_drawdown(curves, f"Portfolio equity, {cost_bps} bps costs"))

    weights = pd.DataFrame({dt: w for dt, w in out["weights_history"]}).T
    wfig = go.Figure()
    for col in weights.columns:
        wfig.add_trace(go.Scatter(x=weights.index, y=weights[col], name=col, stackgroup="w",
                                  line={"shape": "hv"}))
    wfig.update_layout(title="Target weights at each rebalance", yaxis_tickformat=".0%",
                       yaxis_range=[0, 1], height=380)
    st.plotly_chart(wfig)
