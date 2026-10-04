# alpha-forecast 📈

![CI](https://github.com/phoyo008/alpha-forecast-/actions/workflows/ci.yml/badge.svg)

**A multi-model equity return forecasting & walk-forward backtesting engine, built on [OpenBB](https://github.com/OpenBB-finance/OpenBB).**

OpenBB gives you clean "connect once" access to market data but deliberately
**removed its forecasting module**. `alpha-forecast` fills that gap: it takes
OpenBB price data, engineers causal features, and runs classical, baseline, and
machine-learning models through a rigorous **walk-forward backtest** — then tells
you honestly whether any of them actually beat a naive baseline.

> Built as a demonstration of a complete, production-minded quant research
> workflow: **data → features → models → out-of-sample evaluation → serving.**

---

## Why this project is different

Most "stock prediction" projects fail the smell test for a quant/DS role because
they do one or more of these: a single train/test split (look-ahead leakage), no
transaction costs, no baseline to beat, and they report price-level R² (which is
misleadingly high because prices trend). `alpha-forecast` avoids all of them:

| Common mistake | What this project does instead |
|---|---|
| One train/test split | **Expanding-window walk-forward** — every prediction is out-of-sample |
| Overlapping multi-day labels | Training rows whose targets reach into the test window are **purged** |
| Predicting price level | Predicts **forward returns** (the hard, honest target) |
| No baseline | Ships **naive & drift baselines**; "alpha" is only real if it beats them |
| Ranking by luck | **Diebold-Mariano** test vs naive + **block-bootstrap Sharpe intervals** |
| Ignoring costs | Charges **transaction costs per unit of notional traded** |
| Inflated multi-day Sharpe | Trades **non-overlapping** holding periods and annualises by `252 / horizon` |
| Look-ahead features | All features (and HMM regime probabilities) are **strictly causal** (unit-tested) |

---

## Architecture

```
alpha_forecast/
├── data/          # OpenBB → yfinance → synthetic fallback, live quotes, Blockchair on-chain
├── features/      # causal engineering: price + Fama-French + FRED macro + on-chain
├── models/        # naive, drift, ETS, gradient boosting, neural, regime-switching, GARCH vol
├── backtest/      # expanding-window walk-forward engine with label purging
├── evaluation/    # error + economic metrics, significance tests, matplotlib plots
├── risk/          # VaR / Expected Shortfall + Kupiec backtest
├── regime/        # NumPy Gaussian HMM for market regime detection
├── portfolio/     # mean-variance / risk-parity / Black-Litterman + multi-asset backtest + holdings
├── options/       # Black-Scholes pricing, Greeks, implied-volatility solver
├── config.py      # environment switches (offline mode, cache directory)
├── pipeline.py    # end-to-end orchestration → model leaderboard
└── cli.py         # command-line entry point
dashboard/app.py   # Streamlit UI: leaderboard, equity, diagnostics, VaR, regimes, network, portfolio, holdings
tests/             # hermetic offline tests incl. look-ahead and timing regressions
.github/workflows/ # CI: ruff + mypy, tests with coverage on Python 3.10–3.13
```

Every model implements the same `Forecaster` interface (`fit`/`predict`), so the
backtester compares them on a level playing field.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[full,dashboard]"      # core + strong models + dashboard

# Run a comparison from the CLI
python -m alpha_forecast.cli --symbol AAPL --horizon 1 -v

# Launch the dashboard
streamlit run dashboard/app.py
```

Example CLI output:

```
=== alpha-forecast leaderboard: AAPL (horizon=1d) ===

 model    rmse     r2 dir_acc  sharpe   sharpe_95ci ann_return  max_dd  dm_pvalue  n_oos
   gbm 0.01421   1.2%   52.4%    0.71 [-0.32, 1.70]     11.3%  -18.4%      0.214    501
 drift 0.01440  -0.1%   51.1%    0.44 [-0.58, 1.49]      6.8%  -21.0%      0.611    501
 naive 0.01440   0.0%   51.1%    0.44 [-0.58, 1.49]      6.8%  -21.0%        NaN    501
   ets 0.01441  -0.2%   50.3%    0.21 [-0.83, 1.22]      3.1%  -24.7%      0.702    501
```

> Numbers above are illustrative. Real results depend on symbol/period — and
> frequently the baselines win, which is the honest, expected outcome for
> daily single-name forecasting.

### Statistical significance

Every leaderboard row carries two guards against reading noise as skill:

- **`sharpe_95ci`** — a 95% circular block-bootstrap interval for the Sharpe
  ratio (blocks of `n^(1/3)` periods preserve short-range autocorrelation).
  An interval that spans zero means the strategy might simply be lucky.
- **`dm_pvalue`** — one-sided Diebold-Mariano test (Newey-West variance,
  Harvey-Leybourne-Newbold small-sample correction) that the model's squared
  forecast error is lower than the naive zero-return forecast.

```python
from alpha_forecast.evaluation import bootstrap_sharpe_ci, diebold_mariano
from alpha_forecast.pipeline import compare_models

comp = compare_models("AAPL", horizon=5)      # board + per-model results
gbm = comp.results["gbm"]
diebold_mariano(gbm.actuals, gbm.predictions, comp.results["naive"].predictions, horizon=5)
bootstrap_sharpe_ci(comp.strategy["gbm"], periods_per_year=252 / 5)
```

### Multi-day horizons

With `--horizon h > 1`, targets overlap. Two consequences are handled explicitly:
the walk-forward engine **purges** the last `h - 1` training rows of each fold
(their labels are not yet known at the forecast date), and the strategy trades
only every `h`-th forecast, holding for `h` days, so each return is independent
and the Sharpe ratio is annualised with `252 / h` periods per year.

---

## Data sources

The loader tries providers in order and **always produces data**:

1. **OpenBB Platform** — `obb.equity.price.historical` (install `.[openbb]`)
2. **yfinance** — fallback (install `.[full]`)
3. **Synthetic GBM** — reproducible offline fallback so tests/CI always run

The provider used is recorded in `prices.attrs["source"]`, and the dashboard
warns when it is showing synthetic data.

| Environment variable | Effect |
|---|---|
| `ALPHA_FORECAST_OFFLINE=1` | Skip all network providers (same as the CLI's `--offline`) |
| `ALPHA_FORECAST_CACHE_DIR` | Where provider downloads are cached (default `~/.cache/alpha_forecast`; empty string disables, as does `--no-cache`) |
| `BLOCKCHAIR_API_KEY` | Enables Blockchair on-chain data for crypto symbols (see below) |
| `ALPHA_FORECAST_HOLDINGS` | Path to a brokerage holdings CSV for the dashboard's **Holdings** mode (default `~/.config/alpha_forecast/holdings.csv`) |
| `ALPHA_FORECAST_QUOTES` | Path to a broker quote snapshot CSV (default `~/.config/alpha_forecast/quotes.csv`; empty string ignores it) |

Historical ranges are cached indefinitely; ranges ending today are refreshed
after 12 hours. The dashboard also shows the latest price for the selected
tickers: from the broker quote snapshot when it covers the symbol and is less
than 24 hours old, otherwise the delayed Yahoo Finance close. The caption
names the source and timestamp of every price shown.

### Factor, macro & on-chain features (optional)

| Feature set | Flag | API key? | Source |
|---|---|---|---|
| **Fama-French factors** (Mkt-RF, SMB, HML, RF) | `--factors` | No | OpenBB → Ken French library → synthetic |
| **FRED macro** (yield slope, breakeven, VIX) | `--macro` | Free `FRED_API_KEY` | FRED via `fredapi` |
| **On-chain activity** (tx growth, fee/tx, transferred value z-score, difficulty change, block rate) | `--onchain` | `BLOCKCHAIR_API_KEY` | [Blockchair](https://blockchair.com/api/docs) daily block aggregates |

```bash
# Add Fama-French factor exposures (free, no key)
python -m alpha_forecast.cli --symbol AAPL --factors -v

# Add macro features (export your free FRED key first)
export FRED_API_KEY=your_key_here
python -m alpha_forecast.cli --symbol AAPL --factors --macro -v

# Crypto: BTC-USD prices (yfinance) + Bitcoin on-chain features (Blockchair)
export BLOCKCHAIR_API_KEY=your_key_here
python -m alpha_forecast.cli --symbol BTC-USD --onchain -v
```

### Crypto

Any `XXX-USD` pair works as a symbol. Crypto trades every calendar day, so
Sharpe ratios and annualised returns use **365** periods per year instead of
252. On-chain features are available for Bitcoin, Bitcoin Cash, Litecoin,
Dogecoin, Dash, Zcash and Ethereum (Ethereum has no transferred-value or
difficulty features). In the dashboard a crypto ticker adds a **Network** tab
with the live chain snapshot (price, 24h transactions, mempool, hashrate, fees).

### Brokerage holdings

Robinhood exposes account data to AI agents through its official
[Trading MCP server](https://robinhood.com/us/en/support/articles/agentic-trading-overview/)
(`.cursor/mcp.json` registers it; auth is a browser OAuth login, nothing is
stored in the repo). The agent exports positions to a small CSV outside the
repository, and the dashboard's **Holdings** mode values them at the latest
quotes, shows P&L and allocation, and can backtest an allocation scheme over
that universe.

```csv
account,asset_class,symbol,quantity,avg_cost
6050,equity,MSFT,0.419566,382.25
6050,crypto,BTC,0.00196765,70548.44
```

The agent can also export real-time Robinhood quotes to the quote snapshot,
which the dashboard prefers over Yahoo Finance:

```csv
symbol,price,previous_close,as_of,source
MSFT,517.19,512.80,2026-10-02 19:59 ET,robinhood
BTC-USD,85340.50,84831.65,2026-10-04 12:45 ET,robinhood
```

All external features are **forward-filled and lagged one day** before joining,
so they can only ever use information observable at the close of the trading
day — preserving the no-look-ahead guarantee.

### Neural forecasting

A `neural` model is in the registry. It uses a **PyTorch LSTM** over a lookback
window when `torch` is installed, falling back to a scikit-learn **MLP**, then a
ridge linear model — so it runs everywhere while using a sequence model when
available. Inputs are z-scored with **train-fold-only** statistics (no leakage).

```bash
pip install -e ".[full,neural]"      # adds torch
python -m alpha_forecast.cli --symbol AAPL --models naive gbm neural -v
```

### Portfolio construction

Turn per-asset forecasts into an actual allocation and backtest it
cross-sectionally. Four schemes, implemented in pure NumPy (no cvxpy):

| Scheme | Idea |
|---|---|
| `equal` | 1/N baseline |
| `mean_variance` | maximise return per unit risk (long-only, projected gradient) |
| `risk_parity` | each asset contributes equal risk |
| `black_litterman` | blend an equilibrium prior with the forecasts as views, then mean-variance |

Weights chosen at the close of day `t` earn the return from `t` to `t+1`;
holdings drift between rebalances and costs are charged on rebalance turnover.

```bash
# Multi-asset portfolio backtest
python -m alpha_forecast.cli --portfolio AAPL MSFT GOOG AMZN \
    --scheme black_litterman --models gbm --cost-bps 2 -v
```

```python
from alpha_forecast.portfolio import black_litterman, mean_variance_weights, risk_parity_weights
# mu: expected returns (Series), cov: covariance (DataFrame)
w = mean_variance_weights(mu, cov, risk_aversion=10)
w_rp = risk_parity_weights(cov)

# Absolute views (asset -> return) or relative views via P / Q
mu_bl, cov_bl = black_litterman(cov, views=mu, view_confidence=0.5)
w_bl = mean_variance_weights(mu_bl, cov_bl, risk_aversion=10)
```

### Volatility forecasting

`alpha_forecast.models.volatility` provides a **GARCH(1,1)** forecaster (via the
`arch` package, with an EWMA/RiskMetrics fallback). Volatility is far more
forecastable than returns, making this useful for risk and options work:

```python
from alpha_forecast.data import load_prices
from alpha_forecast.models.volatility import rolling_vol_forecast
import numpy as np

prices = load_prices("AAPL")
rets = np.log(prices["close"]).diff().dropna()
vol = rolling_vol_forecast(rets)   # one-step-ahead conditional vol, out-of-sample
```

### Value-at-Risk

`alpha_forecast.risk` turns the volatility forecasts into one-day VaR and
Expected Shortfall (parametric from GARCH, or historical), and checks them with
the **Kupiec** proportion-of-failures test:

```python
from alpha_forecast.risk import kupiec_test, rolling_var_forecast

var = rolling_var_forecast(rets, alpha=0.01, method="garch")   # columns: var, es
kupiec_test(rets, var["var"], alpha=0.01)   # p_value < 0.05 => risk is misstated
```

### Regime detection

`alpha_forecast.regime.GaussianHMM` is a NumPy hidden Markov model (scaled
Baum-Welch, diagonal covariances). `filter()` gives causal state probabilities
safe for trading; `smooth()` is for in-sample analysis only. States are ordered
so `regime_0` is always the calmest.

```python
from alpha_forecast.regime import rolling_regime_probabilities
probs = rolling_regime_probabilities(rets, n_states=2)   # refit every 21 days, no look-ahead
```

The `regime` model in the registry builds on this: it fits one ridge regression
per regime and blends their forecasts by the filtered regime probabilities.

### Options pricing & Greeks

`alpha_forecast.options` is a self-contained Black-Scholes-Merton toolkit for
European options — pricing, the full set of Greeks, and a robust implied-volatility
solver (Newton-Raphson with a bracketed bisection fallback). It pairs naturally
with the GARCH volatility forecaster above.

```python
from alpha_forecast.options import call_price, greeks, implied_volatility

# Price an at-the-money 1-year call: S=100, K=100, r=5%, sigma=20%
price = call_price(S=100, K=100, t=1.0, r=0.05, sigma=0.20)   # ≈ 10.45

# Full Greeks in one call
g = greeks(S=100, K=100, t=1.0, r=0.05, sigma=0.20, option_type="call")
# {'delta': ..., 'gamma': ..., 'vega': ..., 'theta': ..., 'rho': ...}

# Recover implied vol from a market price
iv = implied_volatility(price=10.45, S=100, K=100, t=1.0, r=0.05)  # ≈ 0.20
```

All functions accept scalars or NumPy arrays and support a continuous dividend
yield `q`. Pricing/Greeks are validated against put-call parity and finite-
difference checks in `tests/test_options.py`.

---

## Testing

```bash
pip install -e ".[full,dashboard,dev]"
ruff check .
mypy
pytest -q --cov
```

Tests are hermetic: `tests/conftest.py` forces offline mode and disables the
cache, so they always run on synthetic data. They include explicit
**no-look-ahead** checks (features, label purging, causal HMM filtering),
portfolio timing and cost regressions, and a headless Streamlit smoke test.

---

## Roadmap / extension ideas

- Deep-learning sequence models (N-BEATS / TFT via `darts` or `neuralforecast`)
- Christoffersen independence test for VaR breach clustering
- Multiple-testing control across models (White's Reality Check / SPA test)
- Cross-sectional features (sector-relative momentum) for the portfolio model
- Macro features from ECB via OpenBB

---

## License

MIT — see `LICENSE`.
