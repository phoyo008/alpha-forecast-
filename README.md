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
misleadingly high because prices trend). `alpha-forecast` avoids all four:

| Common mistake | What this project does instead |
|---|---|
| One train/test split | **Expanding-window walk-forward** — every prediction is out-of-sample |
| Predicting price level | Predicts **forward returns** (the hard, honest target) |
| No baseline | Ships **naive & drift baselines**; "alpha" is only real if it beats them |
| Ignoring costs | Applies **per-trade transaction costs** to the strategy P&L |
| Look-ahead features | All features are **strictly causal** (unit-tested) |

---

## Architecture

```
alpha_forecast/
├── data/          # OpenBB → yfinance → synthetic fallback (always runnable)
├── features/      # causal engineering: price features + Fama-French factors + FRED macro
├── models/        # Forecaster interface + naive, drift, ETS, gradient boosting, GARCH vol
├── backtest/      # expanding-window walk-forward engine
├── evaluation/    # forecast-error + economic metrics + matplotlib plots
├── pipeline.py    # end-to-end orchestration → model leaderboard
└── cli.py         # command-line entry point
dashboard/app.py   # Streamlit UI with equity curves + leaderboard
tests/             # offline tests (synthetic data) incl. look-ahead checks
.github/workflows/ # CI: lint + test on Python 3.10–3.12
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

 model     rmse      r2  dir_acc  sharpe  ann_return   max_dd  n_oos
   gbm  0.01421   1.2%   52.4%    0.71      11.3%    -18.4%    501
 drift  0.01440  -0.1%   51.1%    0.44       6.8%    -21.0%    501
   ets  0.01441  -0.2%   50.3%    0.21       3.1%    -24.7%    501
 naive  0.01440   0.0%   50.0%    0.09       1.4%    -22.9%    501
```

> Numbers above are illustrative. Real results depend on symbol/period — and
> frequently the baselines win, which is the honest, expected outcome for
> daily single-name forecasting.

---

## Data sources

The loader tries providers in order and **always produces data**:

1. **OpenBB Platform** — `obb.equity.price.historical` (install `.[openbb]`)
2. **yfinance** — fallback (install `.[full]`)
3. **Synthetic GBM** — reproducible offline fallback so tests/CI always run

### Factor & macro features (optional)

| Feature set | Flag | API key? | Source |
|---|---|---|---|
| **Fama-French factors** (Mkt-RF, SMB, HML, RF) | `--factors` | No | OpenBB → Ken French library → synthetic |
| **FRED macro** (yield slope, breakeven, VIX) | `--macro` | Free `FRED_API_KEY` | FRED via `fredapi` |

```bash
# Add Fama-French factor exposures (free, no key)
python -m alpha_forecast.cli --symbol AAPL --factors -v

# Add macro features (export your free FRED key first)
export FRED_API_KEY=your_key_here
python -m alpha_forecast.cli --symbol AAPL --factors --macro -v
```

All external features are **forward-filled and lagged one day** before joining,
so they can only ever use information observable at the close of the trading
day — preserving the no-look-ahead guarantee.

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

---

## Testing

```bash
pytest -q
```

Tests run fully offline on synthetic data and include an explicit
**no-look-ahead** check: truncating future rows must not change past feature
values.

---

## Roadmap / extension ideas

- Add deep-learning models (LSTM / N-BEATS / TFT via `darts` or `neuralforecast`)
- GARCH-family volatility forecasting + VaR
- Multi-asset portfolio construction from forecasts (mean-variance / Black-Litterman)
- Macro features from FRED/ECB via OpenBB
- Regime detection (HMM) with regime-conditional model switching

---

## License

MIT — see `LICENSE`.
