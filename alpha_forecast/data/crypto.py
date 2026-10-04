"""Crypto market and on-chain data.

Prices for crypto pairs (``BTC-USD``, ``ETH-USD``...) come through the normal
:func:`alpha_forecast.data.load_prices` path, which yfinance serves on a
seven-day calendar. This module adds what equities do not have: on-chain
activity from the Blockchair API (https://blockchair.com/api/docs).

Two reads are exposed:

* :func:`network_stats` -- the current snapshot of a chain (price, hashrate,
  mempool, 24h transactions) for display.
* :func:`daily_onchain` -- daily aggregates over the blocks table (block count,
  transactions, fees, transferred value, difficulty) for use as model
  features. These are lagged one day by the feature builder before use.

The API key is read from ``BLOCKCHAIR_API_KEY``. Blockchair rate-limits and
blacklists unkeyed clients quickly, so without a key both functions return
``None`` rather than attempting unauthenticated calls. Daily aggregates are
cached on disk like price downloads; the ``stats`` snapshot is never cached.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

import pandas as pd

from alpha_forecast.config import blockchair_api_key, cache_dir, is_offline

logger = logging.getLogger(__name__)

BASE_URL = "https://api.blockchair.com"
TIMEOUT_SECONDS = 20
CACHE_TTL_OPEN_RANGE = 12 * 3600  # seconds

# Ticker prefix -> Blockchair chain slug. Only chains whose blocks table
# exposes the aggregate fields used below are listed.
CHAIN_BY_ASSET: dict[str, str] = {
    "BTC": "bitcoin",
    "BCH": "bitcoin-cash",
    "LTC": "litecoin",
    "DOGE": "dogecoin",
    "DASH": "dash",
    "ZEC": "zcash",
    "ETH": "ethereum",
}

# Aggregates requested from ``/{chain}/blocks``; mapped to feature names.
# ``output_total_usd`` does not exist on Ethereum blocks, so it is dropped
# there and the ``volume_usd`` feature is simply absent.
_BLOCK_AGGREGATES: dict[str, str] = {
    "count()": "blocks",
    "sum(transaction_count)": "tx_count",
    "sum(fee_total_usd)": "fees_usd",
    "sum(output_total_usd)": "volume_usd",
    "avg(difficulty)": "difficulty",
}
_ETH_UNSUPPORTED = {"sum(output_total_usd)"}

ONCHAIN_COLUMNS = ["blocks", "tx_count", "fees_usd", "volume_usd", "difficulty"]


def asset_of(symbol: str) -> str:
    """``"btc-usd"`` -> ``"BTC"``; ``"AAPL"`` -> ``"AAPL"``."""
    return symbol.strip().upper().split("-")[0]


def chain_for_symbol(symbol: str) -> str | None:
    """Blockchair chain slug for a ticker, or ``None`` for non-crypto symbols."""
    return CHAIN_BY_ASSET.get(asset_of(symbol))


def is_crypto_symbol(symbol: str) -> bool:
    """True for ``XXX-USD`` style pairs or any asset Blockchair knows."""
    s = symbol.strip().upper()
    return "-" in s or asset_of(s) in CHAIN_BY_ASSET


def _get(path: str, params: dict[str, str] | None = None) -> dict | None:
    key = blockchair_api_key()
    if key is None:
        logger.info("No BLOCKCHAIR_API_KEY set; skipping %s", path)
        return None
    query = dict(params or {})
    query["key"] = key
    url = f"{BASE_URL}/{path.lstrip('/')}?{urllib.parse.urlencode(query)}"
    req = urllib.request.Request(url, headers={"User-Agent": "alpha-forecast"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:  # pragma: no cover - network dependent
        logger.warning("Blockchair %s failed: HTTP %s", path, exc.code)
        return None
    except Exception as exc:  # pragma: no cover - network dependent
        logger.warning("Blockchair %s failed: %s", path, exc)
        return None
    code = payload.get("context", {}).get("code")
    if code != 200:
        logger.warning("Blockchair %s returned code %s: %s", path, code,
                       payload.get("context", {}).get("error"))
        return None
    return payload


def network_stats(chain: str) -> dict | None:
    """Current snapshot of ``chain`` (``/{chain}/stats``), or ``None``."""
    if is_offline():
        return None
    payload = _get(f"{chain}/stats")
    if payload is None:
        return None
    data = payload.get("data")
    return data if isinstance(data, dict) else None


def onchain_frame(rows: list[dict], chain: str) -> pd.DataFrame:
    """Turn Blockchair block aggregates into a daily feature frame."""
    aggregates = {k: v for k, v in _BLOCK_AGGREGATES.items()
                  if chain != "ethereum" or k not in _ETH_UNSUPPORTED}
    if not rows:
        return pd.DataFrame(columns=list(aggregates.values()), index=pd.DatetimeIndex([], name="date"))
    df = pd.DataFrame(rows)
    df = df.rename(columns=aggregates)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    keep = [c for c in aggregates.values() if c in df.columns]
    df = df[keep].astype(float)
    df.index.name = "date"
    return df


def _cache_path(chain: str, start: str, end: str) -> Path | None:
    root = cache_dir()
    if root is None:
        return None
    return root / "onchain" / f"{chain}_{start}_{end}.pkl"


def _read_cache(path: Path | None, end: str) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    if pd.to_datetime(end).date() >= date.today():
        if time.time() - path.stat().st_mtime > CACHE_TTL_OPEN_RANGE:
            return None
    try:
        return pd.read_pickle(path)
    except Exception as exc:
        logger.warning("Ignoring unreadable cache file %s: %s", path, exc)
        return None


def _write_cache(path: Path | None, df: pd.DataFrame) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(path)
    except OSError as exc:
        logger.warning("Could not write on-chain cache %s: %s", path, exc)


def daily_onchain(
    chain: str,
    start: str,
    end: str,
    *,
    use_cache: bool = True,
) -> pd.DataFrame | None:
    """Daily on-chain aggregates for ``chain`` between ``start`` and ``end``.

    Columns (where the chain supports them): ``blocks``, ``tx_count``,
    ``fees_usd``, ``volume_usd``, ``difficulty``. Returns ``None`` when
    offline, when no API key is set, or when the request fails.
    """
    if is_offline():
        logger.info("Offline mode; skipping on-chain features.")
        return None
    if blockchair_api_key() is None:
        logger.info("No BLOCKCHAIR_API_KEY set; skipping on-chain features.")
        return None

    cache_file = _cache_path(chain, start, end) if use_cache else None
    cached = _read_cache(cache_file, end)
    if cached is not None:
        return cached

    aggregates = [k for k in _BLOCK_AGGREGATES if chain != "ethereum" or k not in _ETH_UNSUPPORTED]
    payload = _get(
        f"{chain}/blocks",
        {"a": "date," + ",".join(aggregates), "q": f"time({start}..{end})"},
    )
    if payload is None:
        return None
    rows = payload.get("data")
    if not isinstance(rows, list):
        return None
    df = onchain_frame(rows, chain)
    logger.info("Loaded %d days of %s on-chain data from Blockchair", len(df), chain)
    _write_cache(cache_file, df)
    return df


def onchain_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive model-ready features from raw daily aggregates.

    Levels such as transaction counts trend over years, so features are
    expressed as log changes and ratios that are comparable across time.
    All windows look backwards only.
    """
    out = pd.DataFrame(index=df.index)
    if "tx_count" in df.columns:
        tx = df["tx_count"].replace(0, float("nan"))
        out["tx_growth_7"] = tx.rolling(7).mean().pipe(lambda s: s / s.shift(7) - 1.0)
    if "fees_usd" in df.columns and "tx_count" in df.columns:
        out["fee_per_tx_usd"] = df["fees_usd"] / df["tx_count"].replace(0, float("nan"))
    if "volume_usd" in df.columns:
        vol = df["volume_usd"].replace(0, float("nan"))
        out["volume_z_30"] = (vol - vol.rolling(30).mean()) / vol.rolling(30).std()
    # Proof-of-stake chains (Ethereum) report difficulty 0; the ratio is undefined.
    if "difficulty" in df.columns and (df["difficulty"] > 0).all():
        out["difficulty_chg_14"] = df["difficulty"] / df["difficulty"].shift(14) - 1.0
    if "blocks" in df.columns:
        out["blocks_vs_30"] = df["blocks"] / df["blocks"].rolling(30).mean() - 1.0
    # A column that never resolves would make the feature builder drop every row.
    return out.dropna(axis=1, how="all")
