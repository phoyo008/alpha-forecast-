"""Runtime configuration read from environment variables.

``ALPHA_FORECAST_OFFLINE``
    When truthy (``1``/``true``/``yes``), skip every network data provider and
    use the reproducible synthetic fallbacks. The test suite sets this.
``ALPHA_FORECAST_CACHE_DIR``
    Directory for cached provider downloads (default ``~/.cache/alpha_forecast``).
    Set to an empty string to disable caching.
``BLOCKCHAIR_API_KEY``
    Key for the Blockchair on-chain data API (https://blockchair.com/api).
    Never written to disk or committed; without it on-chain features are skipped.
``ALPHA_FORECAST_HOLDINGS``
    Path to a CSV of brokerage holdings (default
    ``~/.config/alpha_forecast/holdings.csv``). Kept outside the repository.
``ALPHA_FORECAST_QUOTES``
    Path to a broker quote snapshot CSV (default
    ``~/.config/alpha_forecast/quotes.csv``). Set to an empty string to ignore it.
"""

from __future__ import annotations

import os
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on"}


def is_offline() -> bool:
    return os.environ.get("ALPHA_FORECAST_OFFLINE", "").strip().lower() in _TRUTHY


def cache_dir() -> Path | None:
    raw = os.environ.get("ALPHA_FORECAST_CACHE_DIR")
    if raw is None:
        return Path.home() / ".cache" / "alpha_forecast"
    return Path(raw).expanduser() if raw.strip() else None


def blockchair_api_key() -> str | None:
    key = os.environ.get("BLOCKCHAIR_API_KEY", "").strip()
    return key or None


def holdings_path() -> Path:
    raw = os.environ.get("ALPHA_FORECAST_HOLDINGS", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".config" / "alpha_forecast" / "holdings.csv"


def quotes_snapshot_path() -> Path | None:
    raw = os.environ.get("ALPHA_FORECAST_QUOTES")
    if raw is None:
        return Path.home() / ".config" / "alpha_forecast" / "quotes.csv"
    return Path(raw).expanduser() if raw.strip() else None
