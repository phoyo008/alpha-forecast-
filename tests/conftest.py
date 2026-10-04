"""Keep the suite hermetic: no network providers, no on-disk price cache."""

from __future__ import annotations

import os

os.environ["ALPHA_FORECAST_OFFLINE"] = "1"
os.environ["ALPHA_FORECAST_CACHE_DIR"] = ""
os.environ["ALPHA_FORECAST_QUOTES"] = ""
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
