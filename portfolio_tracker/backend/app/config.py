"""Application configuration.

All settings load from environment variables (and an optional `.env` file).
The trading-mode / risk-limit settings here are the primary safety controls
for live execution — read `broker/safety.py` for how they are enforced.
"""

from enum import Enum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class TradingMode(str, Enum):
    """Global execution mode. Defaults to DISABLED so nothing can trade by accident."""

    DISABLED = "DISABLED"  # no orders are ever submitted
    PAPER = "PAPER"        # orders go to Alpaca paper (simulated) account
    LIVE = "LIVE"          # real money — additionally requires LIVE_CONFIRMED=true


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- Core ---
    app_name: str = "Portfolio Tracker"
    database_url: str = "sqlite:///./portfolio.db"
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # --- Scheduler ---
    enable_scheduler: bool = True
    disclosure_poll_hours: int = 24     # how often to scrape new disclosures
    position_sync_minutes: int = 60     # how often to sync broker positions

    # --- Broker (Alpaca) ---
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    # Paper endpoint by default. The live endpoint is only used when TRADING_MODE=LIVE.
    alpaca_paper_base_url: str = "https://paper-api.alpaca.markets"
    alpaca_live_base_url: str = "https://api.alpaca.markets"

    # --- Trading safety controls ---
    trading_mode: TradingMode = TradingMode.DISABLED
    # A second, explicit gate that must ALSO be true for LIVE orders to submit.
    live_confirmed: bool = False
    # When false, signals wait in an approval queue for a human click.
    auto_execute: bool = False

    # Hard risk limits (enforced in broker/safety.py before every order)
    max_order_notional: float = 500.0      # max $ per single order
    max_position_pct: float = 5.0          # max % of equity in one symbol
    max_orders_per_day: int = 10           # daily submitted-order cap
    symbol_denylist: list[str] = []        # symbols that must never be traded

    @property
    def alpaca_base_url(self) -> str:
        return (
            self.alpaca_live_base_url
            if self.trading_mode == TradingMode.LIVE
            else self.alpaca_paper_base_url
        )

    @property
    def live_enabled(self) -> bool:
        """Live orders require BOTH the mode and the explicit confirmation flag."""
        return self.trading_mode == TradingMode.LIVE and self.live_confirmed


@lru_cache
def get_settings() -> Settings:
    return Settings()
