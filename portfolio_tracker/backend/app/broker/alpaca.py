"""Thin wrapper around Alpaca's trading API.

One client class serves both requirements:
  * portfolio sync  -> get_account() / get_positions()
  * copy execution  -> submit_notional_order() / cancel_all()

`alpaca-py` is imported lazily so the app boots (and tests run) without the SDK or
any credentials. `paper=` is derived from TRADING_MODE, and live orders additionally
require `settings.live_enabled` — enforced again here as defense in depth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.config import TradingMode, get_settings

log = logging.getLogger(__name__)


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float
    mode: str


@dataclass
class PositionInfo:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    current_price: float | None


@dataclass
class OrderResult:
    order_id: str | None
    status: str
    submitted_notional: float | None
    error: str | None = None


class AlpacaClient:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._client = None

    # -- connection --------------------------------------------------------- #
    @property
    def configured(self) -> bool:
        return bool(self.settings.alpaca_api_key and self.settings.alpaca_secret_key)

    def _trading_client(self):
        if self._client is not None:
            return self._client
        if not self.configured:
            raise RuntimeError("Alpaca credentials are not configured")
        from alpaca.trading.client import TradingClient

        paper = self.settings.trading_mode != TradingMode.LIVE
        self._client = TradingClient(
            self.settings.alpaca_api_key,
            self.settings.alpaca_secret_key,
            paper=paper,
        )
        return self._client

    # -- reads -------------------------------------------------------------- #
    def get_account(self) -> AccountInfo:
        acct = self._trading_client().get_account()
        return AccountInfo(
            equity=float(acct.equity),
            cash=float(acct.cash),
            buying_power=float(acct.buying_power),
            mode=self.settings.trading_mode.value,
        )

    def get_positions(self) -> list[PositionInfo]:
        positions = self._trading_client().get_all_positions()
        out: list[PositionInfo] = []
        for p in positions:
            out.append(
                PositionInfo(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    unrealized_pl=float(p.unrealized_pl),
                    current_price=float(p.current_price) if p.current_price else None,
                )
            )
        return out

    def is_market_open(self) -> bool:
        try:
            return bool(self._trading_client().get_clock().is_open)
        except Exception as exc:  # noqa: BLE001
            log.warning("market clock check failed: %s", exc)
            return False

    # -- writes ------------------------------------------------------------- #
    def submit_notional_order(self, symbol: str, notional: float, side: str) -> OrderResult:
        """Submit a fractional market order for a dollar amount.

        Guarded: LIVE submissions require settings.live_enabled to be true.
        """
        if self.settings.trading_mode == TradingMode.DISABLED:
            return OrderResult(None, "blocked", None, "trading mode is DISABLED")
        if self.settings.trading_mode == TradingMode.LIVE and not self.settings.live_enabled:
            return OrderResult(None, "blocked", None, "LIVE mode not confirmed")

        try:
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest

            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL
            req = MarketOrderRequest(
                symbol=symbol,
                notional=round(notional, 2),
                side=order_side,
                time_in_force=TimeInForce.DAY,
            )
            order = self._trading_client().submit_order(req)
            return OrderResult(
                order_id=str(order.id),
                status=str(order.status),
                submitted_notional=round(notional, 2),
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("order submit failed for %s: %s", symbol, exc)
            return OrderResult(None, "failed", None, str(exc))

    def cancel_all(self) -> int:
        """Cancel all open orders. Returns the count reported by Alpaca."""
        try:
            resp = self._trading_client().cancel_orders()
            return len(resp or [])
        except Exception as exc:  # noqa: BLE001
            log.warning("cancel_all failed: %s", exc)
            return 0
