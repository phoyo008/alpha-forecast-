"""Pydantic response/request models for the API."""

from datetime import date, datetime

from pydantic import BaseModel, ConfigDict

from app.models import Chamber, SignalStatus, SizingMode, TxType


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# --- Politicians ---
class PoliticianOut(ORMModel):
    id: int
    full_name: str
    chamber: Chamber
    party: str | None
    state: str | None
    followed: bool
    active: bool


class FollowUpdate(BaseModel):
    followed: bool


# --- Trades ---
class TradeOut(ORMModel):
    id: int
    politician_id: int
    ticker: str | None
    asset_type: str | None
    tx_type: TxType
    amount_low: float | None
    amount_high: float | None
    tx_date: date | None
    disclosed_date: date | None
    disclosure_lag_days: int | None
    raw_desc: str | None
    created_at: datetime


class TradeFeedItem(TradeOut):
    politician_name: str
    chamber: Chamber


# --- Portfolio ---
class PositionOut(ORMModel):
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    unrealized_pl: float
    current_price: float | None


class AccountOut(ORMModel):
    equity: float
    cash: float
    buying_power: float
    mode: str
    synced_at: datetime


class PortfolioOut(BaseModel):
    account: AccountOut | None
    positions: list[PositionOut]
    connected: bool


# --- Copy rules / signals ---
class CopyRuleOut(ORMModel):
    id: int
    politician_id: int | None
    enabled: bool
    sizing_mode: SizingMode
    sizing_value: float
    min_amount_filter: float
    buy_only: bool


class CopyRuleIn(BaseModel):
    politician_id: int | None = None
    enabled: bool = True
    sizing_mode: SizingMode = SizingMode.FIXED_NOTIONAL
    sizing_value: float = 100.0
    min_amount_filter: float = 0.0
    buy_only: bool = True


class ExecutionOut(ORMModel):
    id: int
    alpaca_order_id: str | None
    submitted_notional: float | None
    status: str
    error: str | None
    updated_at: datetime


class SignalOut(ORMModel):
    id: int
    politician_trade_id: int
    ticker: str
    side: TxType
    target_notional: float
    status: SignalStatus
    note: str | None
    created_at: datetime
    execution: ExecutionOut | None


# --- Settings / status ---
class TradingStatusOut(BaseModel):
    trading_mode: str
    live_enabled: bool
    auto_execute: bool
    broker_connected: bool
    max_order_notional: float
    max_position_pct: float
    max_orders_per_day: int
    orders_today: int
