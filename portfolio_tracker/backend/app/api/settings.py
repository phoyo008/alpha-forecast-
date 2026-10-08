"""Trading status, runtime mode control, and the kill switch.

Runtime changes mutate the in-memory Settings only; they never widen what the
environment allows. In particular, switching to LIVE at runtime still requires
`LIVE_CONFIRMED=true` in the environment — the UI cannot grant live trading on
its own. On restart, the environment values govern again (a safe default).
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.broker.alpaca import AlpacaClient
from app.broker.safety import orders_today
from app.config import TradingMode, get_settings
from app.db import get_db
from app.schemas import TradingStatusOut

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/status", response_model=TradingStatusOut)
def status(db: Session = Depends(get_db)):
    s = get_settings()
    return TradingStatusOut(
        trading_mode=s.trading_mode.value,
        live_enabled=s.live_enabled,
        auto_execute=s.auto_execute,
        broker_connected=AlpacaClient().configured,
        max_order_notional=s.max_order_notional,
        max_position_pct=s.max_position_pct,
        max_orders_per_day=s.max_orders_per_day,
        orders_today=orders_today(db),
    )


class ModeUpdate(BaseModel):
    mode: TradingMode


@router.post("/mode", response_model=TradingStatusOut)
def set_mode(body: ModeUpdate, db: Session = Depends(get_db)):
    s = get_settings()
    # Moving to LIVE at runtime is only honored if the env already confirmed it.
    if body.mode == TradingMode.LIVE and not s.live_confirmed:
        s.trading_mode = TradingMode.PAPER  # refuse to silently enable live
    else:
        s.trading_mode = body.mode
    return status(db)


class LimitsUpdate(BaseModel):
    max_order_notional: float | None = None
    max_position_pct: float | None = None
    max_orders_per_day: int | None = None
    auto_execute: bool | None = None


@router.post("/limits", response_model=TradingStatusOut)
def set_limits(body: LimitsUpdate, db: Session = Depends(get_db)):
    s = get_settings()
    if body.max_order_notional is not None:
        s.max_order_notional = body.max_order_notional
    if body.max_position_pct is not None:
        s.max_position_pct = body.max_position_pct
    if body.max_orders_per_day is not None:
        s.max_orders_per_day = body.max_orders_per_day
    if body.auto_execute is not None:
        s.auto_execute = body.auto_execute
    return status(db)


@router.post("/kill")
def kill_switch(db: Session = Depends(get_db)):
    """Emergency stop: cancel open broker orders and force mode to DISABLED."""
    s = get_settings()
    cancelled = 0
    client = AlpacaClient()
    if client.configured and s.trading_mode != TradingMode.DISABLED:
        cancelled = client.cancel_all()
    s.trading_mode = TradingMode.DISABLED
    s.auto_execute = False
    return {"status": "killed", "mode": s.trading_mode.value, "orders_cancelled": cancelled}
