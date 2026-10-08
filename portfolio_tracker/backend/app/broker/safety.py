"""Pre-trade risk checks — the last gate before any order is submitted.

Every proposed order passes through `check_order`. A failed check never raises;
it returns a reason string so the signal can be recorded as SKIPPED with context.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import Settings, TradingMode, get_settings
from app.models import AccountSnapshot, Execution


@dataclass
class SafetyResult:
    ok: bool
    reason: str | None = None


def orders_today(db: Session) -> int:
    start = datetime.now(timezone.utc) - timedelta(hours=24)
    stmt = select(func.count(Execution.id)).where(
        Execution.created_at >= start,
        Execution.alpaca_order_id.is_not(None),
    )
    return db.execute(stmt).scalar_one()


def latest_equity(db: Session) -> float | None:
    snap = db.execute(
        select(AccountSnapshot).order_by(AccountSnapshot.synced_at.desc())
    ).scalars().first()
    return snap.equity if snap else None


def check_order(
    db: Session,
    symbol: str,
    notional: float,
    settings: Settings | None = None,
) -> SafetyResult:
    settings = settings or get_settings()

    if settings.trading_mode == TradingMode.DISABLED:
        return SafetyResult(False, "trading mode is DISABLED")
    if settings.trading_mode == TradingMode.LIVE and not settings.live_confirmed:
        return SafetyResult(False, "LIVE mode requires LIVE_CONFIRMED=true")

    if symbol.upper() in {s.upper() for s in settings.symbol_denylist}:
        return SafetyResult(False, f"{symbol} is on the denylist")

    if notional <= 0:
        return SafetyResult(False, "non-positive notional")
    if notional > settings.max_order_notional:
        return SafetyResult(
            False,
            f"order ${notional:.0f} exceeds per-order cap ${settings.max_order_notional:.0f}",
        )

    equity = latest_equity(db)
    if equity:
        pct = (notional / equity) * 100
        if pct > settings.max_position_pct:
            return SafetyResult(
                False,
                f"order is {pct:.1f}% of equity, over the {settings.max_position_pct:.1f}% cap",
            )

    if orders_today(db) >= settings.max_orders_per_day:
        return SafetyResult(False, f"daily order cap ({settings.max_orders_per_day}) reached")

    return SafetyResult(True)
