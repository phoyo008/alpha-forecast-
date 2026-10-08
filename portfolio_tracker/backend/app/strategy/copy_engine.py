r"""Turn politician trades into sized, safety-checked copy signals — and execute them.

Signal lifecycle:
  proposed --(human approve OR auto_execute)--> submitted --> filled/failed
                                    \--(safety fail)--> skipped

Execution is deliberately separate from signal creation so the default flow is
"human approves in the UI"; `auto_execute=true` collapses the two but every order
still passes `broker/safety.check_order`.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.broker.alpaca import AlpacaClient
from app.broker.safety import check_order
from app.config import get_settings
from app.models import (
    CopyRule,
    CopySignal,
    Execution,
    Politician,
    PoliticianTrade,
    SignalStatus,
    SizingMode,
    TxType,
)

log = logging.getLogger(__name__)


def _matching_rules(db: Session, trade: PoliticianTrade) -> list[CopyRule]:
    """Enabled rules that target this politician specifically, or all followed ones."""
    stmt = select(CopyRule).where(
        CopyRule.enabled.is_(True),
        (CopyRule.politician_id == trade.politician_id) | (CopyRule.politician_id.is_(None)),
    )
    return list(db.execute(stmt).scalars().all())


def _target_notional(rule: CopyRule, equity: float | None) -> float:
    if rule.sizing_mode == SizingMode.PCT_PORTFOLIO and equity:
        return round(equity * (rule.sizing_value / 100.0), 2)
    return round(rule.sizing_value, 2)  # FIXED_NOTIONAL (or no equity yet)


def generate_signals_for_trades(db: Session, trades: list[PoliticianTrade]) -> int:
    """Create signals for newly-ingested trades of followed politicians."""
    settings = get_settings()
    created = 0
    for trade in trades:
        if not trade.ticker:
            continue  # can only copy identifiable tickers
        pol = db.get(Politician, trade.politician_id)
        if not pol or not pol.followed:
            continue

        for rule in _matching_rules(db, trade):
            if rule.buy_only and trade.tx_type != TxType.BUY:
                continue
            # Filter out disclosed trades below the rule's minimum size.
            if rule.min_amount_filter and (trade.amount_low or 0) < rule.min_amount_filter:
                continue

            equity = _latest_equity(db)
            notional = _target_notional(rule, equity)
            signal = CopySignal(
                politician_trade_id=trade.id,
                rule_id=rule.id,
                ticker=trade.ticker,
                side=trade.tx_type,
                target_notional=notional,
                status=SignalStatus.PROPOSED,
            )
            db.add(signal)
            db.flush()
            created += 1

            if settings.auto_execute:
                execute_signal(db, signal)
    db.commit()
    return created


def _latest_equity(db: Session) -> float | None:
    from app.broker.safety import latest_equity

    return latest_equity(db)


def execute_signal(db: Session, signal: CopySignal, client: AlpacaClient | None = None) -> CopySignal:
    """Submit a single signal to the broker, enforcing safety first. Idempotent."""
    if signal.execution is not None:
        return signal  # already acted upon

    safety = check_order(db, signal.ticker, signal.target_notional)
    if not safety.ok:
        signal.status = SignalStatus.SKIPPED
        signal.note = safety.reason
        db.add(Execution(signal_id=signal.id, status="skipped", error=safety.reason))
        db.commit()
        log.info("Signal %s skipped: %s", signal.id, safety.reason)
        return signal

    client = client or AlpacaClient()
    result = client.submit_notional_order(
        symbol=signal.ticker,
        notional=signal.target_notional,
        side="buy" if signal.side == TxType.BUY else "sell",
    )
    execution = Execution(
        signal_id=signal.id,
        alpaca_order_id=result.order_id,
        submitted_notional=result.submitted_notional,
        status=result.status,
        error=result.error,
    )
    db.add(execution)
    if result.order_id:
        signal.status = SignalStatus.SUBMITTED
    else:
        signal.status = SignalStatus.FAILED
        signal.note = result.error
    db.commit()
    log.info("Signal %s -> %s (%s)", signal.id, result.status, result.error or "ok")
    return signal
