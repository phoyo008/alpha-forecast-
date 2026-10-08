"""Sync broker account + positions into the DB.

Called by the scheduler (periodically) and by the manual refresh endpoint.
Positions are replaced wholesale; account equity is appended as a snapshot so
`safety.py` can enforce the %-of-equity limit and the dashboard can chart history.
"""

from __future__ import annotations

import logging

from sqlalchemy import delete
from sqlalchemy.orm import Session

from app.broker.alpaca import AlpacaClient
from app.models import AccountSnapshot, Position

log = logging.getLogger(__name__)


def sync_portfolio(db: Session, client: AlpacaClient | None = None) -> dict:
    client = client or AlpacaClient()
    if not client.configured:
        return {"connected": False, "positions": 0}

    account = client.get_account()
    db.add(
        AccountSnapshot(
            equity=account.equity,
            cash=account.cash,
            buying_power=account.buying_power,
            mode=account.mode,
        )
    )

    positions = client.get_positions()
    db.execute(delete(Position))  # replace the whole set each sync
    for p in positions:
        db.add(
            Position(
                symbol=p.symbol,
                qty=p.qty,
                avg_entry_price=p.avg_entry_price,
                market_value=p.market_value,
                unrealized_pl=p.unrealized_pl,
                current_price=p.current_price,
            )
        )
    db.commit()
    log.info("Portfolio synced: equity=%.2f positions=%d", account.equity, len(positions))
    return {"connected": True, "positions": len(positions), "equity": account.equity}
