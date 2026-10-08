from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.ingest.pipeline import run_ingestion
from app.models import Politician, PoliticianTrade
from app.schemas import TradeFeedItem

router = APIRouter(prefix="/api/trades", tags=["trades"])


@router.get("", response_model=list[TradeFeedItem])
def trade_feed(
    limit: int = 100,
    followed_only: bool = False,
    politician_id: int | None = None,
    db: Session = Depends(get_db),
):
    stmt = (
        select(PoliticianTrade, Politician)
        .join(Politician, Politician.id == PoliticianTrade.politician_id)
        .order_by(PoliticianTrade.disclosed_date.desc().nullslast(), PoliticianTrade.id.desc())
        .limit(min(limit, 500))
    )
    if followed_only:
        stmt = stmt.where(Politician.followed.is_(True))
    if politician_id:
        stmt = stmt.where(PoliticianTrade.politician_id == politician_id)

    items: list[TradeFeedItem] = []
    for trade, pol in db.execute(stmt).all():
        items.append(
            TradeFeedItem(
                **{
                    "id": trade.id,
                    "politician_id": trade.politician_id,
                    "ticker": trade.ticker,
                    "asset_type": trade.asset_type,
                    "tx_type": trade.tx_type,
                    "amount_low": trade.amount_low,
                    "amount_high": trade.amount_high,
                    "tx_date": trade.tx_date,
                    "disclosed_date": trade.disclosed_date,
                    "disclosure_lag_days": trade.disclosure_lag_days,
                    "raw_desc": trade.raw_desc,
                    "created_at": trade.created_at,
                    "politician_name": pol.full_name,
                    "chamber": pol.chamber,
                }
            )
        )
    return items


@router.post("/refresh")
def refresh_now(db: Session = Depends(get_db)):
    """Manually trigger a disclosure scrape (also runs on a schedule)."""
    return run_ingestion(db)
