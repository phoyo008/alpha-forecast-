from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.broker.alpaca import AlpacaClient
from app.db import get_db
from app.models import AccountSnapshot, Position
from app.portfolio_sync import sync_portfolio
from app.schemas import AccountOut, PortfolioOut, PositionOut

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


@router.get("", response_model=PortfolioOut)
def get_portfolio(db: Session = Depends(get_db)):
    account = db.execute(
        select(AccountSnapshot).order_by(AccountSnapshot.synced_at.desc())
    ).scalars().first()
    positions = db.execute(select(Position).order_by(Position.market_value.desc())).scalars().all()
    return PortfolioOut(
        account=AccountOut.model_validate(account) if account else None,
        positions=[PositionOut.model_validate(p) for p in positions],
        connected=AlpacaClient().configured,
    )


@router.post("/sync")
def sync_now(db: Session = Depends(get_db)):
    """Pull the latest account + positions from the broker."""
    return sync_portfolio(db)
