from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import Politician
from app.schemas import FollowUpdate, PoliticianOut

router = APIRouter(prefix="/api/politicians", tags=["politicians"])


@router.get("", response_model=list[PoliticianOut])
def list_politicians(followed_only: bool = False, db: Session = Depends(get_db)):
    stmt = select(Politician).order_by(Politician.full_name)
    if followed_only:
        stmt = stmt.where(Politician.followed.is_(True))
    return list(db.execute(stmt).scalars().all())


@router.patch("/{politician_id}/follow", response_model=PoliticianOut)
def set_follow(politician_id: int, body: FollowUpdate, db: Session = Depends(get_db)):
    pol = db.get(Politician, politician_id)
    if not pol:
        raise HTTPException(404, "politician not found")
    pol.followed = body.followed
    db.commit()
    db.refresh(pol)
    return pol
