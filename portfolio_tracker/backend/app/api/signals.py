from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import get_db
from app.models import CopyRule, CopySignal, SignalStatus
from app.schemas import CopyRuleIn, CopyRuleOut, SignalOut
from app.strategy.copy_engine import execute_signal

router = APIRouter(prefix="/api/signals", tags=["signals"])


@router.get("", response_model=list[SignalOut])
def list_signals(status: SignalStatus | None = None, limit: int = 100, db: Session = Depends(get_db)):
    stmt = select(CopySignal).order_by(CopySignal.created_at.desc()).limit(min(limit, 500))
    if status:
        stmt = stmt.where(CopySignal.status == status)
    return list(db.execute(stmt).scalars().all())


@router.post("/{signal_id}/approve", response_model=SignalOut)
def approve_signal(signal_id: int, db: Session = Depends(get_db)):
    """Approve a proposed signal and submit it to the broker (respecting safety rules)."""
    signal = db.get(CopySignal, signal_id)
    if not signal:
        raise HTTPException(404, "signal not found")
    if signal.status != SignalStatus.PROPOSED:
        raise HTTPException(409, f"signal is {signal.status.value}, not proposed")
    signal.status = SignalStatus.APPROVED
    db.commit()
    return execute_signal(db, signal)


@router.post("/{signal_id}/reject", response_model=SignalOut)
def reject_signal(signal_id: int, db: Session = Depends(get_db)):
    signal = db.get(CopySignal, signal_id)
    if not signal:
        raise HTTPException(404, "signal not found")
    if signal.status != SignalStatus.PROPOSED:
        raise HTTPException(409, f"signal is {signal.status.value}, not proposed")
    signal.status = SignalStatus.REJECTED
    db.commit()
    db.refresh(signal)
    return signal


# --- copy rules ---
@router.get("/rules", response_model=list[CopyRuleOut])
def list_rules(db: Session = Depends(get_db)):
    return list(db.execute(select(CopyRule).order_by(CopyRule.id)).scalars().all())


@router.post("/rules", response_model=CopyRuleOut)
def create_rule(body: CopyRuleIn, db: Session = Depends(get_db)):
    rule = CopyRule(**body.model_dump())
    db.add(rule)
    db.commit()
    db.refresh(rule)
    return rule


@router.delete("/rules/{rule_id}")
def delete_rule(rule_id: int, db: Session = Depends(get_db)):
    rule = db.get(CopyRule, rule_id)
    if not rule:
        raise HTTPException(404, "rule not found")
    db.delete(rule)
    db.commit()
    return {"deleted": rule_id}
