"""Persistence + orchestration for disclosure ingestion.

`run_ingestion` runs every source, upserts politicians/filings/trades with dedup,
and asks the copy engine to generate signals for newly-seen trades that belong to
followed politicians.
"""

from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ingest.base import TradeRecord
from app.ingest.house_clerk import HouseClerkSource
from app.ingest.senate_efd import SenateEFDSource
from app.models import Filing, Politician, PoliticianTrade

log = logging.getLogger(__name__)


def default_sources():
    return [HouseClerkSource(), SenateEFDSource()]


def _get_or_create_politician(db: Session, rec: TradeRecord) -> Politician:
    stmt = select(Politician).where(
        Politician.full_name.ilike(rec.politician_name.strip())
    )
    pol = db.execute(stmt).scalar_one_or_none()
    if pol is None:
        pol = Politician(
            full_name=rec.politician_name.strip(),
            chamber=rec.chamber,
            party=rec.party,
            state=rec.state,
            external_id=rec.external_id,
            active=True,
            followed=False,
        )
        db.add(pol)
        db.flush()
    return pol


def _get_or_create_filing(db: Session, rec: TradeRecord, pol: Politician) -> Filing | None:
    if not rec.doc_id:
        return None
    stmt = select(Filing).where(Filing.source == rec.source, Filing.doc_id == rec.doc_id)
    filing = db.execute(stmt).scalar_one_or_none()
    if filing is None:
        filing = Filing(
            politician_id=pol.id,
            source=rec.source,
            doc_id=rec.doc_id,
            filed_date=rec.disclosed_date,
            url=rec.url,
        )
        db.add(filing)
        db.flush()
    return filing


def persist_records(db: Session, records: list[TradeRecord]) -> list[PoliticianTrade]:
    """Insert new trades, skipping any whose dedup_hash already exists."""
    new_trades: list[PoliticianTrade] = []
    for rec in records:
        h = rec.dedup_hash()
        exists = db.execute(
            select(PoliticianTrade.id).where(PoliticianTrade.dedup_hash == h)
        ).first()
        if exists:
            continue
        pol = _get_or_create_politician(db, rec)
        filing = _get_or_create_filing(db, rec, pol)
        trade = PoliticianTrade(
            politician_id=pol.id,
            filing_id=filing.id if filing else None,
            ticker=rec.ticker,
            asset_type=rec.asset_type,
            tx_type=rec.tx_type,
            amount_low=rec.amount_low,
            amount_high=rec.amount_high,
            tx_date=rec.tx_date,
            disclosed_date=rec.disclosed_date,
            raw_desc=rec.raw_desc,
            dedup_hash=h,
        )
        db.add(trade)
        new_trades.append(trade)
    db.commit()
    return new_trades


def run_ingestion(db: Session, since: date | None = None, sources=None) -> dict:
    """Entry point used by the scheduler and the manual trigger endpoint."""
    sources = sources or default_sources()
    all_records: list[TradeRecord] = []
    for src in sources:
        try:
            recs = src.fetch(since=since)
            log.info("%s yielded %d records", src.name, len(recs))
            all_records.extend(recs)
        except Exception as exc:  # noqa: BLE001
            log.warning("Source %s failed: %s", src.name, exc)

    new_trades = persist_records(db, all_records)

    # Generate copy signals for new trades belonging to followed politicians.
    signals_created = 0
    try:
        from app.strategy.copy_engine import generate_signals_for_trades

        signals_created = generate_signals_for_trades(db, new_trades)
    except Exception as exc:  # noqa: BLE001
        log.warning("Signal generation failed: %s", exc)

    return {
        "fetched": len(all_records),
        "new_trades": len(new_trades),
        "signals_created": signals_created,
    }
