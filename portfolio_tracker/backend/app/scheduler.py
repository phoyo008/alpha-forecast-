"""Background jobs: periodic disclosure scraping + broker position sync."""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler

from app.broker.alpaca import AlpacaClient
from app.config import TradingMode, get_settings
from app.db import SessionLocal
from app.ingest.pipeline import run_ingestion
from app.portfolio_sync import sync_portfolio

log = logging.getLogger(__name__)
_scheduler: BackgroundScheduler | None = None


def _ingest_job() -> None:
    with SessionLocal() as db:
        try:
            result = run_ingestion(db)
            log.info("Scheduled ingestion: %s", result)
        except Exception as exc:  # noqa: BLE001
            log.warning("Scheduled ingestion failed: %s", exc)


def _sync_job() -> None:
    settings = get_settings()
    if settings.trading_mode == TradingMode.DISABLED or not AlpacaClient().configured:
        return
    with SessionLocal() as db:
        try:
            sync_portfolio(db)
        except Exception as exc:  # noqa: BLE001
            log.warning("Scheduled portfolio sync failed: %s", exc)


def start_scheduler() -> None:
    global _scheduler
    settings = get_settings()
    if not settings.enable_scheduler or _scheduler is not None:
        return
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _ingest_job, "interval", hours=settings.disclosure_poll_hours, id="ingest"
    )
    _scheduler.add_job(
        _sync_job, "interval", minutes=settings.position_sync_minutes, id="sync"
    )
    _scheduler.start()
    log.info("Scheduler started (ingest every %sh, sync every %sm)",
             settings.disclosure_poll_hours, settings.position_sync_minutes)


def shutdown_scheduler() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
