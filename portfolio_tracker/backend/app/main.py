"""FastAPI application entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import health, politicians, portfolio, settings as settings_api, trades
from app.api import signals
from app.config import get_settings
from app.db import SessionLocal, init_db
from app.scheduler import shutdown_scheduler, start_scheduler
from app.seed import seed_politicians

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with SessionLocal() as db:
        created = seed_politicians(db)
        if created:
            log.info("Seeded %d politicians", created)
    start_scheduler()
    yield
    shutdown_scheduler()


def create_app() -> FastAPI:
    s = get_settings()
    app = FastAPI(title=s.app_name, version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=s.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    for module in (health, politicians, trades, portfolio, signals, settings_api):
        app.include_router(module.router)
    return app


app = create_app()
