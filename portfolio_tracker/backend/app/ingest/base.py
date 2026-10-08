"""Normalization contract shared by every disclosure source.

A source's job is to yield `TradeRecord`s. Persisting them (dedup, matching to
Politician rows, creating Filing rows) is handled centrally in `pipeline.py`, so
adding a new source only means implementing `fetch()`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date
from typing import Protocol

from app.models import Chamber, TxType


@dataclass
class TradeRecord:
    """Source-agnostic representation of one disclosed transaction."""

    source: str                       # "house_clerk" | "senate_efd"
    politician_name: str
    chamber: Chamber
    tx_type: TxType
    ticker: str | None = None
    asset_type: str | None = None
    amount_low: float | None = None
    amount_high: float | None = None
    tx_date: date | None = None
    disclosed_date: date | None = None
    doc_id: str | None = None
    url: str | None = None
    party: str | None = None
    state: str | None = None
    raw_desc: str | None = None
    external_id: str | None = None
    extra: dict = field(default_factory=dict)

    def dedup_hash(self) -> str:
        """Stable fingerprint. Two scrapes of the same disclosed transaction collide."""
        key = "|".join(
            str(x)
            for x in (
                self.source,
                self.politician_name.strip().lower(),
                (self.ticker or self.raw_desc or "").strip().lower(),
                self.tx_type.value,
                self.tx_date.isoformat() if self.tx_date else "",
                self.amount_low or "",
                self.amount_high or "",
                self.doc_id or "",
            )
        )
        return hashlib.sha256(key.encode()).hexdigest()


class DisclosureSource(Protocol):
    """Interface every scraper implements."""

    name: str

    def fetch(self, since: date | None = None) -> list[TradeRecord]:
        ...
