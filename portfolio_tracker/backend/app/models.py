"""SQLAlchemy models.

Domain split:
  * Ingestion side:  Politician -> Filing -> PoliticianTrade
  * Personal side:   AccountSnapshot / Position (synced from the broker)
  * Copy side:       CopyRule -> CopySignal -> Execution
"""

from datetime import date, datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
class Chamber(str, PyEnum):
    HOUSE = "house"
    SENATE = "senate"


class TxType(str, PyEnum):
    BUY = "buy"
    SELL = "sell"
    EXCHANGE = "exchange"


class SignalStatus(str, PyEnum):
    PROPOSED = "proposed"     # awaiting human approval
    APPROVED = "approved"     # approved, queued for submission
    REJECTED = "rejected"     # human declined
    SUBMITTED = "submitted"   # sent to broker
    FILLED = "filled"
    FAILED = "failed"
    SKIPPED = "skipped"       # blocked by a safety rule


# --------------------------------------------------------------------------- #
# Ingestion side
# --------------------------------------------------------------------------- #
class Politician(Base):
    __tablename__ = "politicians"

    id: Mapped[int] = mapped_column(primary_key=True)
    full_name: Mapped[str] = mapped_column(String(200), index=True)
    chamber: Mapped[Chamber] = mapped_column(Enum(Chamber))
    party: Mapped[str | None] = mapped_column(String(40), nullable=True)
    state: Mapped[str | None] = mapped_column(String(4), nullable=True)
    # `active` marks whether we ingest this person; `followed` marks copy interest.
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    followed: Mapped[bool] = mapped_column(Boolean, default=False)
    # Stable key used to match scraped filings back to this row (e.g. bioguide id or slug).
    external_id: Mapped[str | None] = mapped_column(String(80), unique=True, nullable=True)

    filings: Mapped[list["Filing"]] = relationship(back_populates="politician")
    trades: Mapped[list["PoliticianTrade"]] = relationship(back_populates="politician")


class Filing(Base):
    """A single disclosure document (e.g. a Periodic Transaction Report)."""

    __tablename__ = "filings"
    __table_args__ = (UniqueConstraint("source", "doc_id", name="uq_filing_source_doc"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    politician_id: Mapped[int] = mapped_column(ForeignKey("politicians.id"), index=True)
    source: Mapped[str] = mapped_column(String(40))          # "house_clerk" | "senate_efd"
    doc_id: Mapped[str] = mapped_column(String(120))         # source-native document id
    filed_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    url: Mapped[str | None] = mapped_column(Text, nullable=True)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    politician: Mapped[Politician] = relationship(back_populates="filings")
    trades: Mapped[list["PoliticianTrade"]] = relationship(back_populates="filing")


class PoliticianTrade(Base):
    __tablename__ = "politician_trades"

    id: Mapped[int] = mapped_column(primary_key=True)
    politician_id: Mapped[int] = mapped_column(ForeignKey("politicians.id"), index=True)
    filing_id: Mapped[int | None] = mapped_column(ForeignKey("filings.id"), nullable=True)

    ticker: Mapped[str | None] = mapped_column(String(20), index=True, nullable=True)
    asset_type: Mapped[str | None] = mapped_column(String(40), nullable=True)
    tx_type: Mapped[TxType] = mapped_column(Enum(TxType))
    amount_low: Mapped[float | None] = mapped_column(Float, nullable=True)
    amount_high: Mapped[float | None] = mapped_column(Float, nullable=True)
    tx_date: Mapped[date | None] = mapped_column(Date, nullable=True)         # when they traded
    disclosed_date: Mapped[date | None] = mapped_column(Date, nullable=True)  # when it was filed
    raw_desc: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Unique fingerprint so re-scraping never inserts duplicates.
    dedup_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    politician: Mapped[Politician] = relationship(back_populates="trades")
    filing: Mapped[Filing | None] = relationship(back_populates="trades")

    @property
    def disclosure_lag_days(self) -> int | None:
        if self.tx_date and self.disclosed_date:
            return (self.disclosed_date - self.tx_date).days
        return None


# --------------------------------------------------------------------------- #
# Personal side (synced from broker)
# --------------------------------------------------------------------------- #
class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    buying_power: Mapped[float] = mapped_column(Float)
    mode: Mapped[str] = mapped_column(String(20))  # PAPER / LIVE at time of sync
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class Position(Base):
    """Current broker positions — replaced wholesale on each sync."""

    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(primary_key=True)
    symbol: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    qty: Mapped[float] = mapped_column(Float)
    avg_entry_price: Mapped[float] = mapped_column(Float)
    market_value: Mapped[float] = mapped_column(Float)
    unrealized_pl: Mapped[float] = mapped_column(Float)
    current_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    synced_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# --------------------------------------------------------------------------- #
# Copy side
# --------------------------------------------------------------------------- #
class SizingMode(str, PyEnum):
    FIXED_NOTIONAL = "fixed_notional"  # spend a fixed $ per copied trade
    PCT_PORTFOLIO = "pct_portfolio"    # spend a % of current equity per copied trade


class CopyRule(Base):
    __tablename__ = "copy_rules"

    id: Mapped[int] = mapped_column(primary_key=True)
    # Null politician_id == applies to every followed politician.
    politician_id: Mapped[int | None] = mapped_column(
        ForeignKey("politicians.id"), nullable=True, index=True
    )
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    sizing_mode: Mapped[SizingMode] = mapped_column(
        Enum(SizingMode), default=SizingMode.FIXED_NOTIONAL
    )
    sizing_value: Mapped[float] = mapped_column(Float, default=100.0)
    min_amount_filter: Mapped[float] = mapped_column(Float, default=0.0)  # ignore tiny disclosed trades
    buy_only: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class CopySignal(Base):
    __tablename__ = "copy_signals"

    id: Mapped[int] = mapped_column(primary_key=True)
    politician_trade_id: Mapped[int] = mapped_column(
        ForeignKey("politician_trades.id"), index=True
    )
    rule_id: Mapped[int | None] = mapped_column(ForeignKey("copy_rules.id"), nullable=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    side: Mapped[TxType] = mapped_column(Enum(TxType))
    target_notional: Mapped[float] = mapped_column(Float)
    status: Mapped[SignalStatus] = mapped_column(
        Enum(SignalStatus), default=SignalStatus.PROPOSED, index=True
    )
    note: Mapped[str | None] = mapped_column(Text, nullable=True)  # e.g. why skipped
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    execution: Mapped["Execution | None"] = relationship(
        back_populates="signal", uselist=False
    )


class Execution(Base):
    __tablename__ = "executions"

    id: Mapped[int] = mapped_column(primary_key=True)
    signal_id: Mapped[int] = mapped_column(ForeignKey("copy_signals.id"), unique=True)
    alpaca_order_id: Mapped[str | None] = mapped_column(String(80), nullable=True)
    submitted_notional: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(40), default="pending")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    signal: Mapped[CopySignal] = relationship(back_populates="execution")
