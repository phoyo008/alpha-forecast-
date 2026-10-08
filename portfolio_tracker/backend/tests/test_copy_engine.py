from datetime import date

from app.ingest.base import TradeRecord
from app.ingest.pipeline import persist_records
from app.models import (
    AccountSnapshot,
    Chamber,
    CopyRule,
    Politician,
    SignalStatus,
    SizingMode,
    TxType,
)
from app.strategy.copy_engine import _target_notional, generate_signals_for_trades


def test_fixed_notional_sizing():
    rule = CopyRule(sizing_mode=SizingMode.FIXED_NOTIONAL, sizing_value=250)
    assert _target_notional(rule, equity=10000) == 250


def test_pct_portfolio_sizing():
    rule = CopyRule(sizing_mode=SizingMode.PCT_PORTFOLIO, sizing_value=2)
    assert _target_notional(rule, equity=10000) == 200  # 2% of 10k


def test_signal_generated_only_for_followed(db):
    followed = Politician(full_name="Nancy Pelosi", chamber=Chamber.HOUSE, followed=True)
    ignored = Politician(full_name="Someone Else", chamber=Chamber.HOUSE, followed=False)
    db.add_all([followed, ignored])
    db.add(AccountSnapshot(equity=100000, cash=100000, buying_power=100000, mode="PAPER"))
    db.add(CopyRule(politician_id=None, sizing_mode=SizingMode.FIXED_NOTIONAL, sizing_value=100))
    db.commit()

    recs = [
        TradeRecord(
            source="test", politician_name="Nancy Pelosi", chamber=Chamber.HOUSE,
            tx_type=TxType.BUY, ticker="AAPL", tx_date=date(2026, 1, 1),
            disclosed_date=date(2026, 1, 10), doc_id="d1",
        ),
        TradeRecord(
            source="test", politician_name="Someone Else", chamber=Chamber.HOUSE,
            tx_type=TxType.BUY, ticker="MSFT", tx_date=date(2026, 1, 1),
            disclosed_date=date(2026, 1, 10), doc_id="d2",
        ),
    ]
    new_trades = persist_records(db, recs)
    assert len(new_trades) == 2

    count = generate_signals_for_trades(db, new_trades)
    assert count == 1  # only the followed politician's trade

    from app.models import CopySignal

    sig = db.query(CopySignal).one()
    assert sig.ticker == "AAPL"
    assert sig.status == SignalStatus.PROPOSED  # auto_execute is off by default


def test_dedup_on_reingest(db):
    db.add(Politician(full_name="Nancy Pelosi", chamber=Chamber.HOUSE, followed=True))
    db.commit()
    rec = TradeRecord(
        source="test", politician_name="Nancy Pelosi", chamber=Chamber.HOUSE,
        tx_type=TxType.BUY, ticker="AAPL", tx_date=date(2026, 1, 1), doc_id="d1",
    )
    assert len(persist_records(db, [rec])) == 1
    assert len(persist_records(db, [rec])) == 0  # same fingerprint -> skipped
