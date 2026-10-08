from app.broker.safety import check_order
from app.config import Settings, TradingMode
from app.models import AccountSnapshot


def _settings(**kw):
    base = dict(
        trading_mode=TradingMode.PAPER,
        live_confirmed=False,
        max_order_notional=500.0,
        max_position_pct=5.0,
        max_orders_per_day=10,
        symbol_denylist=[],
    )
    base.update(kw)
    return Settings(**base)


def test_disabled_blocks(db):
    r = check_order(db, "AAPL", 100, _settings(trading_mode=TradingMode.DISABLED))
    assert not r.ok and "DISABLED" in r.reason


def test_over_notional_cap(db):
    r = check_order(db, "AAPL", 600, _settings())
    assert not r.ok and "per-order cap" in r.reason


def test_denylist(db):
    r = check_order(db, "GME", 100, _settings(symbol_denylist=["GME"]))
    assert not r.ok and "denylist" in r.reason


def test_position_pct_cap(db):
    db.add(AccountSnapshot(equity=1000, cash=1000, buying_power=1000, mode="PAPER"))
    db.commit()
    # 100 / 1000 = 10% > 5% cap
    r = check_order(db, "AAPL", 100, _settings())
    assert not r.ok and "equity" in r.reason


def test_ok(db):
    db.add(AccountSnapshot(equity=100000, cash=100000, buying_power=100000, mode="PAPER"))
    db.commit()
    r = check_order(db, "AAPL", 100, _settings())
    assert r.ok
