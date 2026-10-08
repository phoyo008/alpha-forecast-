# Portfolio Tracker

Track famous politicians' stock portfolios from their public STOCK Act
disclosures, sync your own broker portfolio, and generate (optionally
auto-executed) **copy-trade** signals — with real safety rails.

> ⚠️ **Reality check:** Congressional trades are disclosed up to **45 days** after
> they happen. Copying them is inherently lagged — this is *not* front-running,
> and past disclosure-following performance is no guarantee of anything. Live
> auto-trading moves real money; you are responsible for outcomes. Validate
> everything in **paper mode** first.

## What it does

- **Ingestion** — scrapes official sources directly (no paid API):
  - House Clerk annual disclosure ZIP → PTR PDFs
  - Senate EFD electronic Periodic Transaction Reports
- **Portfolio sync** — pulls your account + positions from **Alpaca**.
- **Copy engine** — new disclosed trades from *followed* politicians become sized
  **signals**; you approve them (or enable auto-execute).
- **Safety rails** — trading modes (`DISABLED`/`PAPER`/`LIVE`), per-order and
  per-position caps, a daily order cap, a symbol denylist, and a **kill switch**.

## Architecture

```
backend/   FastAPI + SQLAlchemy + APScheduler   (Python)
frontend/  React + Vite + TypeScript            (dashboard)
```

Backend layout: `ingest/` (scrapers + pipeline), `broker/` (Alpaca + safety),
`strategy/` (copy engine), `api/` (routers). See the plan in the PR description.

## Quick start

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # safe defaults: TRADING_MODE=DISABLED
uvicorn app.main:app --reload # serves http://localhost:8000
```

Tables are created automatically on startup (SQLite by default) and a starter set
of politicians is seeded. Open http://localhost:8000/docs for the API.

### Frontend

```bash
cd frontend
npm install
npm run dev                   # http://localhost:5173 (proxies /api to :8000)
```

## Enabling trading (do this in order)

1. Get **paper** API keys from [Alpaca](https://alpaca.markets) and put them in
   `backend/.env` (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`).
2. Set `TRADING_MODE=PAPER`. Sync your portfolio from the Dashboard.
3. On the **Politicians** page, follow the people you want to copy.
4. In **Settings**, add a copy rule (fixed $ or % of equity) and set risk limits.
5. Refresh disclosures → new buys appear as **signals**. Approve them to place
   **paper** orders.
6. Only after you're satisfied, going **LIVE** requires *both*
   `TRADING_MODE=LIVE` **and** `LIVE_CONFIRMED=true` in the env. The UI alone
   cannot enable live trading.

The **kill switch** (sidebar) cancels open orders and forces `DISABLED` instantly.

## Tests

```bash
cd backend && python -m pytest        # parsers, safety limits, copy-engine sizing, dedup
cd frontend && npm run build          # type-checks the UI
```

## Manual scraper smoke tests

```bash
cd backend
python -m app.ingest.house_clerk      # prints a few parsed House transactions
python -m app.ingest.senate_efd       # prints a few parsed Senate transactions
```

Scrapers are best-effort against live government sites; parse failures are logged
and skipped rather than crashing the run.
