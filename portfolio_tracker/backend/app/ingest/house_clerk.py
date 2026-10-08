"""House Clerk disclosure scraper.

Data flow (all from the official site, no third-party service):
  1. Download the annual bundle  .../financial-pdfs/{YEAR}FD.zip
  2. Parse {YEAR}FD.txt (tab-delimited index of every filing that year)
  3. Keep FilingType == 'P'  (Periodic Transaction Report — the ones with trades)
  4. Download each PTR PDF     .../ptr-pdfs/{YEAR}/{DocID}.pdf
  5. Extract the transaction table with pdfplumber and normalize it

House PTRs are inconsistent (many older ones are scanned images we cannot parse);
this scraper extracts what it can and skips the rest with a log line.
"""

from __future__ import annotations

import csv
import io
import logging
import zipfile
from datetime import date

import httpx

from app.ingest.base import TradeRecord
from app.ingest.normalize import extract_ticker, parse_amount_range, parse_date, parse_tx_type
from app.models import Chamber

log = logging.getLogger(__name__)

INDEX_URL = "https://disclosures-clerk.house.gov/public_disc/financial-pdfs/{year}FD.zip"
PTR_PDF_URL = "https://disclosures-clerk.house.gov/public_disc/ptr-pdfs/{year}/{doc_id}.pdf"
_HEADERS = {"User-Agent": "portfolio-tracker/1.0 (personal disclosure research)"}


class HouseClerkSource:
    name = "house_clerk"

    def __init__(self, year: int | None = None, max_docs: int = 40, timeout: float = 30.0):
        self.year = year
        self.max_docs = max_docs
        self.timeout = timeout

    def fetch(self, since: date | None = None) -> list[TradeRecord]:
        year = self.year or (since.year if since else date.today().year)
        try:
            index = self._download_index(year)
        except Exception as exc:  # noqa: BLE001 - network/site fragility is expected
            log.warning("House index download failed for %s: %s", year, exc)
            return []

        ptrs = [row for row in index if (row.get("FilingType") or "").strip().upper() == "P"]
        log.info("House %s: %d PTR filings in index", year, len(ptrs))

        records: list[TradeRecord] = []
        with httpx.Client(timeout=self.timeout, headers=_HEADERS, follow_redirects=True) as client:
            for row in ptrs[: self.max_docs]:
                doc_id = (row.get("DocID") or "").strip()
                if not doc_id:
                    continue
                filed = parse_date(row.get("FilingDate"))
                if since and filed and filed < since:
                    continue
                name = " ".join(
                    p for p in (row.get("First"), row.get("Last")) if p
                ).strip()
                try:
                    records.extend(self._parse_ptr(client, year, doc_id, row, name, filed))
                except Exception as exc:  # noqa: BLE001
                    log.info("Skipping House PTR %s (parse failed): %s", doc_id, exc)
        log.info("House %s: extracted %d transactions", year, len(records))
        return records

    def _download_index(self, year: int) -> list[dict]:
        url = INDEX_URL.format(year=year)
        with httpx.Client(timeout=self.timeout, headers=_HEADERS, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            txt_name = next(n for n in zf.namelist() if n.lower().endswith(".txt"))
            raw = zf.read(txt_name).decode("utf-8", errors="replace")
        return list(csv.DictReader(io.StringIO(raw), delimiter="\t"))

    def _parse_ptr(
        self,
        client: httpx.Client,
        year: int,
        doc_id: str,
        row: dict,
        name: str,
        filed: date | None,
    ) -> list[TradeRecord]:
        import pdfplumber  # imported lazily so the module loads without the dep at import time

        url = PTR_PDF_URL.format(year=year, doc_id=doc_id)
        resp = client.get(url)
        resp.raise_for_status()

        out: list[TradeRecord] = []
        with pdfplumber.open(io.BytesIO(resp.content)) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables() or []:
                    out.extend(self._rows_from_table(table, row, name, filed, doc_id, url))
        return out

    def _rows_from_table(self, table, row, name, filed, doc_id, url) -> list[TradeRecord]:
        """Interpret a PTR transaction table.

        Digital House PTRs use columns roughly:
        [Owner, Asset, Transaction Type, Date, Notification Date, Amount].
        We locate columns heuristically so minor layout drift doesn't break us.
        """
        records: list[TradeRecord] = []
        if not table or len(table) < 2:
            return records
        header = [(c or "").strip().lower() for c in table[0]]

        def col(*keys: str) -> int | None:
            for i, h in enumerate(header):
                if any(k in h for k in keys):
                    return i
            return None

        asset_i = col("asset")
        type_i = col("transaction type", "type")
        date_i = col("date")
        amount_i = col("amount")
        if asset_i is None or type_i is None:
            return records

        for data in table[1:]:
            if not any(data):
                continue
            asset = (data[asset_i] if asset_i < len(data) else "") or ""
            tx_raw = (data[type_i] if type_i is not None and type_i < len(data) else "") or ""
            amount_raw = (data[amount_i] if amount_i is not None and amount_i < len(data) else "") or ""
            tx_date_raw = (data[date_i] if date_i is not None and date_i < len(data) else "") or ""
            if not asset.strip():
                continue
            low, high = parse_amount_range(amount_raw)
            records.append(
                TradeRecord(
                    source=self.name,
                    politician_name=name or f"{row.get('First','')} {row.get('Last','')}".strip(),
                    chamber=Chamber.HOUSE,
                    tx_type=parse_tx_type(tx_raw),
                    ticker=extract_ticker(asset),
                    asset_type=None,
                    amount_low=low,
                    amount_high=high,
                    tx_date=parse_date(tx_date_raw),
                    disclosed_date=filed,
                    doc_id=doc_id,
                    url=url,
                    state=(row.get("StateDst") or None),
                    raw_desc=asset.strip(),
                    external_id=None,
                )
            )
        return records


if __name__ == "__main__":  # manual smoke test: python -m app.ingest.house_clerk
    logging.basicConfig(level=logging.INFO)
    recs = HouseClerkSource(max_docs=5).fetch()
    for r in recs[:20]:
        print(r.politician_name, r.tx_type.value, r.ticker, r.amount_low, r.tx_date)
