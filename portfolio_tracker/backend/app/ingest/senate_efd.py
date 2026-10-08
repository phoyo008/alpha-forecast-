"""Senate Electronic Financial Disclosure (EFD) scraper.

The Senate site requires accepting a usage agreement before searching, which
sets a session cookie + CSRF token. Flow:
  1. GET  /search/home/            -> csrftoken cookie + form token
  2. POST /search/home/            -> accept the prohibition agreement
  3. POST /search/report/data/     -> JSON list of filings (DataTables endpoint)
  4. GET  /search/view/ptr/{uuid}/ -> HTML page; parse the transactions table

Only *electronic* PTRs expose a machine-readable table; paper filings are PDFs
and are recorded as filings without parsed transactions.
"""

from __future__ import annotations

import logging
from datetime import date

import httpx
from bs4 import BeautifulSoup

from app.ingest.base import TradeRecord
from app.ingest.normalize import extract_ticker, parse_amount_range, parse_date, parse_tx_type
from app.models import Chamber

log = logging.getLogger(__name__)

BASE = "https://efdsearch.senate.gov"
HOME = f"{BASE}/search/home/"
SEARCH_DATA = f"{BASE}/search/report/data/"
_HEADERS = {"User-Agent": "portfolio-tracker/1.0 (personal disclosure research)"}


class SenateEFDSource:
    name = "senate_efd"

    def __init__(self, max_reports: int = 40, timeout: float = 30.0):
        self.max_reports = max_reports
        self.timeout = timeout

    def fetch(self, since: date | None = None) -> list[TradeRecord]:
        try:
            with httpx.Client(
                timeout=self.timeout, headers=_HEADERS, follow_redirects=True
            ) as client:
                token = self._accept_agreement(client)
                rows = self._search(client, token, since)
                log.info("Senate EFD: %d filings returned", len(rows))
                records: list[TradeRecord] = []
                for row in rows[: self.max_reports]:
                    try:
                        records.extend(self._parse_row(client, row, since))
                    except Exception as exc:  # noqa: BLE001
                        log.info("Skipping Senate report (parse failed): %s", exc)
                log.info("Senate EFD: extracted %d transactions", len(records))
                return records
        except Exception as exc:  # noqa: BLE001
            log.warning("Senate EFD fetch failed: %s", exc)
            return []

    def _accept_agreement(self, client: httpx.Client) -> str:
        resp = client.get(HOME)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        token_el = soup.find("input", {"name": "csrfmiddlewaretoken"})
        token = token_el["value"] if token_el else client.cookies.get("csrftoken", "")
        client.post(
            HOME,
            data={"prohibition_agreement": "1", "csrfmiddlewaretoken": token},
            headers={"Referer": HOME},
        )
        return client.cookies.get("csrftoken", token)

    def _search(self, client: httpx.Client, token: str, since: date | None) -> list[list]:
        start = (since.strftime("%m/%d/%Y") if since else "")
        payload = {
            "start": "0",
            "length": str(self.max_reports),
            "report_types": "[11]",          # 11 = Periodic Transaction Report
            "filer_types": "[]",
            "submitted_start_date": start,
            "submitted_end_date": "",
            "candidate_state": "",
            "senator_state": "",
            "office_id": "",
            "first_name": "",
            "last_name": "",
            "csrfmiddlewaretoken": token,
        }
        resp = client.post(
            SEARCH_DATA,
            data=payload,
            headers={"X-CSRFToken": token, "Referer": f"{BASE}/search/"},
        )
        resp.raise_for_status()
        return resp.json().get("data", [])

    def _parse_row(self, client: httpx.Client, row: list, since: date | None) -> list[TradeRecord]:
        # Row shape: [first, last, full_name(html link), report_type(html), date_str]
        first, last = (row[0] or "").strip(), (row[1] or "").strip()
        name = f"{first} {last}".strip()
        link_html = row[2] if len(row) > 2 else ""
        filed = parse_date(row[4]) if len(row) > 4 else None
        if since and filed and filed < since:
            return []

        href = BeautifulSoup(link_html, "lxml").find("a")
        if not href or not href.get("href"):
            return []
        report_url = href["href"]
        if report_url.startswith("/"):
            report_url = BASE + report_url
        doc_id = report_url.rstrip("/").split("/")[-1]

        # Only electronic PTR pages have a parseable table.
        if "/view/ptr/" not in report_url:
            return []

        page = client.get(report_url)
        page.raise_for_status()
        return self._parse_ptr_page(page.text, name, filed, doc_id, report_url)

    def _parse_ptr_page(self, html, name, filed, doc_id, url) -> list[TradeRecord]:
        soup = BeautifulSoup(html, "lxml")
        table = soup.find("table")
        if not table:
            return []
        records: list[TradeRecord] = []
        for tr in table.find_all("tr")[1:]:
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            # Electronic PTR columns:
            # [#, Tx Date, Owner, Ticker, Asset Name, Asset Type, Type, Amount, ...]
            if len(cells) < 8:
                continue
            tx_date_raw, ticker_raw = cells[1], cells[3]
            asset_name, asset_type = cells[4], cells[5]
            tx_raw, amount_raw = cells[6], cells[7]
            low, high = parse_amount_range(amount_raw)
            ticker = None if ticker_raw in ("--", "") else ticker_raw
            records.append(
                TradeRecord(
                    source=self.name,
                    politician_name=name,
                    chamber=Chamber.SENATE,
                    tx_type=parse_tx_type(tx_raw),
                    ticker=ticker or extract_ticker(asset_name),
                    asset_type=asset_type or None,
                    amount_low=low,
                    amount_high=high,
                    tx_date=parse_date(tx_date_raw),
                    disclosed_date=filed,
                    doc_id=doc_id,
                    url=url,
                    raw_desc=asset_name,
                )
            )
        return records


if __name__ == "__main__":  # manual smoke test: python -m app.ingest.senate_efd
    logging.basicConfig(level=logging.INFO)
    recs = SenateEFDSource(max_reports=5).fetch()
    for r in recs[:20]:
        print(r.politician_name, r.tx_type.value, r.ticker, r.amount_low, r.tx_date)
