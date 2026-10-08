import type { TradeFeedItem } from "../lib/api";
import { amountRange, date } from "../lib/format";

export default function TradeFeed({ trades }: { trades: TradeFeedItem[] }) {
  if (trades.length === 0) {
    return (
      <p className="muted">
        No disclosures yet. Hit <strong>Refresh disclosures</strong> to scrape the latest House &
        Senate filings.
      </p>
    );
  }
  return (
    <table>
      <thead>
        <tr>
          <th>Politician</th>
          <th>Ticker</th>
          <th>Type</th>
          <th>Amount</th>
          <th>Traded</th>
          <th>Disclosed</th>
          <th>Lag</th>
        </tr>
      </thead>
      <tbody>
        {trades.map((t) => (
          <tr key={t.id}>
            <td>
              {t.politician_name}{" "}
              <span className={`badge ${t.chamber}`}>{t.chamber}</span>
            </td>
            <td>{t.ticker ?? <span className="muted">{t.raw_desc?.slice(0, 24) ?? "—"}</span>}</td>
            <td><span className={`badge ${t.tx_type}`}>{t.tx_type}</span></td>
            <td>{amountRange(t.amount_low, t.amount_high)}</td>
            <td>{date(t.tx_date)}</td>
            <td>{date(t.disclosed_date)}</td>
            <td>
              {t.disclosure_lag_days != null ? (
                <span className={t.disclosure_lag_days > 45 ? "neg" : "muted"}>
                  {t.disclosure_lag_days}d
                </span>
              ) : (
                "—"
              )}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
