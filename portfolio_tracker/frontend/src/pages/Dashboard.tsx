import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";
import { usd, usd2 } from "../lib/format";
import TradeFeed from "../components/TradeFeed";

function Stat({ label, value, cls }: { label: string; value: string; cls?: string }) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className={`value ${cls ?? ""}`}>{value}</div>
    </div>
  );
}

export default function Dashboard() {
  const qc = useQueryClient();
  const portfolio = useQuery({ queryKey: ["portfolio"], queryFn: api.portfolio });
  const trades = useQuery({ queryKey: ["trades", "feed"], queryFn: () => api.trades({ limit: 25 }) });

  const sync = useMutation({
    mutationFn: api.syncPortfolio,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["portfolio"] }),
  });
  const refresh = useMutation({
    mutationFn: api.refreshTrades,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["trades"] }),
  });

  const acct = portfolio.data?.account;
  const positions = portfolio.data?.positions ?? [];
  const totalPl = positions.reduce((s, p) => s + p.unrealized_pl, 0);

  return (
    <div>
      <h1 className="page-title">Dashboard</h1>
      <p className="page-sub">Your portfolio and the latest disclosed politician trades.</p>

      <div className="banner">
        ⏱ Congressional trades are disclosed up to <strong>45 days</strong> after they happen. Copy
        signals are inherently lagged — this is not front-running.
      </div>

      {!portfolio.data?.connected && (
        <div className="banner danger">
          Broker not connected. Add Alpaca paper keys to <code>backend/.env</code> and set a trading
          mode to sync your portfolio.
        </div>
      )}

      <div className="grid grid-4" style={{ marginBottom: 24 }}>
        <Stat label="Equity" value={usd(acct?.equity)} />
        <Stat label="Cash" value={usd(acct?.cash)} />
        <Stat
          label="Unrealized P/L"
          value={usd2(totalPl)}
          cls={totalPl >= 0 ? "pos" : "neg"}
        />
        <Stat label="Positions" value={String(positions.length)} />
      </div>

      <div className="card" style={{ marginBottom: 24 }}>
        <div className="row spread" style={{ marginBottom: 12 }}>
          <strong>My positions {acct ? `(${acct.mode})` : ""}</strong>
          <button className="btn ghost small" disabled={sync.isPending} onClick={() => sync.mutate()}>
            {sync.isPending ? "Syncing…" : "Sync portfolio"}
          </button>
        </div>
        {positions.length === 0 ? (
          <p className="muted">No positions synced yet.</p>
        ) : (
          <table>
            <thead>
              <tr><th>Symbol</th><th>Qty</th><th>Avg</th><th>Price</th><th>Value</th><th>P/L</th></tr>
            </thead>
            <tbody>
              {positions.map((p) => (
                <tr key={p.symbol}>
                  <td><strong>{p.symbol}</strong></td>
                  <td>{p.qty}</td>
                  <td>{usd2(p.avg_entry_price)}</td>
                  <td>{usd2(p.current_price)}</td>
                  <td>{usd2(p.market_value)}</td>
                  <td className={p.unrealized_pl >= 0 ? "pos" : "neg"}>{usd2(p.unrealized_pl)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <div className="row spread" style={{ marginBottom: 12 }}>
          <strong>Latest disclosed trades</strong>
          <button className="btn ghost small" disabled={refresh.isPending} onClick={() => refresh.mutate()}>
            {refresh.isPending ? "Scraping…" : "Refresh disclosures"}
          </button>
        </div>
        {trades.isLoading ? <p className="muted">Loading…</p> : <TradeFeed trades={trades.data ?? []} />}
      </div>
    </div>
  );
}
