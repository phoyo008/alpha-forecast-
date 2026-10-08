import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { Signal } from "../lib/api";
import { api } from "../lib/api";
import { usd2, date } from "../lib/format";

const STATUS_CLASS: Record<string, string> = {
  proposed: "muted",
  approved: "house",
  submitted: "house",
  filled: "buy",
  rejected: "sell",
  failed: "sell",
  skipped: "exchange",
};

function SignalRow({ s }: { s: Signal }) {
  const qc = useQueryClient();
  const invalidate = () => qc.invalidateQueries();
  const approve = useMutation({ mutationFn: () => api.approveSignal(s.id), onSuccess: invalidate });
  const reject = useMutation({ mutationFn: () => api.rejectSignal(s.id), onSuccess: invalidate });

  return (
    <tr>
      <td><strong>{s.ticker}</strong></td>
      <td><span className={`badge ${s.side}`}>{s.side}</span></td>
      <td>{usd2(s.target_notional)}</td>
      <td><span className={`badge ${STATUS_CLASS[s.status] ?? "muted"}`}>{s.status}</span></td>
      <td className="muted">{s.note ?? s.execution?.error ?? s.execution?.status ?? "—"}</td>
      <td>{date(s.created_at)}</td>
      <td>
        {s.status === "proposed" ? (
          <div className="row">
            <button className="btn small" disabled={approve.isPending} onClick={() => approve.mutate()}>
              Approve
            </button>
            <button className="btn ghost small" disabled={reject.isPending} onClick={() => reject.mutate()}>
              Reject
            </button>
          </div>
        ) : (
          <span className="muted">—</span>
        )}
      </td>
    </tr>
  );
}

export default function Signals() {
  const signals = useQuery({ queryKey: ["signals"], queryFn: () => api.signals() });
  const status = useQuery({ queryKey: ["status"], queryFn: api.status });

  return (
    <div>
      <h1 className="page-title">Copy Signals</h1>
      <p className="page-sub">
        Proposed trades derived from followed politicians. Approve one to submit it to your broker.
      </p>

      {status.data && (
        <div className={`banner ${status.data.trading_mode === "LIVE" ? "danger" : ""}`}>
          Mode <strong>{status.data.trading_mode}</strong> ·{" "}
          {status.data.auto_execute ? "auto-execute ON" : "manual approval"} · orders today{" "}
          {status.data.orders_today}/{status.data.max_orders_per_day} · per-order cap{" "}
          {usd2(status.data.max_order_notional)}
        </div>
      )}

      <div className="card">
        {signals.isLoading ? (
          <p className="muted">Loading…</p>
        ) : (signals.data ?? []).length === 0 ? (
          <p className="muted">
            No signals yet. Follow a politician and add a copy rule in Settings — new disclosed buys
            will appear here.
          </p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Ticker</th><th>Side</th><th>Size</th><th>Status</th><th>Note</th><th>Created</th><th></th>
              </tr>
            </thead>
            <tbody>
              {(signals.data ?? []).map((s) => <SignalRow key={s.id} s={s} />)}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
