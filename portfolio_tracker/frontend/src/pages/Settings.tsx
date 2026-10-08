import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../lib/api";

function CopyRules() {
  const qc = useQueryClient();
  const rules = useQuery({ queryKey: ["rules"], queryFn: () => fetch("/api/signals/rules").then((r) => r.json()) });
  const politicians = useQuery({ queryKey: ["politicians"], queryFn: () => api.politicians() });
  const [form, setForm] = useState({ politician_id: "", sizing_mode: "fixed_notional", sizing_value: 100, buy_only: true });

  const create = useMutation({
    mutationFn: () =>
      fetch("/api/signals/rules", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          politician_id: form.politician_id ? Number(form.politician_id) : null,
          sizing_mode: form.sizing_mode,
          sizing_value: Number(form.sizing_value),
          buy_only: form.buy_only,
        }),
      }).then((r) => r.json()),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rules"] }),
  });
  const remove = useMutation({
    mutationFn: (id: number) => fetch(`/api/signals/rules/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["rules"] }),
  });

  const nameOf = (id: number | null) =>
    id == null ? "All followed" : politicians.data?.find((p) => p.id === id)?.full_name ?? `#${id}`;

  return (
    <div className="card" style={{ marginTop: 16 }}>
      <strong>Copy rules</strong>
      <p className="muted" style={{ fontSize: 13 }}>How much to buy when a followed politician trades.</p>
      <table>
        <thead><tr><th>Applies to</th><th>Sizing</th><th>Buy only</th><th></th></tr></thead>
        <tbody>
          {(rules.data ?? []).map((r: any) => (
            <tr key={r.id}>
              <td>{nameOf(r.politician_id)}</td>
              <td>{r.sizing_mode === "pct_portfolio" ? `${r.sizing_value}% of equity` : `$${r.sizing_value}`}</td>
              <td>{r.buy_only ? "yes" : "no"}</td>
              <td><button className="btn ghost small" onClick={() => remove.mutate(r.id)}>Delete</button></td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="row" style={{ marginTop: 14, flexWrap: "wrap" }}>
        <select value={form.politician_id} onChange={(e) => setForm({ ...form, politician_id: e.target.value })}>
          <option value="">All followed</option>
          {(politicians.data ?? []).filter((p) => p.followed).map((p) => (
            <option key={p.id} value={p.id}>{p.full_name}</option>
          ))}
        </select>
        <select value={form.sizing_mode} onChange={(e) => setForm({ ...form, sizing_mode: e.target.value })}>
          <option value="fixed_notional">Fixed $</option>
          <option value="pct_portfolio">% of equity</option>
        </select>
        <input
          type="number" style={{ width: 100 }} value={form.sizing_value}
          onChange={(e) => setForm({ ...form, sizing_value: Number(e.target.value) })}
        />
        <label className="row" style={{ gap: 6 }}>
          <input type="checkbox" checked={form.buy_only} onChange={(e) => setForm({ ...form, buy_only: e.target.checked })} />
          buy only
        </label>
        <button className="btn small" onClick={() => create.mutate()}>Add rule</button>
      </div>
    </div>
  );
}

export default function SettingsPage() {
  const qc = useQueryClient();
  const status = useQuery({ queryKey: ["status"], queryFn: api.status });
  const setMode = useMutation({ mutationFn: api.setMode, onSuccess: () => qc.invalidateQueries({ queryKey: ["status"] }) });
  const setLimits = useMutation({ mutationFn: api.setLimits, onSuccess: () => qc.invalidateQueries({ queryKey: ["status"] }) });
  const s = status.data;

  return (
    <div>
      <h1 className="page-title">Settings</h1>
      <p className="page-sub">Trading mode and risk limits. Live trading additionally requires environment confirmation.</p>

      <div className="card">
        <div className="row spread">
          <div>
            <strong>Trading mode</strong>
            <p className="muted" style={{ fontSize: 13, margin: "4px 0 0" }}>
              DISABLED never trades · PAPER is simulated · LIVE moves real money (env-gated).
            </p>
          </div>
          <div className="row">
            {["DISABLED", "PAPER", "LIVE"].map((m) => (
              <button
                key={m}
                className={`btn small ${s?.trading_mode === m ? "" : "ghost"}`}
                onClick={() => setMode.mutate(m)}
              >
                {m}
              </button>
            ))}
          </div>
        </div>
        {s && !s.live_enabled && s.trading_mode !== "LIVE" && (
          <p className="muted" style={{ fontSize: 13, marginTop: 12 }}>
            To enable LIVE you must set <code>TRADING_MODE=LIVE</code> and <code>LIVE_CONFIRMED=true</code> in the backend env.
          </p>
        )}
      </div>

      {s && (
        <div className="card" style={{ marginTop: 16 }}>
          <strong>Risk limits</strong>
          <div className="grid grid-3" style={{ marginTop: 14 }}>
            <label>
              <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>Max $ / order</div>
              <input
                type="number" defaultValue={s.max_order_notional}
                onBlur={(e) => setLimits.mutate({ max_order_notional: Number(e.target.value) })}
              />
            </label>
            <label>
              <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>Max % / position</div>
              <input
                type="number" defaultValue={s.max_position_pct}
                onBlur={(e) => setLimits.mutate({ max_position_pct: Number(e.target.value) })}
              />
            </label>
            <label>
              <div className="muted" style={{ fontSize: 13, marginBottom: 6 }}>Max orders / day</div>
              <input
                type="number" defaultValue={s.max_orders_per_day}
                onBlur={(e) => setLimits.mutate({ max_orders_per_day: Number(e.target.value) })}
              />
            </label>
          </div>
          <label className="row" style={{ gap: 8, marginTop: 16 }}>
            <input
              type="checkbox" checked={s.auto_execute}
              onChange={(e) => setLimits.mutate({ auto_execute: e.target.checked } as any)}
            />
            <span>Auto-execute signals (skip the manual approval queue)</span>
          </label>
        </div>
      )}

      <CopyRules />
    </div>
  );
}
