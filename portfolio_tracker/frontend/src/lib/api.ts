// Typed API client. All calls are same-origin relative; Vite proxies to the backend.

export type Chamber = "house" | "senate";
export type TxType = "buy" | "sell" | "exchange";
export type SignalStatus =
  | "proposed"
  | "approved"
  | "rejected"
  | "submitted"
  | "filled"
  | "failed"
  | "skipped";

export interface Politician {
  id: number;
  full_name: string;
  chamber: Chamber;
  party: string | null;
  state: string | null;
  followed: boolean;
  active: boolean;
}

export interface TradeFeedItem {
  id: number;
  politician_id: number;
  politician_name: string;
  chamber: Chamber;
  ticker: string | null;
  asset_type: string | null;
  tx_type: TxType;
  amount_low: number | null;
  amount_high: number | null;
  tx_date: string | null;
  disclosed_date: string | null;
  disclosure_lag_days: number | null;
  raw_desc: string | null;
  created_at: string;
}

export interface Position {
  symbol: string;
  qty: number;
  avg_entry_price: number;
  market_value: number;
  unrealized_pl: number;
  current_price: number | null;
}

export interface Account {
  equity: number;
  cash: number;
  buying_power: number;
  mode: string;
  synced_at: string;
}

export interface Portfolio {
  account: Account | null;
  positions: Position[];
  connected: boolean;
}

export interface Execution {
  id: number;
  alpaca_order_id: string | null;
  submitted_notional: number | null;
  status: string;
  error: string | null;
  updated_at: string;
}

export interface Signal {
  id: number;
  politician_trade_id: number;
  ticker: string;
  side: TxType;
  target_notional: number;
  status: SignalStatus;
  note: string | null;
  created_at: string;
  execution: Execution | null;
}

export interface TradingStatus {
  trading_mode: string;
  live_enabled: boolean;
  auto_execute: boolean;
  broker_connected: boolean;
  max_order_notional: number;
  max_position_pct: number;
  max_orders_per_day: number;
  orders_today: number;
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => req<{ status: string }>("/health"),

  politicians: (followedOnly = false) =>
    req<Politician[]>(`/api/politicians?followed_only=${followedOnly}`),
  setFollow: (id: number, followed: boolean) =>
    req<Politician>(`/api/politicians/${id}/follow`, {
      method: "PATCH",
      body: JSON.stringify({ followed }),
    }),

  trades: (params: { followedOnly?: boolean; politicianId?: number; limit?: number } = {}) => {
    const q = new URLSearchParams();
    if (params.followedOnly) q.set("followed_only", "true");
    if (params.politicianId) q.set("politician_id", String(params.politicianId));
    if (params.limit) q.set("limit", String(params.limit));
    return req<TradeFeedItem[]>(`/api/trades?${q.toString()}`);
  },
  refreshTrades: () => req<Record<string, number>>("/api/trades/refresh", { method: "POST" }),

  portfolio: () => req<Portfolio>("/api/portfolio"),
  syncPortfolio: () => req<Record<string, unknown>>("/api/portfolio/sync", { method: "POST" }),

  signals: (status?: SignalStatus) =>
    req<Signal[]>(`/api/signals${status ? `?status=${status}` : ""}`),
  approveSignal: (id: number) =>
    req<Signal>(`/api/signals/${id}/approve`, { method: "POST" }),
  rejectSignal: (id: number) =>
    req<Signal>(`/api/signals/${id}/reject`, { method: "POST" }),

  status: () => req<TradingStatus>("/api/settings/status"),
  setMode: (mode: string) =>
    req<TradingStatus>("/api/settings/mode", {
      method: "POST",
      body: JSON.stringify({ mode }),
    }),
  setLimits: (body: Partial<TradingStatus>) =>
    req<TradingStatus>("/api/settings/limits", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  kill: () => req<Record<string, unknown>>("/api/settings/kill", { method: "POST" }),
};
