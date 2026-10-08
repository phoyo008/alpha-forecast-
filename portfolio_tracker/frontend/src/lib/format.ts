export const usd = (n: number | null | undefined) =>
  n == null ? "—" : n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 0 });

export const usd2 = (n: number | null | undefined) =>
  n == null ? "—" : n.toLocaleString("en-US", { style: "currency", currency: "USD" });

export const amountRange = (low: number | null, high: number | null) => {
  if (low == null && high == null) return "—";
  if (high == null) return `${usd(low)}+`;
  return `${usd(low)} – ${usd(high)}`;
};

export const date = (s: string | null) =>
  s ? new Date(s).toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" }) : "—";
