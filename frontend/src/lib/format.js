// Numeric + semantic formatting helpers for the quant terminal.

export const fmtNum = (v, d = 2) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "\u2014";
  if (!Number.isFinite(v)) return v > 0 ? "\u221e" : "-\u221e";
  return Number(v).toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
};

export const fmtPct = (v, d = 2) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "\u2014";
  return `${(v * 100).toFixed(d)}%`;
};

export const fmtUsd = (v, d = 2) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "\u2014";
  const sign = v < 0 ? "-" : "";
  return `${sign}$${Math.abs(v).toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  })}`;
};

export const signClass = (v) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "text-muted-foreground";
  if (v > 0) return "text-[hsl(var(--pos))]";
  if (v < 0) return "text-[hsl(var(--neg))]";
  return "text-muted-foreground";
};

export const CRITERIA_LABELS = {
  "sharpe_gt_1.0": { label: "Sharpe > 1.0", key: "sharpe_ratio" },
  winrate_gt_54pct: { label: "Win Rate > 54%", key: "win_rate" },
  trades_per_day_gt_5: { label: "Trades/Day > 5", key: "trades_per_day" },
  mdd_lt_15pct: { label: "Max DD < 15%", key: "max_drawdown" },
  "profit_factor_gt_1.3": { label: "Profit Factor > 1.3", key: "profit_factor" },
};
