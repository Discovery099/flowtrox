import React from "react";
import { Card } from "@/components/ui/card";
import { fmtNum, fmtPct, fmtUsd, signClass } from "@/lib/format";

const KPI = ({ label, value, cls = "", testid, sub }) => (
  <div
    data-testid={testid}
    className="rounded-md border border-[hsl(var(--hairline))] bg-[hsl(var(--surface-0))] px-3 py-2.5"
  >
    <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
    <div className={`mt-1 font-mono text-xl font-semibold tabular-nums ${cls}`}>{value}</div>
    {sub && <div className="mt-0.5 font-mono text-[10px] text-muted-foreground">{sub}</div>}
  </div>
);

export const MetricsGrid = ({ metrics }) => {
  const m = metrics || {};
  const empty = !metrics;
  const v = (x) => (empty ? "\u2014" : x);

  return (
    <Card className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))] p-4">
      <h3 className="mb-3 text-sm font-semibold uppercase tracking-wide">
        Test-Set Performance
      </h3>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-4">
        <KPI testid="metrics-grid-sharpe-value" label="Sharpe Ratio" cls={signClass(m.sharpe_ratio)}
          value={v(fmtNum(m.sharpe_ratio, 3))} />
        <KPI testid="metrics-grid-maxdd-value" label="Max Drawdown" cls="text-[hsl(var(--neg))]"
          value={v(fmtPct(m.max_drawdown, 2))}
          sub={empty ? null : `anchored ${fmtPct(m.max_drawdown_anchored, 1)} \u00b7 spec ${fmtPct(m.max_drawdown_spec, 1)}`} />
        <KPI testid="metrics-grid-winrate-value" label="Win Rate"
          value={v(fmtPct(m.win_rate, 2))} />
        <KPI testid="metrics-grid-pf-value" label="Profit Factor" cls={m.profit_factor >= 1 ? "text-[hsl(var(--pos))]" : "text-[hsl(var(--neg))]"}
          value={v(fmtNum(m.profit_factor, 3))} />
        <KPI testid="metrics-grid-tpd-value" label="Trades / Day"
          value={v(fmtNum(m.trades_per_day, 2))} />
        <KPI testid="metrics-grid-avgpnl-value" label="Avg Trade P&L" cls={signClass(m.avg_trade_pnl)}
          value={v(fmtUsd(m.avg_trade_pnl, 2))} />
        <KPI testid="metrics-grid-calmar-value" label="Calmar Ratio" cls={signClass(m.calmar_ratio)}
          value={v(fmtNum(m.calmar_ratio, 2))} />
        <KPI testid="metrics-grid-totalpnl-value" label="Total Net P&L" cls={signClass(m.total_pnl)}
          value={v(fmtUsd(m.total_pnl, 0))} />
        <KPI testid="metrics-grid-trades-value" label="Total Trades"
          value={v(fmtNum(m.total_trades, 0))}
          sub={empty ? null : `L ${fmtNum(m.long_trades, 0)} / S ${fmtNum(m.short_trades, 0)}`} />
        <KPI label="Avg Hold (bars)" value={v(fmtNum(m.avg_hold_bars, 1))}
          sub={empty ? null : `median ${fmtNum(m.median_hold_bars, 0)}`} />
        <KPI label="Sharpe t-stat" cls={signClass(m.sharpe_t_stat)}
          value={v(fmtNum(m.sharpe_t_stat, 2))}
          sub={empty ? null : `p=${fmtNum(m.sharpe_p_value, 3)} ${m.sharpe_significant_5pct ? "sig" : "ns"}`} />
        <KPI label="WinRate t-stat" cls={signClass(m.winrate_t_stat)}
          value={v(fmtNum(m.winrate_t_stat, 2))}
          sub={empty ? null : `p=${fmtNum(m.winrate_p_value, 3)} ${m.winrate_significant_5pct ? "sig" : "ns"}`} />
      </div>

      <h3 className="mb-2 mt-4 text-sm font-semibold uppercase tracking-wide">
        Cost Attribution
      </h3>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4">
        <KPI label="Total Slippage" cls="text-[hsl(var(--neg))]" value={v(fmtUsd(m.total_slippage_cost, 0))} />
        <KPI label="Total Commission" cls="text-[hsl(var(--neg))]" value={v(fmtUsd(m.total_commission, 0))} />
        <KPI label="Cost / Trade" value={v(fmtUsd(m.cost_per_trade, 2))} />
        <KPI label="Cost % of Gross" value={v(fmtPct(m.cost_pct_of_gross, 1))} />
      </div>
    </Card>
  );
};
