import React from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  Cell,
} from "recharts";
import { Card } from "@/components/ui/card";

const AXIS = { fontSize: 10, fontFamily: "var(--font-mono)", fill: "hsl(var(--muted-foreground))" };
const GRID = "hsla(220,14%,30%,0.35)";

const TT = ({ active, payload, label, fmt }) => {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="rounded-md border border-[hsl(var(--hairline))] bg-[hsl(var(--surface-1))] px-2.5 py-1.5 font-mono text-[11px] shadow-lg">
      <div className="mb-1 text-muted-foreground">{label}</div>
      {payload.map((p, i) => (
        <div key={i} className="flex items-center justify-between gap-3">
          <span style={{ color: p.color || p.stroke || p.fill }}>{p.name}</span>
          <span className="tabular-nums">{fmt ? fmt(p.value) : p.value}</span>
        </div>
      ))}
    </div>
  );
};

export const ChartFrame = ({ title, subtitle, children, testid, right }) => (
  <Card data-testid={testid} className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))] p-3">
    <div className="mb-2 flex items-center justify-between">
      <div>
        <h4 className="text-xs font-semibold uppercase tracking-wide">{title}</h4>
        {subtitle && <p className="text-[10px] text-muted-foreground">{subtitle}</p>}
      </div>
      {right}
    </div>
    <div className="h-[240px] w-full">{children}</div>
  </Card>
);

const EmptyHint = ({ msg }) => (
  <div className="flex h-full items-center justify-center rounded-md border border-dashed border-[hsl(var(--hairline))] text-xs text-muted-foreground">
    {msg || "Run a backtest to populate this chart"}
  </div>
);

export const EquityChart = ({ data }) => (
  <ChartFrame title="Equity Curve" subtitle="Cumulative net P&L (USD)" testid="chart-equity-curve-container">
    {!data || !data.length ? <EmptyHint /> : (
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
          <XAxis dataKey="t" tick={AXIS} minTickGap={48} tickLine={false} axisLine={{ stroke: GRID }} />
          <YAxis tick={AXIS} width={56} tickLine={false} axisLine={{ stroke: GRID }}
            tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
          <Tooltip content={<TT fmt={(v) => `$${Number(v).toLocaleString()}`} />} />
          <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" />
          <Line type="monotone" dataKey="equity" name="equity" stroke="hsl(var(--info))"
            strokeWidth={1.6} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    )}
  </ChartFrame>
);

export const DrawdownChart = ({ data }) => (
  <ChartFrame title="Drawdown (Underwater)" subtitle="% from running equity peak" testid="chart-drawdown-container">
    {!data || !data.length ? <EmptyHint /> : (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
          <defs>
            <linearGradient id="ddGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="hsl(var(--neg))" stopOpacity={0.05} />
              <stop offset="100%" stopColor="hsl(var(--neg))" stopOpacity={0.5} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
          <XAxis dataKey="t" tick={AXIS} minTickGap={48} tickLine={false} axisLine={{ stroke: GRID }} />
          <YAxis tick={AXIS} width={44} tickLine={false} axisLine={{ stroke: GRID }}
            tickFormatter={(v) => `${v.toFixed(0)}%`} />
          <Tooltip content={<TT fmt={(v) => `${Number(v).toFixed(2)}%`} />} />
          <Area type="monotone" dataKey="dd" name="drawdown" stroke="hsl(var(--neg))"
            strokeWidth={1} fill="url(#ddGrad)" />
        </AreaChart>
      </ResponsiveContainer>
    )}
  </ChartFrame>
);

export const MonthlyReturnsChart = ({ data }) => (
  <ChartFrame title="Monthly Returns" subtitle="% of reference account / month" testid="chart-monthly-returns-container">
    {!data || !data.length ? <EmptyHint /> : (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
          <XAxis dataKey="month" tick={AXIS} minTickGap={24} tickLine={false} axisLine={{ stroke: GRID }} />
          <YAxis tick={AXIS} width={44} tickLine={false} axisLine={{ stroke: GRID }}
            tickFormatter={(v) => `${v.toFixed(0)}%`} />
          <Tooltip content={<TT fmt={(v) => `${Number(v).toFixed(2)}%`} />} cursor={{ fill: "hsla(220,14%,30%,0.2)" }} />
          <ReferenceLine y={0} stroke="hsl(var(--muted-foreground))" />
          <Bar dataKey="ret" name="return">
            {data.map((d, i) => (
              <Cell key={i} fill={d.ret >= 0 ? "hsl(var(--pos))" : "hsl(var(--neg))"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    )}
  </ChartFrame>
);

export const PnlHistogram = ({ data }) => (
  <ChartFrame title="Per-Trade P&L Distribution" subtitle="net P&L per trade (USD)" testid="chart-pnl-histogram-container">
    {!data || !data.length ? <EmptyHint msg="No trades to plot" /> : (
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
          <XAxis dataKey="bin" tick={AXIS} minTickGap={28} tickLine={false} axisLine={{ stroke: GRID }}
            tickFormatter={(v) => `${v}`} />
          <YAxis tick={AXIS} width={36} tickLine={false} axisLine={{ stroke: GRID }} />
          <Tooltip content={<TT />} cursor={{ fill: "hsla(220,14%,30%,0.2)" }} />
          <ReferenceLine x={0} stroke="hsl(var(--muted-foreground))" />
          <Bar dataKey="count" name="trades">
            {data.map((d, i) => (
              <Cell key={i} fill={d.sign === "pos" ? "hsl(var(--pos))" : "hsl(var(--neg))"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    )}
  </ChartFrame>
);

export const RegimeChart = ({ data }) => (
  <ChartFrame title="HMM Regime Posteriors" subtitle="P(state) over test period" testid="chart-regime-posteriors-container">
    {!data || !data.length ? <EmptyHint /> : (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} stackOffset="expand" margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
          <XAxis dataKey="t" tick={AXIS} minTickGap={48} tickLine={false} axisLine={{ stroke: GRID }} />
          <YAxis tick={AXIS} width={36} tickLine={false} axisLine={{ stroke: GRID }}
            tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
          <Tooltip content={<TT fmt={(v) => Number(v).toFixed(3)} />} />
          <Area type="monotone" dataKey="normal" name="normal" stackId="1"
            stroke="hsla(215,12%,70%,0.8)" fill="hsla(215,12%,70%,0.45)" />
          <Area type="monotone" dataKey="continuation" name="toxic-cont" stackId="1"
            stroke="hsl(var(--pos))" fill="hsla(145,70%,45%,0.5)" />
          <Area type="monotone" dataKey="reversal" name="toxic-rev" stackId="1"
            stroke="hsl(var(--warn))" fill="hsla(45,90%,55%,0.5)" />
        </AreaChart>
      </ResponsiveContainer>
    )}
  </ChartFrame>
);

export const VolRegimeChart = ({ data }) => (
  <ChartFrame title="Volatility Regime" subtitle="0=low 1=medium 2=high (HAR forecast)" testid="chart-vol-regime-container">
    {!data || !data.length ? <EmptyHint /> : (
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
          <XAxis dataKey="t" tick={AXIS} minTickGap={48} tickLine={false} axisLine={{ stroke: GRID }} />
          <YAxis tick={AXIS} width={28} domain={[0, 2]} ticks={[0, 1, 2]} tickLine={false} axisLine={{ stroke: GRID }} />
          <Tooltip content={<TT />} />
          <Area type="stepAfter" dataKey="regime" name="vol-regime" stroke="hsl(var(--info))"
            strokeWidth={1} fill="hsla(195,85%,45%,0.18)" />
        </AreaChart>
      </ResponsiveContainer>
    )}
  </ChartFrame>
);
