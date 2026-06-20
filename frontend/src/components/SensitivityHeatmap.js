import React from "react";
import { Card } from "@/components/ui/card";

// SVG heatmap of avg walk-forward Sharpe across tau1 (rows) x tau2 (cols).
export const SensitivityHeatmap = ({ sensitivity, bestParams }) => {
  if (!sensitivity || !sensitivity.cells || !sensitivity.cells.length) {
    return (
      <Card data-testid="chart-sensitivity-container" className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))] p-3">
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide">
          Parameter Sensitivity
        </h4>
        <div className="flex h-[240px] items-center justify-center rounded-md border border-dashed border-[hsl(var(--hairline))] text-xs text-muted-foreground">
          Run Walk-Forward Optimization to populate the sensitivity map
        </div>
      </Card>
    );
  }

  const { tau1_values, tau2_values, cells } = sensitivity;
  const vals = cells.map((c) => c.sharpe).filter((v) => v !== null && v !== undefined);
  const min = Math.min(...vals, 0);
  const max = Math.max(...vals, 0);

  const colorFor = (v) => {
    if (v === null || v === undefined) return "hsl(var(--surface-2))";
    if (max === min) return "hsla(195,85%,45%,0.4)";
    const t = (v - min) / (max - min);
    // Red (low) -> amber -> green (high)
    if (t < 0.5) {
      const k = t / 0.5;
      const h = 0 + k * 45;
      return `hsla(${h},80%,50%,${0.25 + 0.55 * (1 - k) + 0.0})`;
    }
    const k = (t - 0.5) / 0.5;
    const h = 45 + k * 100;
    return `hsla(${h},75%,48%,${0.35 + 0.5 * k})`;
  };

  const lookup = {};
  cells.forEach((c) => { lookup[`${c.tau1}_${c.tau2}`] = c.sharpe; });

  const cell = 30;
  const labelW = 34;
  const labelH = 20;
  const width = labelW + tau2_values.length * cell;
  const height = labelH + tau1_values.length * cell + 16;

  const bt1 = bestParams ? Number(bestParams.toxic_continuation_threshold).toFixed(2) : null;
  const bt2 = bestParams ? Number(bestParams.toxic_reversal_threshold).toFixed(2) : null;

  return (
    <Card data-testid="chart-sensitivity-container" className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))] p-3">
      <div className="mb-2 flex items-center justify-between">
        <div>
          <h4 className="text-xs font-semibold uppercase tracking-wide">Parameter Sensitivity</h4>
          <p className="text-[10px] text-muted-foreground">
            avg walk-forward Sharpe &middot; rows &tau;1, cols &tau;2 (best over N)
          </p>
        </div>
        <div className="flex items-center gap-1 font-mono text-[10px] text-muted-foreground">
          <span>low</span>
          <span className="inline-block h-3 w-4" style={{ background: "hsla(0,80%,50%,0.6)" }} />
          <span className="inline-block h-3 w-4" style={{ background: "hsla(45,80%,50%,0.6)" }} />
          <span className="inline-block h-3 w-4" style={{ background: "hsla(145,75%,48%,0.7)" }} />
          <span>high</span>
        </div>
      </div>

      <div className="overflow-auto">
        <svg width={width} height={height} className="font-mono">
          {tau2_values.map((t2, j) => (
            <text key={`c${j}`} x={labelW + j * cell + cell / 2} y={labelH - 6}
              textAnchor="middle" fontSize={9} fill="hsl(var(--muted-foreground))">
              {t2.toFixed(2)}
            </text>
          ))}
          {tau1_values.map((t1, i) => (
            <text key={`r${i}`} x={labelW - 6} y={labelH + i * cell + cell / 2 + 3}
              textAnchor="end" fontSize={9} fill="hsl(var(--muted-foreground))">
              {t1.toFixed(2)}
            </text>
          ))}
          {tau1_values.map((t1, i) =>
            tau2_values.map((t2, j) => {
              const v = lookup[`${t1}_${t2}`];
              const isBest = bt1 === t1.toFixed(2) && bt2 === t2.toFixed(2);
              return (
                <g key={`${i}-${j}`}>
                  <rect x={labelW + j * cell} y={labelH + i * cell}
                    width={cell - 1.5} height={cell - 1.5} rx={2}
                    fill={colorFor(v)}
                    stroke={isBest ? "hsl(var(--foreground))" : "transparent"}
                    strokeWidth={isBest ? 1.5 : 0} />
                  <title>{`\u03c41=${t1.toFixed(2)} \u03c42=${t2.toFixed(2)} Sharpe=${v === null || v === undefined ? "n/a" : v.toFixed(3)}`}</title>
                </g>
              );
            })
          )}
        </svg>
      </div>
    </Card>
  );
};
