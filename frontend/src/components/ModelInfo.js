import React from "react";
import { Card } from "@/components/ui/card";
import { fmtNum } from "@/lib/format";

const Row = ({ k, v, mono = true }) => (
  <div className="flex items-center justify-between border-b border-[hsl(var(--hairline)/0.5)] py-1.5">
    <span className="text-[11px] text-muted-foreground">{k}</span>
    <span className={`text-xs ${mono ? "font-mono tabular-nums" : ""}`}>{v}</span>
  </div>
);

export const ModelInfo = ({ modelInfo, bestParams, walkForward }) => {
  const mi = modelInfo;
  if (!mi) {
    return (
      <Card className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))] p-4">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide">Fitted Models</h3>
        <div className="flex h-32 items-center justify-center text-xs text-muted-foreground">
          Run a backtest to fit and display models
        </div>
      </Card>
    );
  }
  const har = mi.har_params || {};
  const gmm = mi.gmm_weights || [];
  const vt = mi.vol_thresholds || {};

  return (
    <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-3">
      <Card className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))] p-4">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide">HAR-RV Model</h3>
        <Row k="intercept (c)" v={har.intercept?.toExponential(3)} />
        <Row k={"\u03b2 daily"} v={fmtNum(har.beta_d, 4)} />
        <Row k={"\u03b2 weekly"} v={fmtNum(har.beta_w, 4)} />
        <Row k={"\u03b2 monthly"} v={fmtNum(har.beta_m, 4)} />
        <Row k={"R\u00b2"} v={fmtNum(har.r_squared, 4)} />
      </Card>

      <Card className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))] p-4">
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-semibold uppercase tracking-wide">Regime Model (3-state)</h3>
          <span className="rounded border border-[hsl(var(--info)/0.4)] bg-[hsl(var(--info)/0.1)] px-1.5 py-0.5 font-mono text-[10px] uppercase text-[hsl(var(--info))]">
            {(mi.regime_model || "gmm").toUpperCase()}
          </span>
        </div>
        <Row k="normal weight" v={fmtNum(gmm[0], 3)} />
        <Row k="toxic-cont weight" v={fmtNum(gmm[1], 3)} />
        <Row k="toxic-rev weight" v={fmtNum(gmm[2], 3)} />
        <Row k="vol p33 thresh" v={vt.p33?.toExponential(2)} />
        <Row k="vol p67 thresh" v={vt.p67?.toExponential(2)} />
        {mi.transition_matrix && (
          <div className="mt-3">
            <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">
              Transition matrix P(t&rarr;t+1)
            </div>
            <table className="w-full border-collapse font-mono text-[11px] tabular-nums">
              <tbody>
                {mi.transition_matrix.map((row, i) => (
                  <tr key={i}>
                    {row.map((cell, j) => (
                      <td
                        key={j}
                        className={`border border-[hsl(var(--hairline))] px-1.5 py-1 text-center ${
                          i === j ? "text-[hsl(var(--info))]" : "text-muted-foreground"
                        }`}
                      >
                        {fmtNum(cell, 2)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-1 text-[10px] text-muted-foreground">
              rows/cols: normal, toxic-cont, toxic-rev &middot; diagonal = persistence
            </div>
          </div>
        )}
      </Card>

      <Card className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))] p-4">
        <h3 className="mb-2 text-sm font-semibold uppercase tracking-wide">Run Configuration</h3>
        {bestParams ? (
          <>
            <Row k={"\u03c41 (toxic-cont)"} v={fmtNum(bestParams.toxic_continuation_threshold, 2)} />
            <Row k={"\u03c42 (toxic-rev)"} v={fmtNum(bestParams.toxic_reversal_threshold, 2)} />
            <Row k="N (max hold)" v={bestParams.max_hold_bars} />
            {walkForward && <Row k="WF Sharpe (in-sample)" v={fmtNum(walkForward.sharpe, 3)} />}
            {walkForward && <Row k="WF windows" v={walkForward.windows} />}
          </>
        ) : (
          <div className="py-2 text-[11px] text-muted-foreground">
            Using manual params (no optimization run yet).
          </div>
        )}
        <Row k="train rows" v={fmtNum(mi.train_rows, 0)} />
        <Row k="test rows" v={fmtNum(mi.test_rows, 0)} />
        <Row k="test days" v={fmtNum(mi.num_test_days, 0)} />
      </Card>
    </div>
  );
};
