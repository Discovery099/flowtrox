import React from "react";
import { Card } from "@/components/ui/card";
import { ShieldAlert, ShieldCheck, Shield, Info } from "lucide-react";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { fmtNum } from "@/lib/format";

const pct = (v) => (v === null || v === undefined ? "\u2014" : `${(v * 100).toFixed(1)}%`);

const InfoTip = ({ text }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="cursor-help text-muted-foreground">
          <Info className="h-3 w-3" />
        </span>
      </TooltipTrigger>
      <TooltipContent className="max-w-[260px] text-xs leading-snug">{text}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

const Stat = ({ label, value, tip, tone = "neutral", testid, sub }) => {
  const toneCls =
    tone === "good"
      ? "text-[hsl(var(--pos))]"
      : tone === "bad"
      ? "text-[hsl(var(--neg))]"
      : tone === "warn"
      ? "text-[hsl(var(--warn))]"
      : "text-foreground";
  return (
    <div
      data-testid={testid}
      className="rounded-md border border-[hsl(var(--hairline))] bg-[hsl(var(--surface-0))] px-3 py-2.5"
    >
      <div className="flex items-center gap-1 text-[10px] uppercase tracking-wide text-muted-foreground">
        {label} {tip && <InfoTip text={tip} />}
      </div>
      <div className={`mt-1 font-mono text-xl font-semibold tabular-nums ${toneCls}`}>{value}</div>
      {sub && <div className="mt-0.5 font-mono text-[10px] text-muted-foreground">{sub}</div>}
    </div>
  );
};

export const RobustnessPanel = ({ diagnostics }) => {
  const d = diagnostics || null;

  // Verdict logic (only meaningful when PBO/DSR available, i.e. after optimization).
  let verdict = null;
  if (d && (d.pbo !== undefined && d.pbo !== null)) {
    const pbo = d.pbo;
    const dsr = d.dsr;
    if (pbo <= 0.3 && (dsr ?? 0) >= 0.95) verdict = { label: "ROBUST", tone: "good", Icon: ShieldCheck };
    else if (pbo >= 0.5 || (dsr ?? 1) < 0.6) verdict = { label: "HIGH OVERFITTING RISK", tone: "bad", Icon: ShieldAlert };
    else verdict = { label: "MARGINAL", tone: "warn", Icon: Shield };
  }

  const pboTone = d?.pbo == null ? "neutral" : d.pbo <= 0.3 ? "good" : d.pbo >= 0.5 ? "bad" : "warn";
  const dsrTone = d?.dsr == null ? "neutral" : d.dsr >= 0.95 ? "good" : d.dsr < 0.6 ? "bad" : "warn";
  const psrTone = d?.psr == null ? "neutral" : d.psr >= 0.95 ? "good" : d.psr < 0.6 ? "bad" : "warn";

  return (
    <Card data-testid="robustness-panel" className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))] p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold uppercase tracking-wide">Robustness / Overfitting</h3>
          <InfoTip text="Quantifies how much of the in-sample edge is likely overfitting. PBO and Deflated Sharpe require a Walk-Forward Optimization run." />
        </div>
        {verdict && (
          <span
            data-testid="robustness-verdict"
            className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 font-mono text-xs font-semibold ${
              verdict.tone === "good"
                ? "border-[hsl(var(--pos)/0.4)] bg-[hsl(var(--pos)/0.12)] text-[hsl(var(--pos))]"
                : verdict.tone === "bad"
                ? "border-[hsl(var(--neg)/0.4)] bg-[hsl(var(--neg)/0.12)] text-[hsl(var(--neg))]"
                : "border-[hsl(var(--warn)/0.4)] bg-[hsl(var(--warn)/0.12)] text-[hsl(var(--warn))]"
            }`}
          >
            <verdict.Icon className="h-3.5 w-3.5" />
            {verdict.label}
          </span>
        )}
      </div>

      {!d ? (
        <div className="flex h-28 items-center justify-center text-xs text-muted-foreground">
          Run a backtest to compute robustness statistics
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-2 md:grid-cols-3 xl:grid-cols-5">
          <Stat
            testid="robustness-pbo"
            label="PBO"
            tip="Probability of Backtest Overfitting (CSCV). Fraction of data splits where the in-sample best params rank below median out-of-sample. <30% good, >50% bad. Requires optimization."
            value={pct(d.pbo)}
            tone={pboTone}
            sub={d.n_splits ? `${d.n_splits} CSCV splits` : "run optimization"}
          />
          <Stat
            testid="robustness-dsr"
            label="Deflated Sharpe"
            tip="Probability the true Sharpe > 0 after deflating for the number of trials tested (multiple-testing correction). >95% strong. Requires optimization."
            value={pct(d.dsr)}
            tone={dsrTone}
            sub={d.n_trials ? `${d.n_trials} trials` : "run optimization"}
          />
          <Stat
            testid="robustness-psr"
            label="Prob. Sharpe (PSR)"
            tip="Probabilistic Sharpe Ratio: P(true Sharpe > 0) adjusting for non-normal returns and sample length."
            value={pct(d.psr)}
            tone={psrTone}
          />
          <Stat
            testid="robustness-sharpe-ci"
            label="Sharpe 90% CI"
            tip="Bootstrap 5-95% confidence interval for the annualized Sharpe ratio."
            value={`${fmtNum(d.sharpe_ci_low, 2)} \u2192 ${fmtNum(d.sharpe_ci_high, 2)}`}
          />
          <Stat
            testid="robustness-boot-p"
            label="Bootstrap P(no edge)"
            tip="Bootstrap probability that the Sharpe ratio is \u2264 0 (i.e. no real edge). Lower is better."
            value={pct(d.bootstrap_p_value)}
            tone={d?.bootstrap_p_value == null ? "neutral" : d.bootstrap_p_value <= 0.05 ? "good" : d.bootstrap_p_value >= 0.5 ? "bad" : "warn"}
          />
        </div>
      )}
    </Card>
  );
};
