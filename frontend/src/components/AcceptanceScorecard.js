import React from "react";
import { CheckCircle2, XCircle, ShieldQuestion } from "lucide-react";
import { Card } from "@/components/ui/card";
import { fmtNum, fmtPct, CRITERIA_LABELS } from "@/lib/format";

const formatActual = (key, metrics) => {
  if (!metrics) return "\u2014";
  if (key === "win_rate" || key === "max_drawdown") return fmtPct(metrics[key], 1);
  if (key === "trades_per_day") return fmtNum(metrics[key], 2);
  return fmtNum(metrics[key], 2);
};

export const AcceptanceScorecard = ({ checks, metrics }) => {
  const allPassed = checks?.all_passed;
  const hasData = !!checks;

  return (
    <Card className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-semibold uppercase tracking-wide">
          Acceptance Scorecard
        </h3>
        {hasData ? (
          <span
            data-testid="acceptance-scorecard-verdict-chip"
            className={`inline-flex items-center gap-1.5 rounded-md border px-2.5 py-1 font-mono text-xs font-semibold ${
              allPassed
                ? "border-[hsl(var(--pos)/0.4)] bg-[hsl(var(--pos)/0.12)] text-[hsl(var(--pos))]"
                : "border-[hsl(var(--neg)/0.4)] bg-[hsl(var(--neg)/0.12)] text-[hsl(var(--neg))]"
            }`}
          >
            {allPassed ? (
              <CheckCircle2 className="h-3.5 w-3.5" />
            ) : (
              <XCircle className="h-3.5 w-3.5" />
            )}
            {allPassed ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED"}
          </span>
        ) : (
          <span className="inline-flex items-center gap-1.5 rounded-md border border-[hsl(var(--hairline))] px-2.5 py-1 font-mono text-xs text-muted-foreground">
            <ShieldQuestion className="h-3.5 w-3.5" /> NOT RUN
          </span>
        )}
      </div>

      <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 xl:grid-cols-5">
        {Object.entries(CRITERIA_LABELS).map(([ck, meta]) => {
          const passed = hasData ? checks[ck] : null;
          return (
            <div
              key={ck}
              data-testid={`acceptance-chip-${ck}`}
              className={`rounded-md border px-3 py-2 ${
                passed === null
                  ? "border-[hsl(var(--hairline))] bg-[hsl(var(--surface-0))]"
                  : passed
                  ? "border-[hsl(var(--pos)/0.35)] bg-[hsl(var(--pos)/0.08)]"
                  : "border-[hsl(var(--neg)/0.35)] bg-[hsl(var(--neg)/0.08)]"
              }`}
            >
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-medium text-muted-foreground">
                  {meta.label}
                </span>
                {passed === null ? (
                  <span className="font-mono text-[10px] text-muted-foreground">&mdash;</span>
                ) : passed ? (
                  <span className="font-mono text-[10px] font-semibold text-[hsl(var(--pos))]">
                    PASS
                  </span>
                ) : (
                  <span className="font-mono text-[10px] font-semibold text-[hsl(var(--neg))]">
                    FAIL
                  </span>
                )}
              </div>
              <div className="mt-1 font-mono text-base font-semibold tabular-nums">
                {formatActual(meta.key, metrics)}
              </div>
            </div>
          );
        })}
      </div>
    </Card>
  );
};
