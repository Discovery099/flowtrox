import React, { useEffect, useRef, useState } from "react";
import { Card } from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Switch } from "@/components/ui/switch";
import { Terminal } from "lucide-react";

const stageColor = (stage) => {
  if (stage === "error") return "text-[hsl(var(--neg))]";
  if (stage === "done") return "text-[hsl(var(--pos))]";
  if (stage === "optimize" || stage === "features") return "text-[hsl(var(--info))]";
  if (stage === "warn") return "text-[hsl(var(--warn))]";
  return "text-muted-foreground";
};

export const ProgressConsole = ({ events = [], pct = 0, status = "idle", bestScore }) => {
  const [autoScroll, setAutoScroll] = useState(true);
  const endRef = useRef(null);

  useEffect(() => {
    if (autoScroll && endRef.current) {
      endRef.current.scrollIntoView({ behavior: "smooth", block: "end" });
    }
  }, [events, autoScroll]);

  return (
    <Card
      data-testid="progress-console-container"
      className="flex flex-col border-[hsl(var(--hairline))] bg-[hsl(var(--surface-0))]"
    >
      <div className="scanline flex items-center justify-between border-b border-[hsl(var(--hairline))] px-3 py-2">
        <div className="flex items-center gap-2">
          <Terminal className="h-4 w-4 text-[hsl(var(--info))]" />
          <span className="font-mono text-xs font-semibold">console</span>
          <Badge
            variant="outline"
            className="h-5 border-[hsl(var(--hairline))] font-mono text-[10px] text-muted-foreground"
          >
            {status}
          </Badge>
          {bestScore !== null && bestScore !== undefined && (
            <Badge
              variant="outline"
              className="h-5 border-[hsl(var(--info)/0.4)] font-mono text-[10px] text-[hsl(var(--info))]"
            >
              best WF Sharpe {Number(bestScore).toFixed(3)}
            </Badge>
          )}
        </div>
        <label className="flex items-center gap-2 text-[10px] text-muted-foreground">
          auto-scroll
          <Switch checked={autoScroll} onCheckedChange={setAutoScroll} />
        </label>
      </div>

      {status === "running" && (
        <div className="px-3 pt-2">
          <Progress value={pct} className="h-1.5" />
          <div className="mt-1 text-right font-mono text-[10px] text-muted-foreground">
            {pct}%
          </div>
        </div>
      )}

      <ScrollArea className="h-[200px] px-3 py-2">
        <div className="space-y-0.5 font-mono text-[11px] leading-5">
          {events.length === 0 && (
            <div className="text-muted-foreground">
              $ awaiting job&hellip; run a backtest or optimization to stream logs.
            </div>
          )}
          {events.map((e, idx) => (
            <div key={idx} className="flex gap-2">
              <span className="shrink-0 text-[hsl(var(--hairline))]">
                {e.ts ? e.ts.slice(11, 19) : "--:--:--"}
              </span>
              <span className={`shrink-0 ${stageColor(e.stage)}`}>
                [{(e.stage || "log").padEnd(8).slice(0, 8)}]
              </span>
              <span className="text-foreground/90">{e.message}</span>
            </div>
          ))}
          <div ref={endRef} />
        </div>
      </ScrollArea>
    </Card>
  );
};
