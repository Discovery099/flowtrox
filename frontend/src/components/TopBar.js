import React from "react";
import { Activity, Download, Cpu, Database } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Button } from "@/components/ui/button";
import { downloadUrl } from "@/lib/api";

export const TopBar = ({ instrument, jobStatus, lastRunAt, runId }) => {
  const statusMap = {
    idle: { label: "IDLE", cls: "bg-[hsl(var(--muted))] text-muted-foreground border-[hsl(var(--hairline))]" },
    running: { label: "RUNNING", cls: "bg-[hsl(var(--info)/0.12)] text-[hsl(var(--info))] border-[hsl(var(--info)/0.4)]" },
    done: { label: "DONE", cls: "bg-[hsl(var(--pos)/0.12)] text-[hsl(var(--pos))] border-[hsl(var(--pos)/0.4)]" },
    failed: { label: "FAILED", cls: "bg-[hsl(var(--neg)/0.12)] text-[hsl(var(--neg))] border-[hsl(var(--neg)/0.4)]" },
  };
  const st = statusMap[jobStatus] || statusMap.idle;

  return (
    <header className="relative z-10 flex items-center justify-between gap-4 border-b border-[hsl(var(--hairline))] bg-[hsl(var(--surface-1)/0.6)] px-4 py-3 backdrop-blur">
      <div className="flex items-center gap-3">
        <div className="flex h-9 w-9 items-center justify-center rounded-md border border-[hsl(var(--info)/0.35)] bg-[hsl(var(--info)/0.1)]">
          <Activity className="h-5 w-5 text-[hsl(var(--info))]" />
        </div>
        <div className="leading-tight">
          <div className="font-mono text-sm font-semibold tracking-tight">
            FLOWTOX_REGIME_01
          </div>
          <div className="text-[11px] text-muted-foreground">
            Adaptive Flow-Toxicity &middot; Regime-Aware Sizing
          </div>
        </div>
      </div>

      <div className="flex items-center gap-2">
        <Badge variant="outline" className="gap-1 border-[hsl(var(--hairline))] font-mono text-[11px]">
          <Database className="h-3 w-3" /> {instrument || "ES"}
        </Badge>
        {lastRunAt && (
          <span className="hidden font-mono text-[11px] text-muted-foreground sm:inline">
            last run {lastRunAt}
          </span>
        )}
        <Badge
          data-testid="top-bar-status-pill"
          variant="outline"
          className={`gap-1 font-mono text-[11px] ${st.cls}`}
        >
          {jobStatus === "running" && <span className="blink">&bull;</span>}
          {st.label}
        </Badge>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              data-testid="top-bar-download-menu"
              variant="outline"
              size="sm"
              disabled={!runId}
              className="gap-1 border-[hsl(var(--hairline))]"
            >
              <Download className="h-4 w-4" /> Export
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="font-mono text-xs">
            <DropdownMenuItem asChild>
              <a href={runId ? downloadUrl(runId, "metrics") : "#"} download>
                metrics.json
              </a>
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <a href={runId ? downloadUrl(runId, "trades") : "#"} download>
                trades.csv
              </a>
            </DropdownMenuItem>
            <DropdownMenuItem asChild>
              <a href={runId ? downloadUrl(runId, "equity") : "#"} download>
                equity.csv
              </a>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
};
