import React, { useMemo, useState } from "react";
import { Download, ArrowUpDown, Search } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { fmtNum, fmtUsd, signClass } from "@/lib/format";
import { downloadUrl } from "@/lib/api";

const COLS = [
  { key: "trade_id", label: "#", num: true },
  { key: "entry_time", label: "Entry", num: false },
  { key: "exit_time", label: "Exit", num: false },
  { key: "direction", label: "Dir", num: false },
  { key: "size", label: "Sz", num: true },
  { key: "entry_price", label: "Entry $", num: true },
  { key: "exit_price", label: "Exit $", num: true },
  { key: "hold_bars", label: "Hold", num: true },
  { key: "gross_pnl", label: "Gross", num: true },
  { key: "commission", label: "Comm", num: true },
  { key: "slippage_cost", label: "Slip", num: true },
  { key: "net_pnl", label: "Net P&L", num: true },
  { key: "exit_reason", label: "Exit Reason", num: false },
];

export const TradeTable = ({ trades = [], runId }) => {
  const [q, setQ] = useState("");
  const [dir, setDir] = useState("all");
  const [sortKey, setSortKey] = useState("trade_id");
  const [sortAsc, setSortAsc] = useState(true);

  const rows = useMemo(() => {
    let r = trades;
    if (dir !== "all") r = r.filter((t) => t.direction === dir);
    if (q.trim()) {
      const s = q.toLowerCase();
      r = r.filter(
        (t) =>
          String(t.trade_id).includes(s) ||
          t.entry_time.toLowerCase().includes(s) ||
          t.exit_time.toLowerCase().includes(s) ||
          t.exit_reason.toLowerCase().includes(s)
      );
    }
    const sorted = [...r].sort((a, b) => {
      const av = a[sortKey];
      const bv = b[sortKey];
      if (av < bv) return sortAsc ? -1 : 1;
      if (av > bv) return sortAsc ? 1 : -1;
      return 0;
    });
    return sorted;
  }, [trades, q, dir, sortKey, sortAsc]);

  const toggleSort = (k) => {
    if (k === sortKey) setSortAsc((s) => !s);
    else {
      setSortKey(k);
      setSortAsc(true);
    }
  };

  const cellVal = (t, c) => {
    const v = t[c.key];
    if (["gross_pnl", "commission", "slippage_cost", "net_pnl"].includes(c.key)) {
      return <span className={signClass(c.key === "commission" || c.key === "slippage_cost" ? -1 : v)}>{fmtUsd(v, 2)}</span>;
    }
    if (["entry_price", "exit_price"].includes(c.key)) return fmtNum(v, 2);
    if (c.key === "direction")
      return (
        <span
          className={`rounded px-1.5 py-0.5 text-[10px] font-semibold ${
            v === "LONG"
              ? "bg-[hsl(var(--pos)/0.15)] text-[hsl(var(--pos))]"
              : "bg-[hsl(var(--neg)/0.15)] text-[hsl(var(--neg))]"
          }`}
        >
          {v}
        </span>
      );
    return v;
  };

  return (
    <Card className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))]">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-[hsl(var(--hairline))] p-3">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold uppercase tracking-wide">Trade Log</h3>
          <span className="font-mono text-xs text-muted-foreground">
            {rows.length} / {trades.length}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="relative">
            <Search className="absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              data-testid="trade-log-search-input"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder={"search\u2026"}
              className="h-8 w-40 border-[hsl(var(--hairline))] bg-[hsl(var(--surface-0))] pl-7 font-mono text-xs"
            />
          </div>
          <Select value={dir} onValueChange={setDir}>
            <SelectTrigger
              data-testid="trade-log-direction-filter"
              className="h-8 w-28 border-[hsl(var(--hairline))] bg-[hsl(var(--surface-0))] font-mono text-xs"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent className="font-mono text-xs">
              <SelectItem value="all">ALL</SelectItem>
              <SelectItem value="LONG">LONG</SelectItem>
              <SelectItem value="SHORT">SHORT</SelectItem>
            </SelectContent>
          </Select>
          <Button
            data-testid="trade-log-download-csv-button"
            variant="outline"
            size="sm"
            asChild
            disabled={!runId}
            className="gap-1 border-[hsl(var(--hairline))]"
          >
            <a href={runId ? downloadUrl(runId, "trades") : "#"} download>
              <Download className="h-3.5 w-3.5" /> CSV
            </a>
          </Button>
        </div>
      </div>

      {trades.length === 0 ? (
        <div className="flex h-40 items-center justify-center text-xs text-muted-foreground">
          No trades yet. Run a backtest to populate the trade log.
        </div>
      ) : (
        <ScrollArea className="h-[520px]">
          <table className="w-full border-collapse text-xs">
            <thead className="sticky top-0 z-10 bg-[hsl(var(--card))]">
              <tr className="border-b border-[hsl(var(--hairline))]">
                {COLS.map((c) => (
                  <th
                    key={c.key}
                    onClick={() => toggleSort(c.key)}
                    className={`cursor-pointer select-none whitespace-nowrap px-2 py-2 font-medium text-muted-foreground hover:text-foreground ${
                      c.num ? "text-right" : "text-left"
                    }`}
                  >
                    <span className="inline-flex items-center gap-1">
                      {c.label}
                      {sortKey === c.key && <ArrowUpDown className="h-3 w-3" />}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((t) => (
                <tr
                  key={t.trade_id}
                  className="border-b border-[hsl(var(--hairline)/0.5)] odd:bg-[hsl(var(--muted)/0.3)] hover:bg-[hsl(var(--muted)/0.55)]"
                >
                  {COLS.map((c) => (
                    <td
                      key={c.key}
                      className={`whitespace-nowrap px-2 py-1.5 ${
                        c.num ? "text-right font-mono tabular-nums" : "text-left"
                      }`}
                    >
                      {cellVal(t, c)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </ScrollArea>
      )}
    </Card>
  );
};
