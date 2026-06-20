import React from "react";
import { Play, Sparkles, RotateCcw, Loader2 } from "lucide-react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";

const ParamRow = ({ label, sym, testid, value, min, max, step, onChange, decimals = 2 }) => (
  <div className="space-y-2">
    <div className="flex items-center justify-between">
      <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">
        {label} <span className="text-[hsl(var(--info))]">{sym}</span>
      </Label>
      <Input
        data-testid={`${testid}-input`}
        type="number"
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(e) => {
          const v = parseFloat(e.target.value);
          if (!Number.isNaN(v)) onChange(Math.min(max, Math.max(min, v)));
        }}
        className="h-7 w-20 border-[hsl(var(--hairline))] bg-[hsl(var(--surface-0))] px-2 text-right font-mono text-xs tabular-nums"
      />
    </div>
    <Slider
      data-testid={`${testid}-slider`}
      value={[value]}
      min={min}
      max={max}
      step={step}
      onValueChange={(v) => onChange(v[0])}
    />
    <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
      <span>{min.toFixed(decimals)}</span>
      <span>{max.toFixed(decimals)}</span>
    </div>
  </div>
);

export const RunPanel = ({
  instruments,
  symbol,
  setSymbol,
  params,
  setParams,
  onRunSingle,
  onRunOptimize,
  onReset,
  singleLoading,
  optimizeRunning,
}) => {
  const busy = singleLoading || optimizeRunning;
  const set = (k, v) => setParams((p) => ({ ...p, [k]: v }));

  return (
    <Card className="border-[hsl(var(--hairline))] bg-[hsl(var(--card))] p-4">
      <div className="mb-3 flex items-center justify-between">
        <h2 className="text-sm font-semibold uppercase tracking-wide">Run Panel</h2>
        <Button
          data-testid="run-panel-reset-button"
          variant="ghost"
          size="icon"
          className="h-7 w-7 text-muted-foreground"
          onClick={onReset}
          disabled={busy}
          title="Reset to defaults"
        >
          <RotateCcw className="h-4 w-4" />
        </Button>
      </div>

      <div className="space-y-2">
        <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">
          Instrument
        </Label>
        <Select value={symbol} onValueChange={setSymbol} disabled={busy}>
          <SelectTrigger
            data-testid="run-panel-instrument-select"
            className="h-8 border-[hsl(var(--hairline))] bg-[hsl(var(--surface-0))] font-mono text-xs"
          >
            <SelectValue />
          </SelectTrigger>
          <SelectContent className="font-mono text-xs">
            {instruments.map((i) => (
              <SelectItem key={i.symbol} value={i.symbol} disabled={!i.available}>
                {i.symbol} &mdash; {i.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <Separator className="my-4 bg-[hsl(var(--hairline))]" />

      <div className="space-y-5">
        <ParamRow
          label="Toxic-Cont Thresh"
          sym={"\u03c41"}
          testid="run-panel-tau1"
          value={params.toxic_continuation_threshold}
          min={0.4}
          max={0.8}
          step={0.05}
          onChange={(v) => set("toxic_continuation_threshold", Number(v.toFixed(2)))}
        />
        <ParamRow
          label="Toxic-Rev Thresh"
          sym={"\u03c42"}
          testid="run-panel-tau2"
          value={params.toxic_reversal_threshold}
          min={0.4}
          max={0.8}
          step={0.05}
          onChange={(v) => set("toxic_reversal_threshold", Number(v.toFixed(2)))}
        />
        <ParamRow
          label="Max Hold Bars"
          sym="N"
          testid="run-panel-n"
          value={params.max_hold_bars}
          min={5}
          max={30}
          step={5}
          decimals={0}
          onChange={(v) => set("max_hold_bars", Math.round(v))}
        />

        <div className="flex items-center justify-between rounded-md border border-[hsl(var(--hairline))] bg-[hsl(var(--surface-0))] px-3 py-2">
          <Label className="text-[11px] uppercase tracking-wide text-muted-foreground">
            Regime Early-Exit
          </Label>
          <Switch
            data-testid="run-panel-regime-exit-switch"
            checked={params.regime_exit_enabled}
            onCheckedChange={(v) => set("regime_exit_enabled", v)}
            disabled={busy}
          />
        </div>
      </div>

      <Separator className="my-4 bg-[hsl(var(--hairline))]" />

      <div className="space-y-2">
        <Button
          data-testid="run-panel-run-single-button"
          className="w-full gap-2 bg-[hsl(var(--info))] font-medium text-[hsl(var(--accent-foreground))] hover:bg-[hsl(var(--info)/0.85)]"
          onClick={onRunSingle}
          disabled={busy}
        >
          {singleLoading ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          Run Single Backtest
        </Button>
        <Button
          data-testid="run-panel-run-wfo-button"
          variant="outline"
          className="w-full gap-2 border-[hsl(var(--info)/0.4)] text-[hsl(var(--info))] hover:bg-[hsl(var(--info)/0.1)]"
          onClick={onRunOptimize}
          disabled={busy}
        >
          {optimizeRunning ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Sparkles className="h-4 w-4" />
          )}
          Walk-Forward Optimize
        </Button>
        <p className="pt-1 text-[10px] leading-4 text-muted-foreground">
          Single backtest evaluates the test set with the chosen thresholds and
          hold window. Optimization grid-searches 486 combos via walk-forward on
          the training set (several minutes) then freezes the best params for an
          out-of-sample test.
        </p>
      </div>
    </Card>
  );
};
