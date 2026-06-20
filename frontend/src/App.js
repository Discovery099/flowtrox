import React, { useEffect, useRef, useState, useCallback } from "react";
import "@/App.css";
import { toast } from "sonner";
import { Toaster } from "@/components/ui/sonner";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { LayoutDashboard, LineChart as LineIcon, Table2, Cpu } from "lucide-react";

import {
  getInstruments,
  getStrategyInfo,
  runSingle,
  startOptimize,
  getJobStatus,
} from "@/lib/api";
import { TopBar } from "@/components/TopBar";
import { RunPanel } from "@/components/RunPanel";
import { ProgressConsole } from "@/components/ProgressConsole";
import { AcceptanceScorecard } from "@/components/AcceptanceScorecard";
import { MetricsGrid } from "@/components/MetricsGrid";
import { ModelInfo } from "@/components/ModelInfo";
import { TradeTable } from "@/components/TradeTable";
import { SensitivityHeatmap } from "@/components/SensitivityHeatmap";
import {
  EquityChart,
  DrawdownChart,
  MonthlyReturnsChart,
  PnlHistogram,
  RegimeChart,
  VolRegimeChart,
} from "@/components/charts/Charts";

const DEFAULT_PARAMS = {
  toxic_continuation_threshold: 0.55,
  toxic_reversal_threshold: 0.55,
  max_hold_bars: 15,
  regime_exit_enabled: true,
};

export default function App() {
  const [instruments, setInstruments] = useState([]);
  const [symbol, setSymbol] = useState("ES");
  const [params, setParams] = useState(DEFAULT_PARAMS);

  const [result, setResult] = useState(null);
  const [singleLoading, setSingleLoading] = useState(false);

  const [jobId, setJobId] = useState(null);
  const [jobStatus, setJobStatus] = useState("idle");
  const [jobEvents, setJobEvents] = useState([]);
  const [jobPct, setJobPct] = useState(0);
  const [bestScore, setBestScore] = useState(null);

  const [tab, setTab] = useState("overview");
  const pollRef = useRef(null);

  useEffect(() => {
    getInstruments()
      .then((d) => setInstruments(d.instruments || []))
      .catch(() => toast.error("Failed to load instruments"));
    getStrategyInfo().catch(() => {});
  }, []);

  const optimizeRunning = jobStatus === "running" || jobStatus === "queued";

  const onRunSingle = useCallback(async () => {
    setSingleLoading(true);
    setJobStatus("idle");
    setJobEvents([
      { ts: new Date().toISOString(), stage: "single", message: `Running single backtest on ${symbol}\u2026 (first run fits models, ~60s)`, pct: 0 },
    ]);
    const tid = toast.loading("Running single backtest (first run warms models ~60s)\u2026");
    try {
      const data = await runSingle({ symbol, ...params });
      setResult(data);
      setBestScore(null);
      setJobEvents((e) => [
        ...e,
        { ts: new Date().toISOString(), stage: "done", message: `Done: ${data.metrics.total_trades} trades, Sharpe ${Number(data.metrics.sharpe_ratio).toFixed(3)}`, pct: 100 },
      ]);
      toast.success("Backtest complete", { id: tid });
      setTab("overview");
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message;
      setJobEvents((e) => [...e, { ts: new Date().toISOString(), stage: "error", message: `FAILED: ${msg}` }]);
      toast.error(`Backtest failed: ${msg}`, { id: tid });
    } finally {
      setSingleLoading(false);
    }
  }, [symbol, params]);

  const stopPolling = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  };

  const onRunOptimize = useCallback(async () => {
    try {
      stopPolling();
      setResult((r) => r); // keep last snapshot
      setJobStatus("queued");
      setJobPct(0);
      setBestScore(null);
      setJobEvents([{ ts: new Date().toISOString(), stage: "queued", message: `Starting walk-forward optimization on ${symbol}\u2026`, pct: 0 }]);
      const { job_id } = await startOptimize(symbol);
      setJobId(job_id);
      toast.info("Walk-forward optimization started (several minutes)");

      pollRef.current = setInterval(async () => {
        try {
          const job = await getJobStatus(job_id);
          setJobStatus(job.status);
          setJobPct(job.pct || 0);
          if (job.events) setJobEvents(job.events);
          const last = (job.events || []).filter((e) => e.best_score !== undefined && e.best_score !== null).slice(-1)[0];
          if (last) setBestScore(last.best_score);
          if (job.status === "done") {
            stopPolling();
            if (job.result) {
              setResult(job.result);
              setBestScore(job.result.walk_forward_sharpe);
              setParams((p) => ({ ...p, ...job.result.best_params }));
              toast.success("Optimization complete \u2014 best params applied");
              setTab("overview");
            }
          } else if (job.status === "failed") {
            stopPolling();
            toast.error(`Optimization failed: ${job.error || "unknown error"}`);
          }
        } catch (e) {
          // transient poll error; keep trying
        }
      }, 1800);
    } catch (err) {
      const msg = err?.response?.data?.detail || err.message;
      setJobStatus("failed");
      toast.error(`Could not start optimization: ${msg}`);
    }
  }, [symbol]);

  useEffect(() => stopPolling, []);

  const onReset = () => {
    setParams(DEFAULT_PARAMS);
    toast.info("Parameters reset to defaults");
  };

  const metrics = result?.metrics || null;
  const checks = result?.checks || null;
  const charts = result?.charts || {};
  const modelInfo = result?.model_info || null;
  const bestParams = result?.best_params || (result ? result.params : null);
  const walkForward = result?.walk_forward_sharpe !== undefined
    ? { sharpe: result.walk_forward_sharpe, windows: result.n_windows }
    : null;
  const runId = result?.run_id || null;
  const lastRunAt = result?.created_at ? result.created_at.slice(11, 19) + " UTC" : null;

  return (
    <div className="app-shell">
      <Toaster position="top-right" theme="dark" richColors />
      <TopBar instrument={symbol} jobStatus={result && jobStatus === "idle" ? "done" : jobStatus} lastRunAt={lastRunAt} runId={runId} />

      <div className="relative z-10 mx-auto grid w-full max-w-[1700px] grid-cols-1 gap-3 p-3 lg:grid-cols-[330px_minmax(0,1fr)]">
        {/* Left rail */}
        <div className="space-y-3">
          <RunPanel
            instruments={instruments}
            symbol={symbol}
            setSymbol={setSymbol}
            params={params}
            setParams={setParams}
            onRunSingle={onRunSingle}
            onRunOptimize={onRunOptimize}
            onReset={onReset}
            singleLoading={singleLoading}
            optimizeRunning={optimizeRunning}
          />
          <ProgressConsole events={jobEvents} pct={jobPct} status={optimizeRunning ? "running" : jobStatus} bestScore={bestScore} />
        </div>

        {/* Main */}
        <div className="min-w-0">
          <Tabs value={tab} onValueChange={setTab} className="w-full">
            <TabsList className="mb-3 grid h-10 w-full grid-cols-4 border border-[hsl(var(--hairline))] bg-[hsl(var(--surface-1))] sm:w-auto sm:inline-grid">
              <TabsTrigger data-testid="tab-overview" value="overview" className="gap-1.5 font-mono text-xs">
                <LayoutDashboard className="h-3.5 w-3.5" /> Overview
              </TabsTrigger>
              <TabsTrigger data-testid="tab-charts" value="charts" className="gap-1.5 font-mono text-xs">
                <LineIcon className="h-3.5 w-3.5" /> Charts
              </TabsTrigger>
              <TabsTrigger data-testid="tab-trades" value="trades" className="gap-1.5 font-mono text-xs">
                <Table2 className="h-3.5 w-3.5" /> Trades
              </TabsTrigger>
              <TabsTrigger data-testid="tab-model" value="model" className="gap-1.5 font-mono text-xs">
                <Cpu className="h-3.5 w-3.5" /> Model
              </TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-3">
              <AcceptanceScorecard checks={checks} metrics={metrics} />
              <MetricsGrid metrics={metrics} />
              <EquityChart data={charts.equity_series} />
            </TabsContent>

            <TabsContent value="charts" className="space-y-3">
              <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                <EquityChart data={charts.equity_series} />
                <DrawdownChart data={charts.drawdown_series} />
                <MonthlyReturnsChart data={charts.monthly_returns} />
                <PnlHistogram data={charts.pnl_histogram} />
                <RegimeChart data={charts.regime_series} />
                <VolRegimeChart data={charts.vol_series} />
              </div>
              <SensitivityHeatmap sensitivity={result?.sensitivity} bestParams={bestParams} />
            </TabsContent>

            <TabsContent value="trades">
              <TradeTable trades={result?.trades || []} runId={runId} />
            </TabsContent>

            <TabsContent value="model" className="space-y-3">
              <ModelInfo modelInfo={modelInfo} bestParams={bestParams} walkForward={walkForward} />
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </div>
  );
}
