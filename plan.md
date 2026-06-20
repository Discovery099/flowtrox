# plan.md — FLOWTOX_REGIME_01 (Adaptive Flow-Toxicity with Regime-Aware Sizing)

## 1) Objectives
- Implement the **spec-faithful** Python strategy engine for ES 5-min RTH data:
  - BVC signed volume + tick rule → toxicity disagreement.
  - Parkinson variance → RV(1D/1W/1M) → HAR-RV OLS forecast.
  - Vol regime classification using **train-only** 33/67 percentiles.
  - 3-state GMM (HMM proxy) on 5 rolling observable features → state posteriors.
  - Signal generation using **τ1/τ2** thresholds, BVC direction, vol gate, **no pyramiding**, exits via **N** and optional regime early-exit.
  - Inverse-vol position sizing (1–10 contracts).
  - Close-to-close backtest with **1 tick/side slippage** + **$2.50 RT commission**.
  - Walk-forward optimization of **exactly 3 params** (τ1, τ2, N) on train only.
  - Frozen out-of-sample test evaluation with full metrics + significance tests + acceptance checks.
- Produce outputs exactly per spec: metrics JSON, trades CSV, equity CSV, report TXT, plots.
- Wrap engine with FastAPI + React dark “terminal” dashboard (MVP) while keeping core authoritative.
- Architect for **multi-instrument** later: registry of instruments (csv path + contract specs).

---

## 2) Implementation Steps

### Phase 1 — Core POC (isolation; do not proceed until green)
**User stories**
1. As a quant, I can run a single Python script on the uploaded ES CSV and get deterministic results.
2. As a quant, I can verify train/test split and RTH constraints match the spec exactly.
3. As a quant, I can generate engineered features and confirm no NaNs and correct value domains.
4. As a quant, I can run a backtest end-to-end and see a non-empty trade log (or an explicit “no trades” outcome).
5. As a quant, I can run walk-forward optimization and get best_params + test metrics without test leakage.

**Steps**
1. **Web research (best practice quick scan)**: performance patterns for walk-forward grid search + caching (feature reuse) + numpy loop optimization for trading sims.
2. Create `strategy_01_flowtox_regime/` package per spec structure (config, pipeline, features, models, signals, backtest, metrics, validation, main).
3. Implement **data pipeline**:
   - Load from `/app/ES_5min_RTH_6year.csv` (and keep path configurable).
   - Drop NaN symbol rows, parse `ts_event` utc, validate OHLC & tick size sampling.
   - Verify RTH session timestamps.
   - Fixed split at index 85599 → train 85600, test 36695.
4. Implement **feature engineering** exactly per formulas and dependency order.
5. Implement **HAR model** fit on train only; clamp negative coeffs; forecast >= 0.
6. Implement **GMM/HMM proxy** fit on train only; compute posteriors on train/test.
7. Implement **signals** and **position sizing**:
   - Use numpy arrays in the bar loop (avoid `df.loc` in loop).
   - Enforce no pyramiding, exits, and volatility regime gate.
8. Implement **backtest**:
   - Numpy arrays for bar pnl/equity.
   - Exact slippage/commission model and trade log schema.
9. Implement **metrics + stats tests** and **acceptance checks**.
10. Implement **walk-forward optimization**:
   - Performance-critical: cache engineered features per window once (features are param-independent), then re-run only signal+backtest per param combo.
   - Enforce min trades + min win-rate rejection rules.
11. Build `poc_run.py` (or `python -m ...main`) that runs:
   - Load → validate → split → fit models (train) → walk-forward optimize → test evaluation → write outputs.
12. POC validation gate (must pass before Phase 2):
   - All data assertions pass.
   - Feature columns present; no NaNs; posteriors sum to 1.
   - Optimization completes in reasonable time (target: minutes, not hours) on this dataset.
   - Outputs generated in `output/`.

---

### Phase 2 — V1 App Development (wrap proven core)
**User stories**
1. As a user, I can run the full pipeline from the UI and see “running / completed / failed” state.
2. As a user, I can run a **single backtest** on the test set with custom τ1/τ2/N and instantly see updated metrics.
3. As a user, I can view equity curve, drawdown, monthly returns, and trade P&L distribution.
4. As a user, I can inspect the trade log table and download CSV outputs.
5. As a user, I can visualize regime posteriors and volatility regime over time.

**Backend (FastAPI)**
- Add `engine/` module that calls the strategy package.
- Endpoints (MVP):
  - `GET /api/instruments` → available instruments (ES now).
  - `POST /api/run/single` → run engineered-features + backtest for given τ1/τ2/N (fast path).
  - `POST /api/run/optimize` → run full walk-forward optimization + test evaluation (live; may take longer).
  - `GET /api/results/{run_id}` → fetch status + outputs.
  - `GET /api/results/{run_id}/download/{file}` → download metrics/trades/equity/report.
- Store run artifacts under `backend/output/{run_id}/`.

**Frontend (React, dark terminal aesthetic)**
- Pages/components:
  - Run panel (instrument selector, τ1/τ2/N inputs, buttons: Single Backtest / Full Optimize).
  - Progress + logs console.
  - Metrics scorecard + acceptance pass/fail.
  - Charts: equity, drawdown, monthly returns, regime posteriors, vol regime, P&L histogram.
  - Trades table with filtering + download.

**Phase 2 exit**
- 1 round of end-to-end test: run single + optimize flows, confirm charts render and downloads work.

---

### Phase 3 — Hardening + Multi-instrument readiness
**User stories**
1. As a user, I can add a new futures CSV + contract specs entry and run the same pipeline.
2. As a user, I can see clear errors when a CSV is malformed (missing columns, timezone issues).
3. As a user, I can compare results across instruments (basic side-by-side metrics).
4. As a user, I can re-run with fixed random seed and get identical results.
5. As a user, I can export a complete run bundle (all outputs + config).

**Work items**
- Instrument registry (`instruments.yml/json`) with: csv_path, tick_size, point_value, commission/slippage overrides if needed.
- Make feature engineering robust to partial sessions (<78 bars) while preserving spec assumptions.
- Add parameter sensitivity plot/heatmap generation after optimization.
- Improve performance: optional multiprocessing for grid search; smarter caching; profiling.
- Testing suite (`pytest`) for data, features, backtest invariants.

**Phase 3 exit**
- 1 round of end-to-end tests on ES and on one “simulated second instrument” fixture.

---

## 3) Next Actions (immediate)
1. Implement Phase 1 POC skeleton package + `poc_run.py`.
2. Validate data assertions on `/app/ES_5min_RTH_6year.csv`.
3. Implement features → HAR → GMM → signals → backtest; confirm outputs.
4. Implement walk-forward optimization with window-level feature caching; profile runtime.
5. Only after POC is green: start Phase 2 FastAPI endpoints + minimal dashboard.

---

## 4) Success Criteria
- **Correctness (hard)**: no lookahead, no test leakage, exactly 3 optimized parameters, cost model matches spec, deterministic runs.
- **POC completeness**: end-to-end pipeline runs on ES CSV and produces all required output files + plots.
- **Performance (practical)**: walk-forward optimization completes without timeouts (minutes-scale).
- **V1 UX**: UI can run single backtest and full optimization and visualize/download results reliably.
- **Extensibility**: adding a new futures CSV requires only adding to instrument registry + contract specs (no core rewrites).
