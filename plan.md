# plan.md — FLOWTOX_REGIME_01 (Adaptive Flow-Toxicity with Regime-Aware Sizing)

## 1) Objectives
- Deliver a **spec-faithful quantitative trading engine** for FLOWTOX_REGIME_01 operating on ES 5‑minute RTH bars:
  - BVC signed volume + tick rule → toxicity disagreement.
  - Parkinson variance → RV(1D/1W/1M) → HAR‑RV OLS forecast.
  - Vol regime classification using **train-only** 33/67 percentiles.
  - 3‑state GMM (HMM proxy) on 5 rolling observable features → state posteriors.
  - Signal generation using **τ1/τ2** thresholds, BVC direction, vol gate, **no pyramiding**, exits via **N** and optional regime early-exit.
  - Inverse‑vol position sizing (1–10 contracts).
  - Close‑to‑close backtest with **1 tick/side slippage** + **$2.50 RT commission**.
  - Walk‑forward optimization of **exactly 3 params** (τ1, τ2, N) on train only.
  - Frozen out‑of‑sample test evaluation with full metrics + significance tests + acceptance checks.
- Produce outputs per spec and make them accessible via API/UI:
  - metrics JSON, trades CSV, equity CSV (downloadable).
  - dashboard charts + tables (equity/drawdown/monthly returns/regimes/vol regime/P&L histogram/sensitivity heatmap).
- Provide a **FastAPI + React** dashboard with a **dark Bloomberg-terminal quant aesthetic**:
  - run single backtests quickly after one-time model warm-up.
  - run walk-forward optimization as a **background job** with live progress polling.
  - present **honest pass/fail acceptance criteria** (failure is not treated as an error).
- Architect for **multi-instrument** extension later:
  - instrument registry (csv path + contract specs) so additional futures CSVs can be added with minimal changes.

**Current status**: Phases 1, 2 complete. P0 cold-start (502 timeout) fix COMPLETE & verified — ES models pre-warm in a non-blocking background thread on FastAPI startup; comprehensive testing passed (backend 9/9, frontend 15/15, integration 100%, no timeouts). Remaining work is optional Phase 3 hardening + future tasks (CSV upload UI, portfolio layer, live bridge).

---

## 2) Implementation Steps

### Phase 1 — Core POC (isolation; do not proceed until green) ✅ COMPLETE
**User stories**
1. As a quant, I can run a single Python script on the uploaded ES CSV and get deterministic results.
2. As a quant, I can verify train/test split and RTH constraints match the spec.
3. As a quant, I can generate engineered features and confirm no NaNs and correct value domains.
4. As a quant, I can run a backtest end-to-end and see a non-empty trade log (or an explicit “no trades” outcome).
5. As a quant, I can run walk-forward optimization and get best_params + test metrics without test leakage.

**Completed work (highlights)**
- Built `strategy_01_flowtox_regime/` package: config, data pipeline, features, HAR, GMM/HMM proxy, signals, backtest, metrics, walk-forward validation, pipeline.
- Implemented performance-critical loops using numpy arrays.
- Implemented walk-forward optimization with **per-window feature caching** (features are param-independent).
- Ran POC on real ES CSV successfully:
  - Data load/split validated on actual dataset (train=85,600; test=36,688).
  - Feature engineering validated; HAR fit strong (R²≈0.78).
  - GMM fit valid (weights 0.05) and posteriors sum to 1.
  - No leakage: test features use frozen train models.
  - Single backtest and optimizer both run end-to-end.

---

### Phase 2 — V1 App Development (wrap proven core) ✅ COMPLETE
**User stories**
1. As a user, I can run the full pipeline from the UI and see “running / completed / failed” state.
2. As a user, I can run a **single backtest** on the test set with custom τ1/τ2/N and instantly see updated metrics.
3. As a user, I can view equity curve, drawdown, monthly returns, and trade P&L distribution.
4. As a user, I can inspect the trade log table and download CSV outputs.
5. As a user, I can visualize regime posteriors and volatility regime over time.
6. As a user, I can run walk-forward optimization (long job) with progress logging and retrieve results.

**Backend (FastAPI) — Completed**
- Implemented routes:
  - `GET /api/instruments`
  - `GET /api/strategy/info`
  - `GET /api/model/status`, `POST /api/model/warm`
  - `POST /api/backtest/single`
  - `POST /api/optimize/start`, `GET /api/optimize/status/{job_id}`
  - `GET /api/runs/{run_id}`
  - `GET /api/runs/{run_id}/download/{kind}` (metrics/trades/equity)
- Added `strategy_service.py`:
  - model caching (first warm-up ~60–90s; subsequent single backtests ~0.4s).
  - background optimization jobs with progress events + polling.
  - persistence of downloadable artifacts under `backend/output/{run_id}/`.
- Verified endpoints via curl:
  - single backtest works, returns charts/trades/metrics.
  - optimization job runs end-to-end, streams progress, produces sensitivity heatmap data.

**Frontend (React) — Completed**
- Implemented dark-first terminal dashboard with tabs:
  - Overview: AcceptanceScorecard + MetricsGrid + Equity chart.
  - Charts: 6 charts (equity, drawdown, monthly returns, P&L histogram, regime posteriors, vol regime) + sensitivity heatmap.
  - Trades: dense sortable/filterable trade log with CSV download.
  - Model: HAR params, GMM weights, vol thresholds, run configuration.
- Implemented RunPanel (instrument select + τ1/τ2/N controls + switches) and ProgressConsole.
- Verified via automation screenshots:
  - UI renders correctly and can run single backtest.
  - optimization flow works and populates sensitivity map.
  - honest acceptance FAIL states display correctly.

**Critical blocker fixed (template issue)**
- Resolved a pre-existing webpack-dev-server v4→v5 incompatibility in `craco.config.js`:
  - migrated `onBeforeSetupMiddleware`/`onAfterSetupMiddleware` → `setupMiddlewares`.
  - migrated `https` → `server`.
  - gated incompatible visual-edits wrapper behind `ENABLE_VISUAL_EDITS=true`.

---

### Phase 3 — Hardening + Multi-instrument readiness (NEXT)
**User stories**
1. As a user, I can trust the system is correct via automated tests (API + UI smoke tests).
2. As a user, I can add a new futures CSV + contract specs entry and run the same pipeline.
3. As a user, I can see clear errors when a CSV is malformed (missing columns, timezone issues).
4. As a user, I can export a complete run bundle (all outputs + config) and reproduce results.

**Work items (updated)**
- ✅ Multi-instrument architecture scaffold exists (`INSTRUMENTS` registry); finalize docs for adding new futures.
- ✅ P0 cold-start fix: pre-warm default ES (gmm+hmm) in background thread on startup (server.py `on_event("startup")` → `svc._EXECUTOR.submit`). Verified non-blocking; no 502s.
- ✅ Comprehensive automated tests via `testing_agent_v3` (iteration_5.json): backend 9/9, frontend 15/15, integration 100% — all passed, zero bugs.
- 🔜 Add/confirm pytest suite for core invariants:
  - posteriors sum to 1; signal domain in {-1,0,1}; no pyramiding; no NaNs; trade log schema.
- 🔜 Robustness improvements (optional):
  - clearer error surfaces for missing CSVs.
  - graceful empty states when no trades occur.
  - better reporting of long-job duration and last update time.

**Phase 3 exit**
- All automated tests pass.
- Documented steps for adding another futures CSV (same schema) and contract specs.

---

## Phase 4 — Faithful Pine Script export (COMPLETED)
Goal: turn the strategy into a TradingView-backtestable Pine v6 script as the
first step toward live data + paper trading.

- ✅ `backend/pine_exporter.py`: fits/loads the frozen model for a (symbol,
  regime_model) and BAKES exact parameters into a self-contained Pine v6
  strategy:
  - HAR-RV OLS coefficients, train-set vol-regime p33/p67, StandardScaler
    mean/scale.
  - Gaussian emission log-likelihoods via baked inverse-covariances + log-norm
    constants (5x5 quadratic form fully unrolled).
  - GMM path -> log(weight)+emission softmax; HMM path -> forward FILTERING
    recursion (baked startprob + transition matrix), no look-ahead.
- ✅ Numerically verified vs. the Python engine to MACHINE PRECISION
  (GMM max-abs-diff 1.6e-15, HMM 1.7e-13) on real ES test bars.
- ✅ Endpoint `GET /api/pine-script/generate?symbol=&regime_model=` (threadpool;
  cold symbol fits ~25-90s, warm instant). Static `/api/pine-script` kept as a
  generic fallback.
- ✅ Frontend: Download button now generates the faithful script for the
  selected instrument + regime model, with a loading toast + blob download
  ("Download Pine Script (ES HMM)" etc.).
- HONEST CAVEAT documented in-script: TradingView's data feed differs from the
  RTH CSV, so trade-for-trade P&L won't match Python; logic + model are faithful.

### Pine fixes (post-user-feedback)
- ROOT CAUSE of "no trades / This report requires trade data": Pine v6 defaults
  margin_long/short = 100% (no leverage). ES notional (~7556 x $50) far exceeds a
  $100k account, so the broker emulator rejected every entry. FIX: set
  margin_long = 0 and margin_short = 0 (futures post margin, not notional).
- overlay = true so TradingView draws real entry/exit arrows ON the candles.
- Regime posteriors moved to the Data Window (display.data_window) so they don't
  flatten the price scale; added on-candle plotshape signal triangles + integer
  qty cast + entry guard (qty >= 1).

**Next (future, per user):** validate results on TradingView -> add live market
data feed -> paper trading mode (IBKR/Tradovate bridge or TradingView alerts).


## 3) Next Actions (immediate)
1. Run `testing_agent_v3` end-to-end (backend + frontend) and fix any failures.
2. Add/confirm unit tests (`pytest`) for core engine invariants and deterministic behavior.
3. Add a short “How to add another futures CSV” README (instrument registry + contract specs).
4. Final polish pass: ensure all interactive elements have `data-testid`, ensure downloads always work, ensure error handling is clean.

---

## 4) Success Criteria (updated)
- **Correctness (hard)**:
  - No lookahead, no test leakage, exactly 3 optimized parameters, cost model matches spec, deterministic runs.
- **POC completeness**:
  - End-to-end pipeline runs on ES CSV and produces outputs; optimizer completes within minutes.
- **App completeness**:
  - Single backtest fast after warm-up; WFO optimization runs as background job with progress polling; results downloadable.
- **Honest-results UX**:
  - PASS/FAIL acceptance states are clearly communicated; failure is not treated as an app error.
- **Extensibility**:
  - Adding a new futures CSV requires only adding to the instrument registry + contract specs (no core rewrites).
