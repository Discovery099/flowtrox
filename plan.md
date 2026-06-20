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
