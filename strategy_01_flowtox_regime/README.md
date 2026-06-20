# FLOWTOX_REGIME_01 — Adaptive Flow-Toxicity with Regime-Aware Sizing

A spec-faithful quantitative trading **strategy engine** + a dark "terminal" web
dashboard for ES E-mini S&P 500 futures (5-minute RTH bars).

## What it does (per `coding_spec_strategy_01.md`)

1. **Flow toxicity** — BVC-signed volume vs. tick-rule direction disagreement.
2. **Volatility** — per-bar Parkinson variance → daily/weekly/monthly realized
   variance → **HAR-RV** OLS forecast (fit on train only).
3. **Vol regime** — low/medium/high via **train-only** 33rd/67th percentiles.
4. **Regime model** — 3-state Gaussian Mixture (HMM proxy) on 5 rolling
   observable features → posteriors {Normal, Toxic-Continuation, Toxic-Reversal},
   deterministically relabelled.
5. **Signals** — long (toxic-continuation) / short (toxic-reversal) gated by HMM
   posterior thresholds **τ1 / τ2**, BVC direction, and non-high vol regime; no
   pyramiding; exits at **N** bars or regime early-exit.
6. **Sizing** — inverse-volatility targeting, clamped to 1–10 contracts.
7. **Backtest** — close-to-close, 1 tick/side slippage + $2.50 RT commission.
8. **Validation** — walk-forward optimization of **exactly 3** params (τ1, τ2, N)
   on train only, then a single frozen out-of-sample test evaluation with full
   metrics, significance tests, and acceptance checks.

> Honest-results philosophy: the strategy is a research hypothesis and may
> **fail** the acceptance criteria out-of-sample. That is reported neutrally
> (PASS/FAIL chips) — it is not an application error.

## Engine package (`/app/strategy_01_flowtox_regime/`)

| Module | Purpose |
|---|---|
| `config.py` | All constants, optimizable param space, **instrument registry** |
| `data_pipeline.py` | Load / validate / RTH-verify / train-test split |
| `features.py` | All feature engineering (Section 2) |
| `har_model.py` | HAR-RV fit + forecast |
| `hmm_model.py` | 3-state GMM fit, posteriors, semantic relabelling |
| `signal_generator.py` | Signal generation (numpy fast loop) |
| `backtest.py` | Cost-aware close-to-close backtester |
| `metrics.py` | Metrics, significance tests, acceptance checks |
| `validation.py` | Walk-forward optimization (per-window feature caching) |
| `pipeline.py` | High-level orchestration used by CLI + API |
| `main.py` | CLI: `python -m strategy_01_flowtox_regime.main` |

Run the standalone POC validation: `python /app/test_core.py`.

## Web app

- **Backend** (`/app/backend/server.py`, `strategy_service.py`): FastAPI wrapping
  the engine. Models are fitted once and cached (first single backtest ~60-90s,
  then ~0.4s). Walk-forward optimization runs as a **background job** (~3-4 min)
  with pollable progress. Artifacts are written under `backend/output/{run_id}/`.
- **Frontend** (`/app/frontend/src`): React dark Bloomberg-terminal dashboard —
  Run Panel (τ1/τ2/N), progress console, acceptance scorecard, metrics grid,
  6 charts, sensitivity heatmap, dense trade log, model info, downloads.

### Key endpoints
- `GET  /api/instruments`
- `GET  /api/strategy/info`
- `POST /api/backtest/single` `{symbol, toxic_continuation_threshold, toxic_reversal_threshold, max_hold_bars, regime_exit_enabled}`
- `POST /api/optimize/start` `{symbol}` → `{job_id}`; poll `GET /api/optimize/status/{job_id}`
- `GET  /api/runs/{run_id}` and `GET /api/runs/{run_id}/download/{metrics|trades|equity}`

## Adding another futures instrument (same CSV schema as ES)

The system is multi-instrument ready. To add e.g. **NQ**:

1. Drop the CSV (columns: `ts_event, symbol, open, high, low, close, volume`)
   into `/app/backend/data/NQ_5min_RTH_6year.csv` (or `/app/`).
2. Add a registry entry in `strategy_01_flowtox_regime/config.py`:

   ```python
   INSTRUMENTS["NQ"] = {
       "name": "E-mini Nasdaq-100",
       "csv_path": "/app/backend/data/NQ_5min_RTH_6year.csv",
       "contract_specs": {"tick_size": 0.25, "tick_value": 5.00,
                            "point_value": 20.00, "multiplier": 20.00,
                            "symbol": "NQ", "exchange": "CME", "currency": "USD"},
       "slippage_ticks": 1,
       "commission_rt": 2.50,
       "bars_per_day": 78,
       "train_end_index": <inclusive train end index for that file>,
   }
   ```

3. Restart the backend. The new instrument appears in the dashboard selector and
   uses its own contract specs for sizing and P&L — no core code changes needed.

## Notes / spec-permitted deviations
- `HMM_N_INIT` reduced (10→5 default; smaller for walk-forward windows) for
  tractable runtime — a minor, spec-permitted deviation (Section 8.3).
- Walk-forward features are engineered **once per window** (they are
  param-independent) instead of per param combo — a pure caching optimization
  that produces identical results far faster.
- Max drawdown is anchored to the reference account value so it reads as a
  fraction of equity.
