"""Service layer that wraps the FLOWTOX_REGIME_01 engine for the web API.

Responsibilities:
- Cache fitted train models + engineered test features per instrument (the
  expensive GMM/HAR fit runs once; subsequent single backtests are ~0.2s).
- Produce JSON/chart-friendly payloads (downsampled series, histograms,
  monthly returns, model info).
- Manage long-running walk-forward optimization jobs in background threads
  with pollable progress.
- Persist downloadable artifacts (metrics.json, trades.csv, equity.csv).
"""

import os
import sys
import json
import threading
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Make the engine package importable.
sys.path.insert(0, "/app")

from strategy_01_flowtox_regime.config import (  # noqa: E402
    INSTRUMENTS,
    PARAM_DEFAULTS,
    PARAM_SEARCH_SPACE,
    ACCEPTANCE_CRITERIA,
    get_instrument,
)
from strategy_01_flowtox_regime.pipeline import (  # noqa: E402
    load_instrument_data,
    _fit_train_models,
    _evaluate_on_test,
    run_full_pipeline,
)

OUTPUT_DIR = "/app/backend/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory caches and job registry
# ---------------------------------------------------------------------------
_MODEL_CACHE = {}          # symbol -> cached fitted models + test features
_MODEL_LOCKS = {}          # symbol -> threading.Lock
_RUNS = {}                 # run_id -> run payload (for download + retrieval)
_JOBS = {}                 # job_id -> optimization job state
_EXECUTOR = ThreadPoolExecutor(max_workers=2)
_GLOBAL_LOCK = threading.Lock()


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _get_lock(symbol: str) -> threading.Lock:
    with _GLOBAL_LOCK:
        if symbol not in _MODEL_LOCKS:
            _MODEL_LOCKS[symbol] = threading.Lock()
        return _MODEL_LOCKS[symbol]


# ---------------------------------------------------------------------------
# Instrument / strategy metadata
# ---------------------------------------------------------------------------
def list_instruments() -> list:
    out = []
    for sym, info in INSTRUMENTS.items():
        out.append({
            "symbol": sym,
            "name": info["name"],
            "available": os.path.exists(info["csv_path"]),
            "tick_size": info["contract_specs"]["tick_size"],
            "point_value": info["contract_specs"]["point_value"],
            "commission_rt": info["commission_rt"],
            "slippage_ticks": info["slippage_ticks"],
        })
    return out


def strategy_info() -> dict:
    return {
        "name": "Adaptive Flow-Toxicity with Regime-Aware Sizing",
        "codename": "FLOWTOX_REGIME_01",
        "summary": (
            "Trades ES futures using BVC/tick flow-toxicity disagreement, gated "
            "by a 3-state HMM regime model and Parkinson-HAR volatility regimes, "
            "with inverse-volatility position sizing."
        ),
        "param_defaults": PARAM_DEFAULTS,
        "param_space": {
            "toxic_continuation_threshold": {"min": 0.40, "max": 0.80, "step": 0.05},
            "toxic_reversal_threshold": {"min": 0.40, "max": 0.80, "step": 0.05},
            "max_hold_bars": {"min": 5, "max": 30, "step": 5},
        },
        "acceptance_criteria": ACCEPTANCE_CRITERIA,
    }


# ---------------------------------------------------------------------------
# Model caching
# ---------------------------------------------------------------------------
def ensure_models(symbol: str) -> dict:
    """Fit (once) and cache train models + engineered test features."""
    symbol = symbol.upper()
    get_instrument(symbol)  # validates symbol
    lock = _get_lock(symbol)
    with lock:
        if symbol in _MODEL_CACHE:
            return _MODEL_CACHE[symbol]

        train_df, test_df, inst = load_instrument_data(symbol)
        fitted = _fit_train_models(train_df, hmm_n_init=10)

        # Engineer test features once (frozen models).
        from strategy_01_flowtox_regime.features import engineer_features
        test_feat, _ = engineer_features(
            test_df, fit_models=False,
            har_params=fitted["har_params"], hmm_model=fitted["hmm_model"],
            scaler=fitted["scaler"], label_map=fitted["label_map"],
            vol_thresholds=fitted["vol_thresholds"],
        )
        num_test_days = int(test_df["ts_event"].dt.date.nunique())

        model_info = {
            "har_params": {k: float(v) for k, v in fitted["har_params"].items()},
            "gmm_weights": [float(w) for w in fitted["hmm_model"].weights_],
            "vol_thresholds": {k: float(v) for k, v in fitted["vol_thresholds"].items()},
            "label_map": [int(x) for x in fitted["label_map"]],
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "num_test_days": num_test_days,
            "train_start": train_df["ts_event"].min().isoformat(),
            "train_end": train_df["ts_event"].max().isoformat(),
            "test_start": test_df["ts_event"].min().isoformat(),
            "test_end": test_df["ts_event"].max().isoformat(),
            "toxicity_rate": float((test_feat["toxicity_disagreement"] < 0).mean()),
            "tick_size": float(inst["contract_specs"]["tick_size"]),
            "point_value": float(inst["contract_specs"]["point_value"]),
            "commission_rt": float(inst["commission_rt"]),
            "instrument_name": inst["name"],
        }

        _MODEL_CACHE[symbol] = {
            "train_df": train_df,
            "test_df": test_df,
            "fitted": fitted,
            "test_feat": test_feat,
            "num_test_days": num_test_days,
            "model_info": model_info,
            "inst": inst,
        }
        return _MODEL_CACHE[symbol]


def is_model_ready(symbol: str) -> bool:
    return symbol.upper() in _MODEL_CACHE


# ---------------------------------------------------------------------------
# Chart payload helpers
# ---------------------------------------------------------------------------
def _downsample_idx(n: int, target: int = 700) -> np.ndarray:
    if n <= target:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, target).astype(int))


def _build_chart_payload(test_df: pd.DataFrame, result: dict) -> dict:
    df = result["df_with_pnl"]
    n = len(df)
    idx = _downsample_idx(n, 700)

    ts = df["ts_event"].dt.strftime("%Y-%m-%d %H:%M").to_numpy()
    equity = df["equity"].to_numpy(np.float64)

    # Equity curve.
    equity_series = [
        {"t": str(ts[i]), "equity": round(float(equity[i]), 2)} for i in idx
    ]

    # Underwater drawdown (% of anchored account equity).
    anchored = equity + 100_000.0
    peak = np.maximum.accumulate(anchored)
    dd = (anchored - peak) / peak * 100.0
    drawdown_series = [
        {"t": str(ts[i]), "dd": round(float(dd[i]), 3)} for i in idx
    ]

    # Regime posteriors (downsampled).
    s0 = df["hmm_state_0_posterior"].to_numpy()
    s1 = df["hmm_state_1_posterior"].to_numpy()
    s2 = df["hmm_state_2_posterior"].to_numpy()
    regime_series = [
        {
            "t": str(ts[i]),
            "normal": round(float(s0[i]), 3),
            "continuation": round(float(s1[i]), 3),
            "reversal": round(float(s2[i]), 3),
        }
        for i in idx
    ]

    # Volatility regime timeline.
    vol = df["vol_regime"].to_numpy()
    vol_series = [{"t": str(ts[i]), "regime": int(vol[i])} for i in idx]

    # Monthly returns from daily returns.
    dr = result["daily_returns"].copy()
    dr.index = pd.to_datetime(dr.index)
    monthly = dr.groupby([dr.index.to_period("M")]).sum()
    monthly_returns = [
        {"month": str(p), "ret": round(float(v) * 100.0, 3)}
        for p, v in monthly.items()
    ]

    # Per-trade P&L histogram.
    tl = result["trade_log"]
    hist = []
    if tl is not None and len(tl) > 0:
        pnl = tl["net_pnl"].to_numpy(np.float64)
        counts, edges = np.histogram(pnl, bins=30)
        for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
            center = (lo + hi) / 2.0
            hist.append({
                "bin": round(float(center), 1),
                "count": int(c),
                "sign": "pos" if center >= 0 else "neg",
            })

    return {
        "equity_series": equity_series,
        "drawdown_series": drawdown_series,
        "regime_series": regime_series,
        "vol_series": vol_series,
        "monthly_returns": monthly_returns,
        "pnl_histogram": hist,
    }


def _trades_payload(trade_log: pd.DataFrame) -> list:
    if trade_log is None or len(trade_log) == 0:
        return []
    tl = trade_log.copy()
    tl["entry_time"] = pd.to_datetime(tl["entry_time"]).dt.strftime("%Y-%m-%d %H:%M")
    tl["exit_time"] = pd.to_datetime(tl["exit_time"]).dt.strftime("%Y-%m-%d %H:%M")
    records = []
    for _, r in tl.iterrows():
        records.append({
            "trade_id": int(r["trade_id"]),
            "entry_time": r["entry_time"],
            "exit_time": r["exit_time"],
            "direction": r["direction"],
            "size": int(r["size"]),
            "entry_price": round(float(r["entry_price"]), 2),
            "exit_price": round(float(r["exit_price"]), 2),
            "hold_bars": int(r["hold_bars"]),
            "gross_pnl": round(float(r["gross_pnl"]), 2),
            "commission": round(float(r["commission"]), 2),
            "slippage_cost": round(float(r["slippage_cost"]), 2),
            "net_pnl": round(float(r["net_pnl"]), 2),
            "entry_signal": r["entry_signal"],
            "exit_reason": r["exit_reason"],
        })
    return records


def _persist_run(run_id: str, params: dict, metrics: dict, result: dict) -> None:
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    ser = {k: v for k, v in metrics.items() if isinstance(v, (int, float, bool, str))}
    ser["params"] = params
    with open(os.path.join(run_dir, "strategy_01_metrics.json"), "w") as f:
        json.dump(ser, f, indent=2)
    result["trade_log"].to_csv(os.path.join(run_dir, "strategy_01_trades.csv"), index=False)
    result["equity_curve"].to_csv(os.path.join(run_dir, "strategy_01_equity.csv"))


def _jsonify_metrics(metrics: dict) -> dict:
    out = {}
    for k, v in metrics.items():
        if isinstance(v, (np.floating,)):
            v = float(v)
        elif isinstance(v, (np.integer,)):
            v = int(v)
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
            v = None
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# Single backtest
# ---------------------------------------------------------------------------
def run_single(symbol: str, params: dict) -> dict:
    """Run a single backtest on the test set with the given params."""
    cache = ensure_models(symbol)
    from strategy_01_flowtox_regime.signal_generator import generate_signals
    from strategy_01_flowtox_regime.backtest import backtest
    from strategy_01_flowtox_regime.metrics import compute_all_metrics, check_acceptance
    from strategy_01_flowtox_regime.pipeline import costs_from_inst

    costs = costs_from_inst(cache["inst"])
    test_feat = cache["test_feat"]
    sigs = generate_signals(test_feat, params, point_value=costs["point_value"])
    result = backtest(
        sigs,
        point_value=costs["point_value"],
        slippage_points=costs["slippage_points"],
        commission_rt=costs["commission_rt"],
    )
    metrics = compute_all_metrics(result, cache["num_test_days"])
    checks = check_acceptance(metrics)

    run_id = str(uuid.uuid4())
    _persist_run(run_id, params, metrics, result)

    charts = _build_chart_payload(cache["test_df"], result)
    trades = _trades_payload(result["trade_log"])

    payload = {
        "run_id": run_id,
        "symbol": symbol.upper(),
        "mode": "single",
        "params": params,
        "metrics": _jsonify_metrics(metrics),
        "checks": {k: bool(v) for k, v in checks.items()},
        "model_info": cache["model_info"],
        "charts": charts,
        "trades": trades,
        "created_at": _now_iso(),
    }
    _RUNS[run_id] = {"payload": payload}
    return payload


# ---------------------------------------------------------------------------
# Walk-forward optimization (background job)
# ---------------------------------------------------------------------------
def start_optimization(symbol: str) -> str:
    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "job_id": job_id,
        "symbol": symbol.upper(),
        "status": "queued",
        "pct": 0,
        "events": [{"ts": _now_iso(), "stage": "queued", "message": "Job queued", "pct": 0}],
        "result": None,
        "error": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }
    _EXECUTOR.submit(_run_optimization_job, job_id)
    return job_id


def _run_optimization_job(job_id: str) -> None:
    job = _JOBS[job_id]
    job["status"] = "running"

    def progress_cb(ev: dict):
        ev = dict(ev)
        ev["ts"] = _now_iso()
        job["events"].append(ev)
        if "pct" in ev:
            job["pct"] = int(ev["pct"])
        job["updated_at"] = _now_iso()
        # Cap event history.
        if len(job["events"]) > 400:
            job["events"] = job["events"][-400:]

    try:
        symbol = job["symbol"]
        results = run_full_pipeline(symbol, progress_cb=progress_cb)

        metrics = results["metrics"]
        checks = results["checks"]
        result = results["result"]
        best_params = results["best_params"]

        run_id = str(uuid.uuid4())
        _persist_run(run_id, best_params, metrics, result)

        # Build model info from fitted.
        fitted = results["fitted"]
        test_df = results["test_df"]
        train_df = results["train_df"]
        model_info = {
            "har_params": {k: float(v) for k, v in fitted["har_params"].items()},
            "gmm_weights": [float(w) for w in fitted["hmm_model"].weights_],
            "vol_thresholds": {k: float(v) for k, v in fitted["vol_thresholds"].items()},
            "label_map": [int(x) for x in fitted["label_map"]],
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "num_test_days": int(test_df["ts_event"].dt.date.nunique()),
            "train_start": train_df["ts_event"].min().isoformat(),
            "train_end": train_df["ts_event"].max().isoformat(),
            "test_start": test_df["ts_event"].min().isoformat(),
            "test_end": test_df["ts_event"].max().isoformat(),
        }

        charts = _build_chart_payload(test_df, result)
        trades = _trades_payload(result["trade_log"])

        # Parameter sensitivity grid (avg over N for each tau1 x tau2).
        sens = _build_sensitivity(results["walk_forward"]["combo_scores"])

        payload = {
            "run_id": run_id,
            "symbol": symbol,
            "mode": "optimize",
            "params": best_params,
            "best_params": best_params,
            "walk_forward_sharpe": results["walk_forward"]["best_score"],
            "n_windows": results["walk_forward"]["n_windows"],
            "n_combos": results["walk_forward"]["n_combos"],
            "metrics": _jsonify_metrics(metrics),
            "checks": {k: bool(v) for k, v in checks.items()},
            "model_info": model_info,
            "charts": charts,
            "trades": trades,
            "sensitivity": sens,
            "created_at": _now_iso(),
        }
        _RUNS[run_id] = {"payload": payload}

        job["result"] = payload
        job["status"] = "done"
        job["pct"] = 100
        job["events"].append({"ts": _now_iso(), "stage": "done", "message": "Optimization complete", "pct": 100})
        job["updated_at"] = _now_iso()
    except Exception as exc:  # noqa: BLE001
        job["status"] = "failed"
        job["error"] = str(exc)
        job["events"].append({
            "ts": _now_iso(), "stage": "error",
            "message": f"FAILED: {exc}", "pct": job.get("pct", 0),
        })
        job["updated_at"] = _now_iso()
        traceback.print_exc()


def _build_sensitivity(combo_scores: list) -> dict:
    """Aggregate combo scores into a tau1 x tau2 heatmap (best over N)."""
    t1s = sorted(set(round(c["toxic_continuation_threshold"], 2) for c in combo_scores))
    t2s = sorted(set(round(c["toxic_reversal_threshold"], 2) for c in combo_scores))
    best = {}
    for c in combo_scores:
        key = (round(c["toxic_continuation_threshold"], 2), round(c["toxic_reversal_threshold"], 2))
        val = c["avg_sharpe"]
        if val <= -900:
            val = None
        prev = best.get(key)
        if val is not None and (prev is None or val > prev):
            best[key] = val
    cells = []
    for t1 in t1s:
        for t2 in t2s:
            v = best.get((t1, t2))
            cells.append({"tau1": t1, "tau2": t2, "sharpe": None if v is None else round(v, 3)})
    return {"tau1_values": t1s, "tau2_values": t2s, "cells": cells}


def get_job(job_id: str) -> dict:
    job = _JOBS.get(job_id)
    if job is None:
        return None
    # Return last ~60 events to keep payload light; client merges.
    return {
        "job_id": job["job_id"],
        "symbol": job["symbol"],
        "status": job["status"],
        "pct": job["pct"],
        "events": job["events"][-80:],
        "error": job["error"],
        "result": job["result"],
        "updated_at": job["updated_at"],
    }


def get_run(run_id: str) -> dict:
    r = _RUNS.get(run_id)
    return r["payload"] if r else None


def run_file_path(run_id: str, kind: str) -> str:
    mapping = {
        "metrics": "strategy_01_metrics.json",
        "trades": "strategy_01_trades.csv",
        "equity": "strategy_01_equity.csv",
    }
    fname = mapping.get(kind)
    if fname is None:
        return None
    path = os.path.join(OUTPUT_DIR, run_id, fname)
    return path if os.path.exists(path) else None
