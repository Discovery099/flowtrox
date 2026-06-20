"""High-level orchestration used by both the CLI and the web backend.

Provides:
- load_instrument_data: load + split for a registered instrument.
- run_single_backtest: fit models on train, evaluate a single param set on test.
- run_full_pipeline: walk-forward optimize on train, evaluate best params on test.

Results are returned as plain-Python/JSON-friendly structures plus the raw
backtest result for charting.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import get_instrument, PARAM_DEFAULTS
from .data_pipeline import load_data, split_train_test, validate_data
from .features import engineer_features
from .signal_generator import generate_signals
from .backtest import backtest
from .metrics import compute_all_metrics, check_acceptance
from .validation import run_walk_forward_optimization


def costs_from_inst(inst: dict) -> dict:
    """Build the cost/contract kwargs for signal sizing + backtesting."""
    cs = inst["contract_specs"]
    return {
        "point_value": float(cs["point_value"]),
        "slippage_points": float(inst["slippage_ticks"]) * float(cs["tick_size"]),
        "commission_rt": float(inst["commission_rt"]),
    }


def load_instrument_data(symbol: str = "ES"):
    """Load, validate and split data for a registered instrument."""
    inst = get_instrument(symbol)
    df = load_data(inst["csv_path"])
    validate_data(df, tick_size=inst["contract_specs"]["tick_size"])
    train_end_index = inst.get("train_end_index")
    if train_end_index is None:
        # Proportional ~70% split (matches the ES 85600/122288 ratio).
        train_end_index = int(round(0.70 * len(df))) - 1
    train_df, test_df = split_train_test(df, train_end_index)
    return train_df, test_df, inst


def _fit_train_models(train_df: pd.DataFrame, hmm_n_init: int = 10):
    """Fit HAR + GMM + vol thresholds on the full training set."""
    _, fitted = engineer_features(train_df, fit_models=True, hmm_n_init=hmm_n_init)
    return fitted


def _evaluate_on_test(test_df: pd.DataFrame, fitted: dict, params: dict, costs: dict = None) -> dict:
    """Engineer test features with frozen models, run signals + backtest + metrics."""
    costs = costs or {}
    test_feat, _ = engineer_features(
        test_df,
        fit_models=False,
        har_params=fitted["har_params"],
        hmm_model=fitted["hmm_model"],
        scaler=fitted["scaler"],
        label_map=fitted["label_map"],
        vol_thresholds=fitted["vol_thresholds"],
    )
    signals = generate_signals(test_feat, params, point_value=costs.get("point_value", 50.0))
    result = backtest(
        signals,
        point_value=costs.get("point_value", 50.0),
        slippage_points=costs.get("slippage_points", 0.25),
        commission_rt=costs.get("commission_rt", 2.50),
    )
    num_test_days = test_df["ts_event"].dt.date.nunique()
    metrics = compute_all_metrics(result, num_test_days)
    checks = check_acceptance(metrics)
    return {"features": test_feat, "result": result, "metrics": metrics, "checks": checks}


def run_single_backtest(
    symbol: str = "ES",
    params: Optional[dict] = None,
    hmm_n_init: int = 10,
    cached=None,
) -> dict:
    """Fit models on train and evaluate a single parameter set on the test set.

    ``cached`` may carry (train_df, test_df, inst, fitted) to avoid refitting.
    """
    if params is None:
        params = dict(PARAM_DEFAULTS)

    if cached is not None:
        train_df, test_df, inst, fitted = cached
    else:
        train_df, test_df, inst = load_instrument_data(symbol)
        fitted = _fit_train_models(train_df, hmm_n_init=hmm_n_init)

    evald = _evaluate_on_test(test_df, fitted, params, costs=costs_from_inst(inst))
    return {
        "symbol": symbol,
        "params": params,
        "metrics": evald["metrics"],
        "checks": evald["checks"],
        "result": evald["result"],
        "fitted": fitted,
        "train_df": train_df,
        "test_df": test_df,
        "har_params": {k: fitted["har_params"][k] for k in fitted["har_params"]},
        "vol_thresholds": fitted["vol_thresholds"],
    }


def run_full_pipeline(
    symbol: str = "ES",
    progress_cb: Optional[Callable[[dict], None]] = None,
    hmm_n_init_final: int = 10,
    hmm_n_init_wf: int = 3,
) -> dict:
    """Full pipeline: walk-forward optimize on train, evaluate best params on test."""
    if progress_cb:
        progress_cb({"stage": "load", "message": "Loading + validating data", "pct": 1})
    train_df, test_df, inst = load_instrument_data(symbol)
    costs = costs_from_inst(inst)

    if progress_cb:
        progress_cb({"stage": "optimize", "message": "Starting walk-forward optimization", "pct": 3})
    wf = run_walk_forward_optimization(train_df, progress_cb=progress_cb, hmm_n_init=hmm_n_init_wf, costs=costs)
    best_params = wf["best_params"]

    if progress_cb:
        progress_cb({"stage": "fit", "message": "Fitting final models on full train set", "pct": 92})
    fitted = _fit_train_models(train_df, hmm_n_init=hmm_n_init_final)

    if progress_cb:
        progress_cb({"stage": "test", "message": "Evaluating best params on test set", "pct": 96})
    evald = _evaluate_on_test(test_df, fitted, best_params, costs=costs)

    if progress_cb:
        progress_cb({"stage": "done", "message": "Pipeline complete", "pct": 100})

    return {
        "symbol": symbol,
        "best_params": best_params,
        "walk_forward": wf,
        "metrics": evald["metrics"],
        "checks": evald["checks"],
        "result": evald["result"],
        "fitted": fitted,
        "train_df": train_df,
        "test_df": test_df,
        "har_params": {k: fitted["har_params"][k] for k in fitted["har_params"]},
        "vol_thresholds": fitted["vol_thresholds"],
    }
