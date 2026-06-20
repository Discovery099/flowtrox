"""Walk-forward optimization and validation framework (Spec Section 5).

Key performance refactor vs the spec's reference ``validate()``: engineered
features are param-independent, so for each walk-forward window we engineer the
test-window features ONCE (a single GMM/HAR fit per window) and then iterate the
486 parameter combinations running only signal-generation + backtest. This
turns thousands of GMM refits into a handful, making live runs tractable while
producing identical results.
"""

import itertools
from typing import Callable, Optional

import numpy as np
import pandas as pd

from .config import PARAM_SEARCH_SPACE, WALK_FORWARD
from .features import engineer_features
from .signal_generator import generate_signals
from .backtest import backtest
from .metrics import compute_sharpe, compute_win_rate

MIN_TRADES_PER_WINDOW = 100
MIN_WIN_RATE = 0.50
REJECT_SCORE = -999.0


def build_param_grid() -> list:
    """Build the full 9x9x6 = 486 parameter grid (Spec 5.3)."""
    grid = []
    for t1, t2, n in itertools.product(
        PARAM_SEARCH_SPACE["toxic_continuation_threshold"],
        PARAM_SEARCH_SPACE["toxic_reversal_threshold"],
        PARAM_SEARCH_SPACE["max_hold_bars"],
    ):
        grid.append({
            "toxic_continuation_threshold": float(t1),
            "toxic_reversal_threshold": float(t2),
            "max_hold_bars": int(n),
            "regime_exit_enabled": True,
        })
    return grid


def _window_schedule(n_bars: int) -> list:
    """Generate (w_start, w_train_end, w_test_end) walk-forward windows."""
    tw = WALK_FORWARD["train_window_bars"]
    vw = WALK_FORWARD["test_window_bars"]
    step = WALK_FORWARD["step_size_bars"]
    windows = []
    w_start = 0
    while w_start + tw + vw <= n_bars:
        windows.append((w_start, w_start + tw, w_start + tw + vw))
        w_start += step
    return windows


def run_walk_forward_optimization(
    train_df: pd.DataFrame,
    progress_cb: Optional[Callable[[dict], None]] = None,
    hmm_n_init: int = 3,
    costs: Optional[dict] = None,
) -> dict:
    """Optimize the 3 parameters (tau1, tau2, N) via walk-forward analysis.

    Returns dict with best_params, best_score, per-combo scores, and the
    sensitivity grid used for plotting.
    """
    windows = _window_schedule(len(train_df))
    if len(windows) < 3:
        # Failure handling FH-2: shrink windows.
        WALK_FORWARD["train_window_bars"] = 20_000
        WALK_FORWARD["test_window_bars"] = 4_000
        WALK_FORWARD["step_size_bars"] = 2_000
        windows = _window_schedule(len(train_df))

    # Precompute per-window engineered TEST features (param-independent).
    if progress_cb:
        progress_cb({"stage": "features", "message": f"Engineering {len(windows)} walk-forward windows", "pct": 5})

    window_feats = []
    for wi, (w_start, w_train_end, w_test_end) in enumerate(windows):
        w_train = train_df.iloc[w_start:w_train_end]
        w_test = train_df.iloc[w_train_end:w_test_end]
        _, w_fitted = engineer_features(w_train, fit_models=True, hmm_n_init=hmm_n_init)
        w_test_feat, _ = engineer_features(
            w_test,
            fit_models=False,
            har_params=w_fitted["har_params"],
            hmm_model=w_fitted["hmm_model"],
            scaler=w_fitted["scaler"],
            label_map=w_fitted["label_map"],
            vol_thresholds=w_fitted["vol_thresholds"],
        )
        n_days = w_test["ts_event"].dt.date.nunique()
        window_feats.append((w_test_feat, n_days))
        if progress_cb:
            progress_cb({
                "stage": "features",
                "message": f"Window {wi + 1}/{len(windows)} features ready",
                "pct": 5 + int(20 * (wi + 1) / len(windows)),
            })

    grid = build_param_grid()
    total = len(grid)
    best_params = None
    best_score = -np.inf
    combo_scores = []

    costs = costs or {}
    pv = costs.get("point_value", 50.0)
    slip = costs.get("slippage_points", 0.25)
    comm = costs.get("commission_rt", 2.50)

    matrix_rows = []  # raw per-window Sharpe for every combo (for PBO/CSCV)
    trial_sharpes = []  # avg test Sharpe per combo (for Deflated Sharpe)

    for ci, params in enumerate(grid):
        window_sharpes = []
        raw_sharpes = []
        for (w_test_feat, n_days) in window_feats:
            sigs = generate_signals(w_test_feat, params, point_value=pv)
            res = backtest(sigs, point_value=pv, slippage_points=slip, commission_rt=comm)
            tl = res["trade_log"]
            raw = compute_sharpe(res["daily_returns"])
            raw_sharpes.append(float(raw))
            if len(tl) >= MIN_TRADES_PER_WINDOW:
                wr = compute_win_rate(tl)
                if wr >= MIN_WIN_RATE:
                    window_sharpes.append(raw)
                else:
                    window_sharpes.append(REJECT_SCORE)
            else:
                window_sharpes.append(REJECT_SCORE)

        matrix_rows.append(raw_sharpes)
        valid = [s for s in window_sharpes if s > -900]
        avg_sharpe = float(np.mean(valid)) if valid else REJECT_SCORE
        trial_sharpes.append(float(np.mean(raw_sharpes)))
        combo_scores.append({
            "toxic_continuation_threshold": params["toxic_continuation_threshold"],
            "toxic_reversal_threshold": params["toxic_reversal_threshold"],
            "max_hold_bars": params["max_hold_bars"],
            "avg_sharpe": avg_sharpe,
            "n_valid_windows": len(valid),
        })
        if avg_sharpe > best_score:
            best_score = avg_sharpe
            best_params = dict(params)

        if progress_cb and (ci % 20 == 0 or ci == total - 1):
            progress_cb({
                "stage": "optimize",
                "message": f"Tested {ci + 1}/{total} param combos",
                "pct": 25 + int(65 * (ci + 1) / total),
                "best_score": None if best_score == -np.inf else round(best_score, 4),
            })

    if best_params is None:
        # FH-3: relax and pick central defaults.
        best_params = {
            "toxic_continuation_threshold": 0.55,
            "toxic_reversal_threshold": 0.55,
            "max_hold_bars": 15,
            "regime_exit_enabled": True,
        }
        best_score = REJECT_SCORE

    # window_combo_matrix: shape (S windows, C combos) for CSCV/PBO.
    window_combo_matrix = np.asarray(matrix_rows, dtype=float).T.tolist() if matrix_rows else []

    return {
        "best_params": best_params,
        "best_score": float(best_score),
        "combo_scores": combo_scores,
        "n_windows": len(windows),
        "n_combos": total,
        "window_combo_matrix": window_combo_matrix,
        "trial_sharpes": trial_sharpes,
    }
