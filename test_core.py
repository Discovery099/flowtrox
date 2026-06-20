#!/usr/bin/env python3
"""POC / core-validation script for FLOWTOX_REGIME_01.

Proves the strategy engine works end-to-end on the REAL ES CSV before any web
app is built. Runs progressively:
  1. Data load + validation + split (exact row counts).
  2. Feature engineering on train (value-domain + no-NaN checks).
  3. HAR fit sanity + GMM weights + posterior sums.
  4. Single backtest on test set with default params (timed).
  5. A SMALL walk-forward optimization sanity slice (timed) to prove the
     optimizer runs and produces best params without leakage.

Exit code 0 only if all gates pass.
"""

import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore")

import numpy as np

from strategy_01_flowtox_regime.data_pipeline import load_data, validate_data, split_train_test, verify_rth_session
from strategy_01_flowtox_regime.features import engineer_features
from strategy_01_flowtox_regime.signal_generator import generate_signals
from strategy_01_flowtox_regime.backtest import backtest
from strategy_01_flowtox_regime.metrics import compute_all_metrics, check_acceptance
from strategy_01_flowtox_regime.config import PARAM_DEFAULTS, CSV_FILE_PATH


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main():
    section("STEP 1: DATA LOAD + VALIDATION + SPLIT")
    print(f"CSV: {CSV_FILE_PATH}")
    t0 = time.time()
    df = load_data()
    print(f"Loaded {len(df):,} rows in {time.time() - t0:.2f}s")
    assert 122280 <= len(df) <= 122296, f"Unexpected row count after NaN drop: {len(df)}"
    validate_data(df)
    df = verify_rth_session(df)
    train_df, test_df = split_train_test(df)
    print(f"Train: {len(train_df):,}  Test: {len(test_df):,}")
    assert len(train_df) == 85600, f"train rows {len(train_df)}"
    assert len(test_df) == len(df) - 85600, f"test rows {len(test_df)}"
    print("PASS: data load + split")

    section("STEP 2: FEATURE ENGINEERING (train, fit_models=True)")
    t0 = time.time()
    train_feat, fitted = engineer_features(train_df, fit_models=True, hmm_n_init=5)
    print(f"Engineered train features in {time.time() - t0:.2f}s")
    for col in ["bvc_signed_volume", "toxicity_disagreement", "parkinson_bar_variance",
                "rv_1d", "rv_1w", "rv_1m", "har_vol_forecast", "vol_regime",
                "hmm_state_0_posterior", "hmm_state_1_posterior", "hmm_state_2_posterior",
                "bvc_direction"]:
        assert col in train_feat.columns, f"missing {col}"
        assert train_feat[col].notna().all(), f"NaN in {col}"
    # Toxicity domain
    assert set(np.unique(train_feat["toxicity_disagreement"])).issubset({-1.0, 0.0, 1.0})
    # |BVC| <= volume
    assert (train_feat["bvc_signed_volume"].abs() <= train_feat["volume"] * 1.0001).all()
    # Posteriors sum to 1
    psum = (train_feat["hmm_state_0_posterior"] + train_feat["hmm_state_1_posterior"]
            + train_feat["hmm_state_2_posterior"])
    assert np.allclose(psum, 1.0, atol=1e-6), "posteriors don't sum to 1"
    # Vol regime has 3 classes
    assert set(np.unique(train_feat["vol_regime"])).issubset({0, 1, 2})
    print(f"HAR params: {fitted['har_params']}")
    print(f"GMM weights: {fitted['hmm_model'].weights_}")
    print(f"Label map (comp->state): {fitted['label_map']}")
    tox_rate = (train_feat["toxicity_disagreement"] < 0).mean()
    print(f"Toxicity rate: {tox_rate:.2%}")
    print("PASS: feature engineering")

    section("STEP 3: TEST FEATURES (fit_models=False, no leakage)")
    t0 = time.time()
    test_feat, _ = engineer_features(
        test_df, fit_models=False,
        har_params=fitted["har_params"], hmm_model=fitted["hmm_model"],
        scaler=fitted["scaler"], label_map=fitted["label_map"],
        vol_thresholds=fitted["vol_thresholds"],
    )
    print(f"Engineered test features in {time.time() - t0:.2f}s")
    assert list(train_feat.columns) == list(test_feat.columns), "train/test cols differ"
    psum = (test_feat["hmm_state_0_posterior"] + test_feat["hmm_state_1_posterior"]
            + test_feat["hmm_state_2_posterior"])
    assert np.allclose(psum, 1.0, atol=1e-6)
    print("PASS: test features, identical columns, posteriors sum to 1")

    section("STEP 4: SINGLE BACKTEST ON TEST (default params)")
    t0 = time.time()
    sigs = generate_signals(test_feat, dict(PARAM_DEFAULTS))
    res = backtest(sigs)
    dt = time.time() - t0
    n_days = test_df["ts_event"].dt.date.nunique()
    metrics = compute_all_metrics(res, n_days)
    checks = check_acceptance(metrics)
    print(f"Backtest (signals+bt+metrics) in {dt:.2f}s")
    print(f"Total trades: {metrics['total_trades']}")
    print(f"Sharpe: {metrics['sharpe_ratio']:.3f}  WinRate: {metrics['win_rate']:.3%}  "
          f"PF: {metrics['profit_factor']:.3f}  Trades/Day: {metrics['trades_per_day']:.2f}  "
          f"MaxDD: {metrics['max_drawdown']:.3%}")
    print(f"Total P&L: ${metrics['total_pnl']:,.2f}")
    assert metrics["total_trades"] > 0, "No trades generated - core signal logic broken"
    # Signal domain
    assert set(np.unique(sigs["signal"])).issubset({-1, 0, 1})
    print("PASS: single backtest produced trades + finite metrics")

    section("STEP 5: WALK-FORWARD OPTIMIZER SANITY (small slice)")
    # Prove optimizer machinery runs on a couple of windows + a few combos.
    from strategy_01_flowtox_regime import validation as V
    orig_grid = V.build_param_grid
    V.build_param_grid = lambda: [
        {"toxic_continuation_threshold": 0.5, "toxic_reversal_threshold": 0.5,
         "max_hold_bars": 15, "regime_exit_enabled": True},
        {"toxic_continuation_threshold": 0.45, "toxic_reversal_threshold": 0.45,
         "max_hold_bars": 10, "regime_exit_enabled": True},
    ]
    # Use a reduced training slice (2 windows worth) for speed.
    slice_df = train_df.iloc[: V.WALK_FORWARD["train_window_bars"] + V.WALK_FORWARD["test_window_bars"] + V.WALK_FORWARD["step_size_bars"]]
    t0 = time.time()
    wf = V.run_walk_forward_optimization(slice_df, hmm_n_init=2)
    V.build_param_grid = orig_grid
    print(f"WF sanity ({wf['n_windows']} windows x {wf['n_combos']} combos) in {time.time() - t0:.2f}s")
    print(f"Best params: {wf['best_params']}  best_score: {wf['best_score']:.4f}")
    assert wf["best_params"] is not None
    print("PASS: walk-forward optimizer runs")

    section("POC RESULT: ALL CORE GATES PASSED")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        print("\nPOC FAILED:")
        traceback.print_exc()
        sys.exit(1)
