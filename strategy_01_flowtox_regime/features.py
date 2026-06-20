"""Feature engineering functions (Spec Section 2).

Features are computed in the exact dependency order specified by the spec.
All model-fitting (HAR, GMM, vol-regime thresholds) happens only when
``fit_models=True`` (training); otherwise pre-fitted objects are reused so
the test set never leaks information into model parameters.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from .config import BARS_PER_DAY, HAR_WEEKLY_WINDOW, HAR_MONTHLY_WINDOW, HMM_LOOKBACK, HMM_N_INIT as _HMM_N_INIT
from .har_model import fit_har_model, compute_har_vol_forecast
from .hmm_model import fit_hmm_model, compute_hmm_posteriors


# ---------------------------------------------------------------------------
# 2.1 BVC Signed Volume
# ---------------------------------------------------------------------------
def compute_bvc_signed_volume(df: pd.DataFrame) -> pd.Series:
    """Compute BVC-signed volume for each bar (Spec 2.1)."""
    numerator = 2 * df["close"] - df["low"] - df["high"]
    denominator = df["high"] - df["low"]

    with np.errstate(divide="ignore", invalid="ignore"):
        bvc = df["volume"] * numerator / denominator

    zero_range_mask = denominator == 0
    bvc = bvc.where(~zero_range_mask, 0.0)

    assert bvc.notna().all(), "NaN in bvc_signed_volume"
    assert (bvc.abs() <= df["volume"] * 1.0001).all(), "BVC magnitude exceeds volume"
    return bvc.astype(np.float64)


# ---------------------------------------------------------------------------
# 2.2 Tick Rule Direction
# ---------------------------------------------------------------------------
def compute_tick_rule_direction(df: pd.DataFrame) -> pd.Series:
    """Compute tick-rule direction for each bar (Spec 2.2)."""
    price_diff = df["close"].diff()
    tick_dir = np.sign(price_diff)
    tick_dir.iloc[0] = 1.0                       # First bar convention.
    tick_dir = tick_dir.where(tick_dir != 0, 1.0)  # Unchanged price -> up-tick.
    assert tick_dir.isin([-1.0, 1.0]).all(), "Invalid tick_rule_direction values"
    return tick_dir.astype(np.float64)


# ---------------------------------------------------------------------------
# 2.3 Toxicity Disagreement
# ---------------------------------------------------------------------------
def compute_toxicity_disagreement(
    bvc_signed_volume: pd.Series, tick_rule_direction: pd.Series
) -> pd.Series:
    """Compute toxicity disagreement: sign(BVC) * tick_direction (Spec 2.3)."""
    bvc_sign = np.sign(bvc_signed_volume)
    tox = bvc_sign * tick_rule_direction
    tox = tox.where(bvc_sign != 0, 0.0)
    assert tox.isin([-1.0, 0.0, 1.0]).all(), "Invalid toxicity_disagreement values"
    return tox.astype(np.float64)


# ---------------------------------------------------------------------------
# 2.4 Parkinson Bar Variance
# ---------------------------------------------------------------------------
def compute_parkinson_bar_variance(df: pd.DataFrame) -> pd.Series:
    """Compute per-bar Parkinson variance contribution (Spec 2.4)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        log_ratio_sq = (np.log(df["high"] / df["low"])) ** 2
    parkinson_bar = log_ratio_sq / (4.0 * np.log(2.0))
    zero_range = df["high"] == df["low"]
    parkinson_bar = parkinson_bar.where(~zero_range, 0.0)
    assert parkinson_bar.notna().all(), "NaN in parkinson_bar_variance"
    assert (parkinson_bar >= 0).all(), "Negative Parkinson variance"
    return parkinson_bar.astype(np.float64)


# ---------------------------------------------------------------------------
# 2.5 Parkinson Realized Variances (1D / 1W / 1M)
# ---------------------------------------------------------------------------
def compute_parkinson_realized_variances(
    df: pd.DataFrame, parkinson_bar: pd.Series
):
    """Compute daily, weekly, and monthly Parkinson realized variances (Spec 2.5).

    Days are grouped by the bar's actual (UTC) calendar date rather than a fixed
    bars-per-day count, so the aggregation is correct for any session length
    (ES/MES/MNQ/M2K = 78 bars, MGC ~62, MCL ~66).

    Returns ``(per_bar_df, daily_df)`` where ``per_bar_df`` has rv_1d/rv_1w/rv_1m
    broadcast to every bar and ``daily_df`` is indexed by date (one row per day).
    """
    date_series = pd.Series(df["ts_event"].dt.date.values, index=df.index)
    pbar = pd.Series(parkinson_bar.values, index=df.index)

    daily_rv = pbar.groupby(date_series).sum().sort_index()
    weekly_rv = daily_rv.rolling(window=HAR_WEEKLY_WINDOW, min_periods=1).mean()
    monthly_rv = daily_rv.rolling(window=HAR_MONTHLY_WINDOW, min_periods=1).mean()

    daily_df = pd.DataFrame(
        {"rv_1d": daily_rv, "rv_1w": weekly_rv, "rv_1m": monthly_rv}
    )

    rv_1d = date_series.map(daily_rv)
    rv_1w = date_series.map(weekly_rv)
    rv_1m = date_series.map(monthly_rv)

    per_bar = pd.DataFrame(
        {"rv_1d": rv_1d, "rv_1w": rv_1w, "rv_1m": rv_1m}, index=df.index
    )
    per_bar = per_bar.ffill().bfill()
    per_bar = per_bar.fillna(float(per_bar["rv_1d"].mean()))

    for col in ["rv_1d", "rv_1w", "rv_1m"]:
        assert (per_bar[col] >= 0).all(), f"Negative values in {col}"
    return per_bar, daily_df


# ---------------------------------------------------------------------------
# 2.7 Volatility Regime Classification
# ---------------------------------------------------------------------------
def compute_vol_regime(
    har_vol_forecast: pd.Series, p33: float, p67: float
) -> pd.Series:
    """Classify volatility into 3 regimes using fixed thresholds (Spec 2.7).

    Thresholds (p33/p67) are computed once on the training data and reused
    on the test data to avoid leakage.
    """
    regime = pd.Series(1, index=har_vol_forecast.index, dtype=np.int64)  # medium
    regime[har_vol_forecast <= p33] = 0  # low
    regime[har_vol_forecast > p67] = 2   # high
    return regime


# ---------------------------------------------------------------------------
# 2.8 HMM Observable Features
# ---------------------------------------------------------------------------
def compute_hmm_observable_features(
    df: pd.DataFrame,
    toxicity: pd.Series,
    bvc: pd.Series,
    parkinson_bar: pd.Series,
) -> pd.DataFrame:
    """Compute the 5 observable features for HMM state inference (Spec 2.8)."""
    features = pd.DataFrame(index=df.index)

    is_toxic = (toxicity < 0).astype(float)
    features["hmm_feat_1"] = is_toxic.rolling(window=HMM_LOOKBACK, min_periods=1).mean()

    bvc_sign = np.sign(bvc)
    features["hmm_feat_2"] = bvc_sign.rolling(window=HMM_LOOKBACK, min_periods=1).mean()

    features["hmm_feat_3"] = np.log1p(
        parkinson_bar.rolling(window=HMM_LOOKBACK, min_periods=1).mean()
    )

    features["hmm_feat_4"] = df["close"].diff(5)

    vol_mean_20 = df["volume"].rolling(window=HMM_LOOKBACK, min_periods=1).mean()
    features["hmm_feat_5"] = df["volume"] / vol_mean_20.clip(lower=1)

    features = features.fillna(0.0)
    return features.astype(np.float64)


# ---------------------------------------------------------------------------
# 2.13 Master Feature Engineering Pipeline
# ---------------------------------------------------------------------------
def engineer_features(
    df: pd.DataFrame,
    har_params: dict = None,
    hmm_model=None,
    scaler=None,
    label_map: list = None,
    vol_thresholds: dict = None,
    fit_models: bool = True,
    hmm_n_init: int = None,
    regime_model: str = "gmm",
) -> Tuple[pd.DataFrame, dict]:
    """Master feature engineering pipeline (Spec 2.13).

    When ``fit_models=True`` the HAR model, GMM/HMM, and vol-regime thresholds
    are fitted on ``df`` and returned in ``fitted``. When ``fit_models=False``
    the provided pre-fitted objects are reused (no leakage).
    """
    result = df.copy()
    fitted = {}

    # 2.1 - 2.4 base features
    result["bvc_signed_volume"] = compute_bvc_signed_volume(result)
    result["tick_rule_direction"] = compute_tick_rule_direction(result)
    result["toxicity_disagreement"] = compute_toxicity_disagreement(
        result["bvc_signed_volume"], result["tick_rule_direction"]
    )
    result["parkinson_bar_variance"] = compute_parkinson_bar_variance(result)

    # 2.5 realized variances (grouped by actual date; instrument-agnostic)
    rv_df, daily_rv = compute_parkinson_realized_variances(
        result, result["parkinson_bar_variance"]
    )
    result = pd.concat([result, rv_df], axis=1)

    # 2.6 HAR forecast
    if fit_models:
        har_params = fit_har_model(daily_rv)
        fitted["har_params"] = har_params
    result["har_vol_forecast"] = compute_har_vol_forecast(
        result["rv_1d"], result["rv_1w"], result["rv_1m"], har_params
    )

    # 2.7 vol regime
    if fit_models:
        vol_thresholds = {
            "p33": float(result["har_vol_forecast"].quantile(0.33)),
            "p67": float(result["har_vol_forecast"].quantile(0.67)),
        }
        fitted["vol_thresholds"] = vol_thresholds
    result["vol_regime"] = compute_vol_regime(
        result["har_vol_forecast"], vol_thresholds["p33"], vol_thresholds["p67"]
    )

    # 2.8 HMM observable features
    hmm_features = compute_hmm_observable_features(
        result,
        result["toxicity_disagreement"],
        result["bvc_signed_volume"],
        result["parkinson_bar_variance"],
    )

    # 2.9 HMM posteriors
    if fit_models:
        _ninit = hmm_n_init if hmm_n_init is not None else _HMM_N_INIT
        hmm_model, scaler, label_map = fit_hmm_model(
            hmm_features, n_init=_ninit, model_type=regime_model
        )
        fitted["hmm_model"] = hmm_model
        fitted["scaler"] = scaler
        fitted["label_map"] = label_map
        fitted["regime_model"] = regime_model

    posteriors = compute_hmm_posteriors(
        hmm_features, hmm_model, scaler, label_map, model_type=regime_model
    )
    result = pd.concat([result, posteriors], axis=1)

    # 2.10 BVC direction
    result["bvc_direction"] = np.sign(result["bvc_signed_volume"])

    # 2.11 recent momentum
    result["recent_momentum"] = result["close"].diff(5)

    # 2.12 signal-generation composite features
    vol_not_high = (result["vol_regime"] != 2).astype(float)
    result["toxic_cont_signal"] = (
        result["hmm_state_1_posterior"] * result["bvc_direction"] * vol_not_high
    )
    result["toxic_rev_signal"] = (
        result["hmm_state_2_posterior"] * (-result["bvc_direction"]) * vol_not_high
    )

    return result, fitted
