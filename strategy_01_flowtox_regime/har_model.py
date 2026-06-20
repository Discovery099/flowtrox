"""HAR-RV model fitting and forecasting (Spec Section 2.6)."""

import numpy as np
import pandas as pd

from .config import BARS_PER_DAY


def fit_har_model(daily_rv: pd.DataFrame) -> dict:
    """Fit HAR-RV model via OLS on daily-level training data (Spec 2.6).

    ``daily_rv`` is indexed by calendar date with columns rv_1d/rv_1w/rv_1m
    (one row per trading day). Target is next day's RV_1D. Grouping by actual
    date (instead of assuming a fixed bars-per-day) makes the model correct for
    instruments with different session lengths (e.g. gold, crude).
    Returns dict with keys: intercept, beta_d, beta_w, beta_m, r_squared.
    """
    daily_data = daily_rv[["rv_1d", "rv_1w", "rv_1m"]].dropna().sort_index()

    # Target: next day's RV.
    y = daily_data["rv_1d"].shift(-1).dropna()
    X = daily_data[["rv_1d", "rv_1w", "rv_1m"]].loc[y.index]

    X_with_const = np.column_stack(
        [np.ones(len(X)), X["rv_1d"].values, X["rv_1w"].values, X["rv_1m"].values]
    )

    beta, _, _, _ = np.linalg.lstsq(X_with_const, y.values, rcond=None)

    # R-squared of the fit.
    y_hat = X_with_const @ beta
    ss_res = float(np.sum((y.values - y_hat) ** 2))
    ss_tot = float(np.sum((y.values - y.values.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    params = {
        "intercept": float(beta[0]),
        "beta_d": float(beta[1]),
        "beta_w": float(beta[2]),
        "beta_m": float(beta[3]),
        "r_squared": float(r_squared),
    }

    # Non-negativity constraint (Spec SC-3): clamp negative coeffs to 0.001.
    for k in ["intercept", "beta_d", "beta_w", "beta_m"]:
        if params[k] < 0:
            params[k] = 0.001

    return params


def compute_har_vol_forecast(
    rv_1d: pd.Series, rv_1w: pd.Series, rv_1m: pd.Series, har_params: dict
) -> pd.Series:
    """Compute HAR volatility forecast for next period (Spec 2.6)."""
    forecast = (
        har_params["intercept"]
        + har_params["beta_d"] * rv_1d
        + har_params["beta_w"] * rv_1w
        + har_params["beta_m"] * rv_1m
    )
    forecast = forecast.clip(lower=1e-12)
    return forecast.astype(np.float64)
