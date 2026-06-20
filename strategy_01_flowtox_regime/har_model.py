"""HAR-RV model fitting and forecasting (Spec Section 2.6)."""

import numpy as np
import pandas as pd

from .config import BARS_PER_DAY


def fit_har_model(rv_1d: pd.Series, rv_1w: pd.Series, rv_1m: pd.Series) -> dict:
    """Fit HAR-RV model via OLS on training data (Spec 2.6).

    Target: RV_1D[t+1] (next day's realized variance).
    Features: [1, RV_1D[t], RV_1W[t], RV_1M[t]] at the daily level.
    Returns dict with keys: intercept, beta_d, beta_w, beta_m, r_squared.
    """
    daily_data = pd.DataFrame({
        "rv_1d": rv_1d.iloc[::BARS_PER_DAY].reset_index(drop=True),
        "rv_1w": rv_1w.iloc[::BARS_PER_DAY].reset_index(drop=True),
        "rv_1m": rv_1m.iloc[::BARS_PER_DAY].reset_index(drop=True),
    }).dropna()

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
