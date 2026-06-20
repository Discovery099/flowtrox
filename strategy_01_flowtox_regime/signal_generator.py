"""Signal generation engine (Spec Section 3).

The stateful bar-by-bar loop is implemented over numpy arrays (not df.loc) for
performance, which is essential for the walk-forward grid search. The logic is
identical to Spec 3.7.
"""

import numpy as np
import pandas as pd

from .config import (
    SIGNAL_FLAT,
    SIGNAL_LONG,
    SIGNAL_SHORT,
    MAX_SIZE,
    ACCOUNT_VALUE,
    SIGMA_TARGET,
    CONTRACT_SPECS,
    HMM_LOOKBACK,
)


def compute_position_size(
    har_forecast: float, close_price: float, point_value: float = CONTRACT_SPECS["point_value"]
) -> int:
    """Compute position size via inverse-vol targeting (Spec 3.6)."""
    forecast_vol = np.sqrt(max(har_forecast, 1e-12))
    contract_vol_usd = forecast_vol * close_price * point_value
    if contract_vol_usd <= 0:
        return 1
    size_raw = (ACCOUNT_VALUE * SIGMA_TARGET) / contract_vol_usd
    return int(np.clip(round(size_raw), 1, MAX_SIZE))


def generate_signals(
    features_df: pd.DataFrame,
    params: dict,
    point_value: float = CONTRACT_SPECS["point_value"],
) -> pd.DataFrame:
    """Generate trading signals from engineered features (Spec 3.7).

    Adds columns: signal, position_size, entry_price, hold_bars, bars_in_trade.
    Enforces: tau1/tau2 posterior gates, BVC direction, non-high vol regime,
    no pyramiding, time-based exit at N bars, optional regime early-exit.
    """
    df = features_df.copy()
    n = len(df)

    tau1 = params["toxic_continuation_threshold"]
    tau2 = params["toxic_reversal_threshold"]
    N = int(params["max_hold_bars"])
    regime_exit = bool(params.get("regime_exit_enabled", True))

    # Pull arrays once.
    p1 = df["hmm_state_1_posterior"].to_numpy(np.float64)
    p2 = df["hmm_state_2_posterior"].to_numpy(np.float64)
    bvc_dir = df["bvc_direction"].to_numpy(np.float64)
    vol_reg = df["vol_regime"].to_numpy(np.int64)
    har = df["har_vol_forecast"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)

    signal = np.zeros(n, dtype=np.int64)
    position_size = np.zeros(n, dtype=np.int64)
    entry_price = np.full(n, np.nan, dtype=np.float64)
    hold_bars = np.zeros(n, dtype=np.int64)
    bars_in_trade = np.zeros(n, dtype=np.int64)

    position = SIGNAL_FLAT
    hold_counter = 0
    current_size = 0
    entry_px = 0.0

    start_bar = max(20, HMM_LOOKBACK)
    for i in range(start_bar, n - 1):
        if position != SIGNAL_FLAT:
            hold_counter += 1
            bars_in_trade[i] = hold_counter

            # Time-based exit.
            if hold_counter >= N:
                signal[i] = SIGNAL_FLAT
                position_size[i] = current_size
                hold_bars[i] = hold_counter
                position = SIGNAL_FLAT
                hold_counter = 0
                current_size = 0
                continue

            # Regime-based early exit.
            if regime_exit:
                if position == SIGNAL_LONG and p1[i] < tau1 / 2:
                    signal[i] = SIGNAL_FLAT
                    position_size[i] = current_size
                    hold_bars[i] = hold_counter
                    position = SIGNAL_FLAT
                    hold_counter = 0
                    current_size = 0
                    continue
                if position == SIGNAL_SHORT and p2[i] < tau2 / 2:
                    signal[i] = SIGNAL_FLAT
                    position_size[i] = current_size
                    hold_bars[i] = hold_counter
                    position = SIGNAL_FLAT
                    hold_counter = 0
                    current_size = 0
                    continue

            # Hold position.
            signal[i] = position
            position_size[i] = current_size
            continue

        # Flat - check entries (no pyramiding implied by position==FLAT here).
        if p1[i] > tau1 and bvc_dir[i] > 0 and vol_reg[i] != 2:
            position = SIGNAL_LONG
            hold_counter = 0
            entry_px = close[i]
            fv = np.sqrt(max(har[i], 1e-12))
            cvu = fv * close[i] * point_value
            current_size = int(np.clip(round((ACCOUNT_VALUE * SIGMA_TARGET) / cvu), 1, MAX_SIZE)) if cvu > 0 else 1
            signal[i] = SIGNAL_LONG
            position_size[i] = current_size
            entry_price[i] = entry_px
            continue

        if p2[i] > tau2 and bvc_dir[i] < 0 and vol_reg[i] != 2:
            position = SIGNAL_SHORT
            hold_counter = 0
            entry_px = close[i]
            fv = np.sqrt(max(har[i], 1e-12))
            cvu = fv * close[i] * point_value
            current_size = int(np.clip(round((ACCOUNT_VALUE * SIGMA_TARGET) / cvu), 1, MAX_SIZE)) if cvu > 0 else 1
            signal[i] = SIGNAL_SHORT
            position_size[i] = current_size
            entry_price[i] = entry_px
            continue

        signal[i] = SIGNAL_FLAT
        position_size[i] = 0

    df["signal"] = signal
    df["position_size"] = position_size
    df["entry_price"] = entry_price
    df["hold_bars"] = hold_bars
    df["bars_in_trade"] = bars_in_trade
    return df
