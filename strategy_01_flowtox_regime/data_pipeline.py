"""Data loading, validation, and splitting functions (Spec Section 1)."""

from datetime import time
from typing import Tuple

import numpy as np
import pandas as pd

from .config import CSV_FILE_PATH, TRAIN_END_INDEX


def load_data(csv_path: str = CSV_FILE_PATH) -> pd.DataFrame:
    """Load ES 5-minute RTH data from CSV (Spec 1.4).

    Returns a validated DataFrame with columns:
        ts_event (tz-aware UTC Timestamp), symbol (str), open/high/low/close
        (float64), volume (int64).
    """
    df = pd.read_csv(csv_path)

    expected_cols = ["ts_event", "symbol", "open", "high", "low", "close", "volume"]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Parse timestamps - utc=True handles timezone offsets uniformly.
    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True)

    # Drop rows with NaN symbol (zero-volume during halts).
    df = df.dropna(subset=["symbol"]).reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        if df[col].isnull().any():
            raise ValueError(f"Unexpected NaN values in column: {col}")

    ohlc_invalid = (
        (df["low"] > df["high"]) |
        (df["close"] > df["high"]) |
        (df["close"] < df["low"]) |
        (df["open"] > df["high"]) |
        (df["open"] < df["low"])
    )
    if ohlc_invalid.any():
        bad_idx = ohlc_invalid[ohlc_invalid].index.tolist()
        raise ValueError(f"OHLC inconsistency at indices: {bad_idx[:10]}")

    df["volume"] = df["volume"].astype(np.int64)
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(np.float64)

    df = df.sort_values("ts_event").reset_index(drop=True)
    return df


def verify_rth_session(df: pd.DataFrame) -> pd.DataFrame:
    """Verify all bars fall within RTH (09:30-16:00 ET) (Spec 1.5)."""
    ts_et = df["ts_event"].dt.tz_convert("America/New_York")
    bar_time = ts_et.dt.time

    rth_start = time(9, 30)
    rth_end = time(16, 0)
    in_rth = (bar_time >= rth_start) & (bar_time <= rth_end)

    if not in_rth.all():
        bad_count = int((~in_rth).sum())
        print(f"WARNING: {bad_count} bars outside RTH - removing them")
        df = df[in_rth].reset_index(drop=True)

    return df


def split_train_test(
    df: pd.DataFrame, train_end_index: int = TRAIN_END_INDEX
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into development (train) and test sets (Spec 1.6).

    For ES the canonical split is index 85599 -> train 85600 / test 36695.
    For other instruments the split index is provided by the caller.
    """
    train_df = df.iloc[: train_end_index + 1].copy()
    test_df = df.iloc[train_end_index + 1:].copy()

    assert len(train_df) + len(test_df) == len(df), "Split rows don't sum to total"
    assert train_df["ts_event"].is_monotonic_increasing, "Train data not sorted"
    assert test_df["ts_event"].is_monotonic_increasing, "Test data not sorted"
    if len(test_df) > 0:
        assert train_df["ts_event"].max() < test_df["ts_event"].min(), "Train/test overlap"

    return train_df, test_df


def validate_data(df: pd.DataFrame, tick_size: float = 0.25) -> None:
    """Run data validation assertions (Spec 1.7). Raises ValueError on failure.

    ``tick_size`` is instrument-specific (ES/MES/MNQ=0.25, M2K/MGC=0.10, MCL=0.01).
    """
    for col in ["open", "high", "low", "close", "volume"]:
        if df[col].isnull().any():
            raise ValueError(f"NaN found in {col}")

    if (df["volume"] < 0).any():
        raise ValueError("Negative volume found")

    ohlc_invalid = (
        (df["high"] < df["low"]) |
        (df["high"] < df["close"]) |
        (df["close"] < df["low"]) |
        (df["high"] < df["open"]) |
        (df["open"] < df["low"])
    )
    if ohlc_invalid.any():
        raise ValueError("OHLC anomaly found")

    if not df["ts_event"].is_monotonic_increasing:
        raise ValueError("Timestamps not monotonic increasing")

    # Tick-size sampling check on up to 1000 random samples.
    rng = np.random.default_rng(42)
    n = len(df)
    sample_idx = rng.choice(n, size=min(1000, n), replace=False)
    closes = df["close"].values[sample_idx]
    remainder = np.abs(np.round(closes / tick_size) * tick_size - closes)
    if (remainder > 1e-6).any():
        raise ValueError(f"Price not a multiple of tick size ({tick_size})")
