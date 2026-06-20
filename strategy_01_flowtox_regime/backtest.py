"""Backtesting engine (Spec Section 4).

Close-to-close execution on 5-minute bars with 1-tick/side slippage and
$2.50 round-turn commission. Numpy arrays drive the per-bar loop for speed.
"""

import numpy as np
import pandas as pd

from .config import (
    SIGNAL_FLAT,
    SIGNAL_LONG,
    CONTRACT_SPECS,
    SLIPPAGE_POINTS,
    COMMISSION_RT,
)


def backtest(
    df_with_signals: pd.DataFrame,
    point_value: float = CONTRACT_SPECS["point_value"],
    slippage_points: float = SLIPPAGE_POINTS,
    commission_rt: float = COMMISSION_RT,
) -> dict:
    """Run a complete backtest on a signal-generated DataFrame (Spec 4.7).

    Returns dict with: equity_curve, trade_log, daily_returns, bar_pnl,
    df_with_pnl.
    """
    df = df_with_signals
    n = len(df)

    close = df["close"].to_numpy(np.float64)
    sig = df["signal"].to_numpy(np.int64)
    psize = df["position_size"].to_numpy(np.int64)
    hold_bars_col = df["hold_bars"].to_numpy(np.int64) if "hold_bars" in df else np.zeros(n, np.int64)
    ts = df["ts_event"].to_numpy()

    bar_pnl = np.zeros(n, dtype=np.float64)
    position_array = np.zeros(n, dtype=np.int64)
    size_array = np.zeros(n, dtype=np.int64)

    trades = []
    trade_id = 0

    in_position = False
    position_direction = 0
    current_size = 0
    entry_bar_idx = -1
    entry_price_slipped = 0.0
    entry_signal_type = ""
    slippage_entry = 0.0

    for i in range(1, n):
        signal = sig[i]

        # Mark-to-market for an existing position.
        if in_position:
            price_change = close[i] - close[i - 1]
            bar_pnl[i] = position_direction * current_size * point_value * price_change

        if not in_position and signal != SIGNAL_FLAT:
            # New entry.
            in_position = True
            position_direction = int(signal)
            current_size = int(psize[i])
            entry_bar_idx = i
            if position_direction == SIGNAL_LONG:
                entry_price_slipped = close[i] + slippage_points
                entry_signal_type = "toxic_cont"
            else:
                entry_price_slipped = close[i] - slippage_points
                entry_signal_type = "toxic_rev"
            slippage_entry = slippage_points * current_size * point_value

        elif in_position and signal == SIGNAL_FLAT:
            # Exit.
            if position_direction == SIGNAL_LONG:
                exit_price_slipped = close[i] - slippage_points
                gross_pnl = (exit_price_slipped - entry_price_slipped) * current_size * point_value
            else:
                exit_price_slipped = close[i] + slippage_points
                gross_pnl = (entry_price_slipped - exit_price_slipped) * current_size * point_value

            slippage_exit = slippage_points * current_size * point_value
            total_slippage = slippage_entry + slippage_exit
            commission = current_size * commission_rt
            net_pnl = gross_pnl - commission - total_slippage

            hold = i - entry_bar_idx
            planned = int(hold_bars_col[i]) if hold_bars_col[i] > 0 else hold
            exit_reason = "time_exit" if hold >= planned else "regime_exit"

            trade_id += 1
            trades.append({
                "trade_id": trade_id,
                "entry_bar": int(entry_bar_idx),
                "exit_bar": int(i),
                "entry_time": pd.Timestamp(ts[entry_bar_idx]),
                "exit_time": pd.Timestamp(ts[i]),
                "direction": "LONG" if position_direction == SIGNAL_LONG else "SHORT",
                "size": int(current_size),
                "entry_price": float(entry_price_slipped),
                "exit_price": float(exit_price_slipped),
                "hold_bars": int(hold),
                "gross_pnl": float(gross_pnl),
                "commission": float(commission),
                "slippage_cost": float(total_slippage),
                "net_pnl": float(net_pnl),
                "entry_signal": entry_signal_type,
                "exit_reason": exit_reason,
            })

            in_position = False
            position_direction = 0
            current_size = 0
            entry_bar_idx = -1

        position_array[i] = position_direction if in_position else 0
        size_array[i] = current_size if in_position else 0

    equity = np.cumsum(bar_pnl)

    out = df.copy()
    out["bar_pnl"] = bar_pnl
    out["position"] = position_array
    out["size"] = size_array
    out["equity"] = equity

    trade_log = pd.DataFrame(trades)

    out["date"] = out["ts_event"].dt.date
    daily_pnl = out.groupby("date")["bar_pnl"].sum()
    daily_returns = daily_pnl / 100_000.0  # ACCOUNT_VALUE reference

    return {
        "equity_curve": pd.Series(equity, index=df.index),
        "trade_log": trade_log,
        "daily_returns": daily_returns,
        "bar_pnl": pd.Series(bar_pnl, index=df.index),
        "df_with_pnl": out,
    }
