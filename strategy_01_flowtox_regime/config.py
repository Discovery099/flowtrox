"""Configuration module - all hard-coded constants and default parameters.

NO values in this file should be modified during optimization.
Only the 3 parameters in PARAM_SEARCH_SPACE are optimized: tau1, tau2, N.

Multi-instrument ready: INSTRUMENTS registry maps a symbol -> data path +
contract specs. ES is provided now; additional futures CSVs (identical schema)
can be registered later without touching the core engine.
"""

import os
import numpy as np

# ============================================================================
# DATA SPECIFICATION
# ============================================================================
# Resolve a sensible default CSV path. The original spec path is
# /mnt/agents/upload/ES_5min_RTH_6year.csv; in this environment the uploaded
# file lives at /app/ES_5min_RTH_6year.csv. We probe a few candidates.
_CSV_CANDIDATES = [
    os.environ.get("ES_CSV_PATH", ""),
    "/app/ES_5min_RTH_6year.csv",
    "/app/backend/data/ES_5min_RTH_6year.csv",
    "/mnt/agents/upload/ES_5min_RTH_6year.csv",
]


def _resolve_default_csv() -> str:
    for p in _CSV_CANDIDATES:
        if p and os.path.exists(p):
            return p
    # Fall back to the canonical app path even if missing (clear error later).
    return "/app/ES_5min_RTH_6year.csv"


CSV_FILE_PATH: str = _resolve_default_csv()

BARS_PER_DAY: int = 78          # 09:30-16:00 ET at 5-min frequency
TRAIN_END_INDEX: int = 85599    # Inclusive end index for train

# ============================================================================
# CONTRACT SPECIFICATIONS (E-mini S&P 500)
# ============================================================================
CONTRACT_SPECS = {
    "tick_size": 0.25,
    "tick_value": 12.50,
    "point_value": 50.00,
    "multiplier": 50.00,
    "symbol": "ES",
    "exchange": "CME",
    "currency": "USD",
}

# ============================================================================
# SLIPPAGE AND COMMISSION
# ============================================================================
SLIPPAGE_TICKS: int = 1
SLIPPAGE_POINTS: float = SLIPPAGE_TICKS * CONTRACT_SPECS["tick_size"]  # 0.25
COMMISSION_RT: float = 2.50     # USD per contract round-turn

# ============================================================================
# POSITION SIZING
# ============================================================================
ACCOUNT_VALUE: float = 100_000.0
SIGMA_TARGET: float = 0.15      # 15% annualized target volatility
MAX_SIZE: int = 10              # Maximum contracts per trade

# ============================================================================
# HAR MODEL PARAMETERS
# ============================================================================
HAR_DAILY_WINDOW: int = 1
HAR_WEEKLY_WINDOW: int = 5
HAR_MONTHLY_WINDOW: int = 22

# ============================================================================
# HMM PARAMETERS
# ============================================================================
HMM_N_COMPONENTS: int = 3
HMM_COVARIANCE_TYPE: str = "full"
HMM_LOOKBACK: int = 20
HMM_RANDOM_STATE: int = 42
# Minor, spec-permitted deviation (Section 8.3): n_init reduced from 10 to 5 to
# keep walk-forward optimization tractable. Final/full fits can override.
HMM_N_INIT: int = 5
HMM_MAX_ITER: int = 200

# ============================================================================
# VOLATILITY REGIME
# ============================================================================
VOL_REGIME_P33: float = 0.33
VOL_REGIME_P67: float = 0.67

# ============================================================================
# OPTIMIZABLE PARAMETERS (exactly 3)
# ============================================================================
PARAM_DEFAULTS = {
    "toxic_continuation_threshold": 0.55,   # tau1
    "toxic_reversal_threshold": 0.55,        # tau2
    "max_hold_bars": 15,                     # N
    "regime_exit_enabled": True,
}

PARAM_SEARCH_SPACE = {
    "toxic_continuation_threshold": [round(x, 2) for x in np.arange(0.40, 0.81, 0.05)],
    "toxic_reversal_threshold": [round(x, 2) for x in np.arange(0.40, 0.81, 0.05)],
    "max_hold_bars": [5, 10, 15, 20, 25, 30],
}

# ============================================================================
# WALK-FORWARD PARAMETERS
# ============================================================================
WALK_FORWARD = {
    "train_window_bars": 39_000,
    "test_window_bars": 7_800,
    "step_size_bars": 3_900,
}

# ============================================================================
# SIGNAL ENCODING
# ============================================================================
SIGNAL_FLAT: int = 0
SIGNAL_LONG: int = 1
SIGNAL_SHORT: int = -1

# ============================================================================
# ACCEPTANCE CRITERIA
# ============================================================================
ACCEPTANCE_CRITERIA = {
    "min_sharpe": 1.0,
    "min_win_rate": 0.54,
    "min_trades_per_day": 5,
    "max_drawdown": 0.15,
    "min_profit_factor": 1.3,
}

# ============================================================================
# INSTRUMENT REGISTRY (multi-instrument ready)
# ============================================================================
# Each entry fully describes how to load + cost a given futures instrument.
# Add new futures here (same CSV schema as ES) to make them available.
INSTRUMENTS = {
    "ES": {
        "name": "E-mini S&P 500",
        "csv_path": CSV_FILE_PATH,
        "contract_specs": dict(CONTRACT_SPECS),
        "slippage_ticks": SLIPPAGE_TICKS,
        "commission_rt": COMMISSION_RT,
        "bars_per_day": BARS_PER_DAY,
        "train_end_index": TRAIN_END_INDEX,
    },
}


def get_instrument(symbol: str) -> dict:
    """Return the registry entry for a symbol (case-insensitive)."""
    key = symbol.upper()
    if key not in INSTRUMENTS:
        raise KeyError(f"Unknown instrument '{symbol}'. Known: {list(INSTRUMENTS)}")
    return INSTRUMENTS[key]
