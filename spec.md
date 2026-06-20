# CODING SPECIFICATION: Strategy 1 — Adaptive Flow-Toxicity with Regime-Aware Sizing

## Complete Implementation-Ready Specification for Agentic Coding

---

## SECTION 0: EXECUTIVE SUMMARY

### Strategy Identity
- **Strategy Name**: Adaptive Flow-Toxicity with Regime-Aware Sizing
- **Internal Codename**: FLOWTOX_REGIME_01
- **Source Problems**: 
  - 0021-pine/BVC-Cross-Asset-Toxicity (BVC-tick disagreement for directional flow toxicity)
  - 0027/HMM-Censored-Flow (Hidden Markov Model for regime persistence)
  - 0033-pine/Parkinson-HAR-Vol-Regime (Parkinson realized variance + HAR forecasting for volatility regime classification)
- **Cross-Pollination Mechanism**: Flow toxicity (BVC-tick disagreement) provides directional signals, but signal efficacy varies across volatility regimes. Fusing BVC toxicity with a 3-state HMM regime persistence model and Parkinson-HAR volatility-state classification creates an adaptive system that trades only when toxicity is regime-informative and sizes positions inversely to forecast volatility.

### Core Economic Logic
The strategy exploits the informational content of volume flow toxicity as measured by the disagreement between Bulk Volume Classification (BVC) signed volume and the tick-rule direction. When BVC and tick rule disagree (toxic flow), informed traders may be active. However, toxicity alone is insufficient — its predictive power depends on the market regime (normal, toxic-continuation, toxic-reversal). A 3-state Hidden Markov Model, whose observable features are rolling toxicity rates, Parkinson volatility estimates, and short-term price momentum, identifies the current regime. The Heterogeneous Autoregressive (HAR) model forecasts next-period volatility using daily, weekly, and monthly Parkinson realized variance components, enabling inverse-volatility position sizing.

### Key Parameters (Hard-Coded or Optimized)
| Parameter | Symbol | Role | Type | Value / Range |
|-----------|--------|------|------|---------------|
| HMM regime threshold long | τ1 | Min posterior for toxic-continuation long entry | Optimized | [0.40, 0.80] |
| HMM regime threshold short | τ2 | Min posterior for toxic-reversal short entry | Optimized | [0.40, 0.80] |
| Max hold duration | N | Maximum bars to hold a position | Optimized | [5, 30] |
| HAR daily window | w_d | Bars for daily RV (1 trading day) | Fixed | 78 |
| HAR weekly window | w_w | Trading days for weekly RV | Fixed | 5 |
| HAR monthly window | w_m | Trading days for monthly RV | Fixed | 22 |
| Volatility target | σ_target | Annualized target volatility for sizing | Fixed | 0.15 |
| Max position size | max_size | Maximum contracts per trade | Fixed | 10 |
| Vol regime high threshold | vol_high_pct | Percentile for "high vol" classification | Fixed | 67 |
| HMM lookback bars | hmm_lookback | Bars for HMM feature rolling | Fixed | 20 |
| Slippage | — | Per side, in ticks | Fixed | 1 tick = 0.25 index points |
| Commission RT | — | Round-turn per contract | Fixed | $2.50 |
| Tick value | — | Dollar value per tick per contract | Fixed | $12.50 |

### Expected Performance Targets
| Metric | Target | Minimum Acceptable |
|--------|--------|-------------------|
| Directional Accuracy | > 54% | > 52% |
| Trades per Day | > 5 | > 3 |
| Sharpe Ratio (after costs) | > 1.0 | > 0.5 |
| Max Drawdown | < 15% annual | < 20% annual |
| Profit Factor | > 1.3 | > 1.1 |

### Number of Optimized Parameters
**Exactly 3**: τ1 (long entry threshold), τ2 (short entry threshold), N (max hold bars). All other parameters are fixed by domain knowledge or data structure. This constraint is enforced to prevent overfitting per the anti-curve-fit guardrails in Section 5.

---

## SECTION 1: DATA SPECIFICATION & PIPELINE

### 1.1 Source Data File
```python
# EXACT file path — do not modify
CSV_FILE_PATH: str = "/mnt/agents/upload/ES_5min_RTH_6year.csv"
```

### 1.2 Exact CSV Column Specification
| Column Name | Data Type | Description | Example |
|-------------|-----------|-------------|---------|
| `ts_event` | datetime string (ISO 8601 with timezone) | Bar timestamp in ET | `2020-01-02 09:30:00-05:00` |
| `symbol` | string | Instrument symbol | `ES.v.0` |
| `open` | float | Opening price of 5-min bar | `3246.50` |
| `high` | float | Highest price of 5-min bar | `3249.50` |
| `low` | float | Lowest price of 5-min bar | `3245.00` |
| `close` | float | Closing price of 5-min bar | `3247.50` |
| `volume` | int | Total traded contracts in bar | `52003` |

### 1.3 Exact Dataset Statistics (Validated)
- **Total rows**: 122,296 (indices 0 to 122295)
- **Total columns**: 7
- **Date range**: 2020-01-02 09:30:00-05:00 to 2026-03-06 15:55:00-05:00
- **Frequency**: 5-minute bars
- **RTH session**: 09:30-16:00 ET = 78 bars per complete trading day
- **Trading days**: 1,594
- **Days with 78 bars**: 1,538 (96.5%)
- **Days with < 78 bars**: 56 (3.5%, mostly partial/holiday sessions)
- **Zero-volume bars**: 7 (all during March 2020 COVID circuit breakers)
- **NaN symbol rows**: 7 (identical to zero-volume bars, rows 3536, 3770, 3771, 3925, 3926, 4122, 4123)
- **OHLC anomalies**: 0 (all bars satisfy low <= open, close <= high)
- **Price increment (tick size)**: 0.25 index points (all price changes are multiples of 0.25)
- **Min close price**: 2,183.50 (2020-03-23)
- **Max close price**: 7,029.50 (2026-02-19)

### 1.4 Exact Data Loading Code
```python
import pandas as pd
import numpy as np
from datetime import datetime, time
import warnings

def load_data(csv_path: str = "/mnt/agents/upload/ES_5min_RTH_6year.csv") -> pd.DataFrame:
    """
    Load ES 5-minute RTH data from CSV.

    Returns DataFrame with columns:
        ts_event (pd.Timestamp with UTC), symbol (str), open (float64),
        high (float64), low (float64), close (float64), volume (int64)
    """
    # Load CSV with exact column names
    df = pd.read_csv(csv_path)

    # Validate expected columns exist
    expected_cols = ['ts_event', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    # Parse timestamps — use utc=True to handle timezone offsets uniformly
    df['ts_event'] = pd.to_datetime(df['ts_event'], utc=True)

    # Drop rows with NaN symbol (7 rows, all zero-volume during COVID halts)
    df = df.dropna(subset=['symbol']).reset_index(drop=True)

    # Validate no remaining NaNs in price columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if df[col].isnull().any():
            raise ValueError(f"Unexpected NaN values in column: {col}")

    # Validate OHLC consistency
    ohlc_invalid = (
        (df['low'] > df['high']) |
        (df['close'] > df['high']) |
        (df['close'] < df['low']) |
        (df['open'] > df['high']) |
        (df['open'] < df['low'])
    )
    if ohlc_invalid.any():
        bad_idx = ohlc_invalid[ohlc_invalid].index.tolist()
        raise ValueError(f"OHLC inconsistency at indices: {bad_idx[:10]}")

    # Ensure correct dtypes
    df['volume'] = df['volume'].astype(np.int64)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].astype(np.float64)

    # Sort by timestamp (defensive)
    df = df.sort_values('ts_event').reset_index(drop=True)

    return df
```

### 1.5 Exact RTH Session Filtering
The dataset is already RTH-filtered (09:30-16:00 ET). Verify this assumption programmatically. No additional RTH filtering is required. However, the code MUST verify that bars fall within expected RTH hours.

```python
def verify_rth_session(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verify all bars fall within RTH (09:30-16:00 ET).
    Input df must have ts_event as timezone-aware datetime.
    """
    # Convert to US/Eastern for time-of-day extraction
    ts_et = df['ts_event'].dt.tz_convert('America/New_York')

    # Extract time component
    bar_time = ts_et.dt.time

    # RTH bounds (inclusive)
    rth_start = time(9, 30)
    rth_end = time(16, 0)

    # All bars must be within RTH
    in_rth = (bar_time >= rth_start) & (bar_time <= rth_end)

    if not in_rth.all():
        bad_count = (~in_rth).sum()
        print(f"WARNING: {bad_count} bars outside RTH — removing them")
        df = df[in_rth].reset_index(drop=True)

    # Verify 5-minute frequency
    time_diffs = df['ts_event'].diff().dropna()
    expected_diff = pd.Timedelta(minutes=5)
    intraday_diffs = time_diffs[time_diffs < pd.Timedelta(hours=12)]  # Exclude overnight gaps

    if not (intraday_diffs == expected_diff).all():
        off_freq = intraday_diffs[intraday_diffs != expected_diff]
        print(f"WARNING: {len(off_freq)} bars not on 5-min frequency")

    return df
```

### 1.6 Exact Train/Test Split
```python
def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into development (train) and test sets.

    Development set: indices 0 to 85599 (85,600 bars, ~4.2 years, through ~April 2024)
    Test set:       indices 85600 to end (36,695 bars, ~1.8 years, from ~April 2024 onward)

    Returns: (train_df, test_df)
    """
    TRAIN_END_INDEX: int = 85599  # Inclusive end index for train

    train_df = df.iloc[:TRAIN_END_INDEX + 1].copy()
    test_df = df.iloc[TRAIN_END_INDEX + 1:].copy()

    # Validation assertions
    assert len(train_df) == 85600, f"Expected 85600 train rows, got {len(train_df)}"
    assert len(test_df) == 36695, f"Expected 36695 test rows, got {len(test_df)}"
    assert len(train_df) + len(test_df) == len(df), "Split rows don't sum to total"

    # Verify chronological order
    assert train_df['ts_event'].is_monotonic_increasing, "Train data not sorted"
    assert test_df['ts_event'].is_monotonic_increasing, "Test data not sorted"
    assert train_df['ts_event'].max() < test_df['ts_event'].min(), "Train/test overlap"

    print(f"Train: {len(train_df)} bars, {train_df['ts_event'].min()} to {train_df['ts_event'].max()}")
    print(f"Test:  {len(test_df)} bars, {test_df['ts_event'].min()} to {test_df['ts_event'].max()}")

    return train_df, test_df
```

### 1.7 Data Validation Checklist
After loading and splitting, the following assertions MUST pass:
1. Total rows after NaN removal = 122,289 (122,296 - 7 NaN symbol rows)
2. Train rows = 85,600; Test rows = 36,695
3. No NaN values in `open`, `high`, `low`, `close`, `volume`
4. All `volume` >= 0
5. All `high` >= `low`, `high` >= `close` >= `low`, `high` >= `open` >= `low`
6. Timestamps are monotonic increasing
7. All price changes are multiples of 0.25 (tick size validation on 1,000 random samples)
8. Intraday time differences are exactly 5 minutes
9. Train and test sets are non-overlapping

### 1.8 Contract Specifications (Hard-Coded Constants)
```python
# ES E-mini S&P 500 contract specs — NEVER change these values
CONTRACT_SPECS = {
    'tick_size': 0.25,           # Index points per tick
    'tick_value': 12.50,         # USD per tick per contract
    'point_value': 50.00,        # USD per index point per contract
    'multiplier': 50.00,         # Same as point_value
    'symbol': 'ES',
    'exchange': 'CME',
    'currency': 'USD',
}
```

---

## SECTION 2: FEATURE ENGINEERING (COMPLETE)

### Feature Computation Order
Features MUST be computed in this exact dependency order:
1. BVC_Signed_Volume (depends on: open, high, low, close, volume)
2. Tick_Rule_Direction (depends on: close)
3. Toxicity_Disagreement (depends on: BVC_Signed_Volume, Tick_Rule_Direction)
4. Parkinson_Bar_Variance (depends on: high, low)
5. Parkinson_Realized_Variance_1D (depends on: Parkinson_Bar_Variance)
6. Parkinson_Realized_Variance_1W (depends on: Parkinson_Realized_Variance_1D)
7. Parkinson_Realized_Variance_1M (depends on: Parkinson_Realized_Variance_1D)
8. HAR_Volatility_Forecast (depends on: RV_1D, RV_1W, RV_1M)
9. HMM_Observable_Features (depends on: Toxicity_Disagreement, Parkinson_Bar_Variance, close)
10. HMM_State_Posteriors (depends on: HMM_Observable_Features)
11. Volatility_Regime_Classification (depends on: HAR_Volatility_Forecast)
12. BVC_Direction (depends on: BVC_Signed_Volume)
13. Recent_Price_Momentum (depends on: close)
14. Signal_Generation_Features (depends on: all above)

---

### 2.1 Feature: BVC_Signed_Volume

**Feature name in code**: `bvc_signed_volume`

**Mathematical Formula**:
For each bar i:
```
V_BVC[i] = V[i] * ((close[i] - low[i]) - (high[i] - close[i])) / (high[i] - low[i])
         = V[i] * (2*close[i] - low[i] - high[i]) / (high[i] - low[i])
```

Where:
- V[i] = volume of bar i
- close[i], low[i], high[i] = OHLC prices of bar i
- Numerator = (close - low) represents buying pressure
- Denominator = (high - low) represents total range
- When close is at the high: V_BVC = +V (all volume is buying)
- When close is at the low: V_BVC = -V (all volume is selling)
- When close is at midpoint: V_BVC = 0 (balanced)

**Edge case — zero-range bars (high == low)**:
When `high[i] == low[i]`, the denominator is zero. In this case:
```python
if high[i] == low[i]:
    V_BVC[i] = 0.0
```
Zero-range bars occur when the market halts or there is no price movement. In the dataset, there are 7 zero-range bars (all during March 2020 circuit breakers, volume = 0).

**Exact computation pseudocode**:
```python
def compute_bvc_signed_volume(df: pd.DataFrame) -> pd.Series:
    """
    Compute BVC-signed volume for each bar.

    Parameters
    ----------
    df : pd.DataFrame with columns 'high', 'low', 'close', 'volume'

    Returns
    -------
    pd.Series : bvc_signed_volume, float64, same index as df
    """
    numerator = 2 * df['close'] - df['low'] - df['high']
    denominator = df['high'] - df['low']

    bvc = df['volume'] * numerator / denominator

    # Handle zero-range bars
    zero_range_mask = denominator == 0
    bvc = bvc.where(~zero_range_mask, 0.0)

    # Validate
    assert bvc.notna().all(), "NaN in bvc_signed_volume"
    assert (bvc.abs() <= df['volume'] * 1.0001).all(), "BVC magnitude exceeds volume"

    return bvc.astype(np.float64)
```

**Validation statistics (train set)**:
- Range: [-221,102.29, 187,006.95]
- Mean: 239.32
- Std: 12,871.94
- Theoretical bound: |V_BVC| <= V (verified on all bars)

---

### 2.2 Feature: Tick_Rule_Direction

**Feature name in code**: `tick_rule_direction`

**Mathematical Formula**:
```
D_tick[i] = sign(close[i] - close[i-1])
```
Where `sign(x)` returns:
- +1.0 if x > 0
- -1.0 if x < 0
- +1.0 if x == 0 (convention: unchanged price = up-tick)

**Edge case — first bar and unchanged prices**:
- For the first bar (i = 0): D_tick[0] = +1.0 (convention, no previous close)
- When close[i] == close[i-1]: D_tick[i] = +1.0 (up-tick convention)

**Exact computation pseudocode**:
```python
def compute_tick_rule_direction(df: pd.DataFrame) -> pd.Series:
    """
    Compute tick-rule direction for each bar.

    Parameters
    ----------
    df : pd.DataFrame with column 'close'

    Returns
    -------
    pd.Series : tick_rule_direction, float64 with values {-1.0, +1.0}
    """
    price_diff = df['close'].diff()
    tick_dir = np.sign(price_diff)

    # First bar: convention +1.0
    tick_dir.iloc[0] = 1.0

    # Unchanged prices: convention +1.0 (up-tick)
    tick_dir = tick_dir.where(tick_dir != 0, 1.0)

    assert tick_dir.isin([-1.0, 1.0]).all(), "Invalid tick_rule_direction values"

    return tick_dir.astype(np.float64)
```

---

### 2.3 Feature: Toxicity_Disagreement

**Feature name in code**: `toxicity_disagreement`

**Mathematical Formula**:
```
T_t[i] = sign(V_BVC[i]) * D_tick[i]
```

Interpretation:
- T_t[i] = +1.0: BVC and tick rule agree (non-toxic, benign flow)
- T_t[i] = -1.0: BVC and tick rule disagree (TOXIC flow — informed traders may be active)
- T_t[i] = 0.0: V_BVC[i] = 0 (perfectly balanced volume, classify as non-toxic)

**Why negative = toxic**: When BVC classifies volume as buying (positive V_BVC) but the tick rule says the price went down (D_tick = -1), this indicates large sell orders executed on upticks — a signature of informed selling. Conversely, negative V_BVC with positive D_tick indicates informed buying on downticks.

**Exact computation pseudocode**:
```python
def compute_toxicity_disagreement(bvc_signed_volume: pd.Series, 
                                   tick_rule_direction: pd.Series) -> pd.Series:
    """
    Compute toxicity disagreement: sign(BVC) * tick_direction.

    Returns +1.0 for agreement, -1.0 for toxic disagreement, 0.0 for zero BVC.
    """
    bvc_sign = np.sign(bvc_signed_volume)
    tox = bvc_sign * tick_rule_direction

    # Zero BVC volume -> 0.0 (neutral)
    tox = tox.where(bvc_sign != 0, 0.0)

    assert tox.isin([-1.0, 0.0, 1.0]).all(), "Invalid toxicity_disagreement values"

    return tox.astype(np.float64)
```

**Train set statistics**:
- Value +1.0: 70,748 bars (82.7%) — non-toxic agreement
- Value -1.0: 12,040 bars (14.1%) — TOXIC disagreement
- Value 0.0: 2,811 bars (3.3%) — zero BVC (neutral)
- Toxicity rate (fraction -1.0): 14.07%

---

### 2.4 Feature: Parkinson_Bar_Variance

**Feature name in code**: `parkinson_bar_variance`

**Mathematical Formula** (per bar):
```
σ²_bar[i] = (ln(high[i] / low[i]))^2 / (4 * ln(2))
          = (ln(high[i]) - ln(low[i]))^2 / (4 * ln(2))
```

The constant 4*ln(2) ≈ 2.7726 is the Parkinson (1980) scaling factor that makes this an unbiased estimator of the variance when the true process is Brownian motion.

**Edge case — zero-range bars**:
When high == low, ln(high/low) = ln(1) = 0, so σ²_bar = 0. No special handling needed.

**Exact computation pseudocode**:
```python
def compute_parkinson_bar_variance(df: pd.DataFrame) -> pd.Series:
    """
    Compute Parkinson variance contribution per bar.

    Returns unbiased variance estimate component for each bar.
    """
    log_ratio_sq = (np.log(df['high'] / df['low'])) ** 2
    parkinson_bar = log_ratio_sq / (4.0 * np.log(2.0))

    # Handle zero-range bars (already produce 0, but be explicit)
    zero_range = df['high'] == df['low']
    parkinson_bar = parkinson_bar.where(~zero_range, 0.0)

    assert parkinson_bar.notna().all(), "NaN in parkinson_bar_variance"
    assert (parkinson_bar >= 0).all(), "Negative Parkinson variance"

    return parkinson_bar.astype(np.float64)
```

---

### 2.5 Features: Parkinson_Realized_Variance (1-day, 1-week, 1-month)

These features aggregate the per-bar Parkinson variance to daily, weekly, and monthly horizons. The daily window is exactly 78 bars (one RTH session). The weekly is a 5-day rolling average of daily RV. The monthly is a 22-day rolling average of daily RV.

**2.5.1 Daily Realized Variance (RV_1D)**
**Feature name in code**: `rv_1d`

**Mathematical Formula**:
```
RV_1D[t] = Σ_{i in day t} σ²_bar[i]
```
This is the simple sum of per-bar Parkinson variances within each trading day. No division by N — the Parkinson scaling factor in σ²_bar already produces the correct variance units.

**2.5.2 Weekly Realized Variance (RV_1W)**
**Feature name in code**: `rv_1w`

**Mathematical Formula**:
```
RV_1W[t] = (1/5) * Σ_{k=0}^{4} RV_1D[t-k]
```
Simple moving average of the last 5 daily RV values, including the current day.

**2.5.3 Monthly Realized Variance (RV_1M)**
**Feature name in code**: `rv_1m`

**Mathematical Formula**:
```
RV_1M[t] = (1/22) * Σ_{k=0}^{21} RV_1D[t-k]
```
Simple moving average of the last 22 daily RV values, including the current day.

**Important**: RV_1W and RV_1M are computed as DAILY-level rolling means, then broadcast back to each bar within the day. All 78 bars of a given trading day share the same RV_1W and RV_1M values.

**Exact computation pseudocode**:
```python
BARS_PER_DAY: int = 78  # Fixed: 09:30-16:00 ET at 5-min frequency
HAR_DAILY_WINDOW: int = 1    # 1 day
HAR_WEEKLY_WINDOW: int = 5   # 5 trading days
HAR_MONTHLY_WINDOW: int = 22 # ~1 month of trading days

def compute_parkinson_realized_variances(df: pd.DataFrame, 
                                         parkinson_bar: pd.Series) -> pd.DataFrame:
    """
    Compute daily, weekly, and monthly Parkinson realized variances.

    Parameters
    ----------
    df : pd.DataFrame with 'ts_event' column
    parkinson_bar : pd.Series of per-bar Parkinson variances

    Returns
    -------
    pd.DataFrame with columns ['rv_1d', 'rv_1w', 'rv_1m'], same index as df
    """
    # Assign day_id based on bar index (every 78 bars = 1 day)
    # Use integer division: bars 0-77 = day 0, bars 78-155 = day 1, etc.
    day_id = df.index // BARS_PER_DAY

    # Daily RV: sum of parkinson_bar within each day
    daily_rv = parkinson_bar.groupby(day_id).sum()

    # Weekly RV: 5-day rolling mean of daily RV
    weekly_rv = daily_rv.rolling(window=HAR_WEEKLY_WINDOW, min_periods=1).mean()

    # Monthly RV: 22-day rolling mean of daily RV
    monthly_rv = daily_rv.rolling(window=HAR_MONTHLY_WINDOW, min_periods=1).mean()

    # Map daily values back to each bar
    rv_1d = day_id.map(daily_rv)
    rv_1w = day_id.map(weekly_rv)
    rv_1m = day_id.map(monthly_rv)

    result = pd.DataFrame({
        'rv_1d': rv_1d,
        'rv_1w': rv_1w,
        'rv_1m': rv_1m
    }, index=df.index)

    # Fill any remaining NaN with forward fill, then backward fill
    result = result.ffill().bfill().fillna(result['rv_1d'].mean())

    # Validate all non-negative
    for col in ['rv_1d', 'rv_1w', 'rv_1m']:
        assert (result[col] >= 0).all(), f"Negative values in {col}"

    return result
```

**Train set statistics**:
- Daily RV mean: 0.00010517 (annualized vol ≈ 16.3%)
- Daily RV std: 0.00024771
- Daily RV range: [0.00000527, 0.00302362]

---

### 2.6 Feature: HAR_Volatility_Forecast

**Feature name in code**: `har_vol_forecast`

**Mathematical Formula** (Corsi, 2009 HAR-RV model):
```
σ²_forecast[t+1] = c + β_d * RV_1D[t] + β_w * RV_1W[t] + β_m * RV_1M[t]
```

Where:
- c = intercept (fitted on training data)
- β_d = coefficient for daily component (fitted)
- β_w = coefficient for weekly component (fitted)
- β_m = coefficient for monthly component (fitted)
- RV_1D[t], RV_1W[t], RV_1M[t] = current day's realized variance components

**Fitting procedure** (performed ONCE on training data only):
```python
def fit_har_model(rv_1d: pd.Series, rv_1w: pd.Series, rv_1m: pd.Series) -> dict:
    """
    Fit HAR-RV model via OLS on training data.
    Target: RV_1D[t+1] (next day's realized variance)
    Features: [1, RV_1D[t], RV_1W[t], RV_1M[t]]

    Returns dict with keys: 'intercept', 'beta_d', 'beta_w', 'beta_m'
    """
    # Prepare data — use daily-level series (1 value per day)
    daily_data = pd.DataFrame({
        'rv_1d': rv_1d.iloc[::BARS_PER_DAY].reset_index(drop=True),
        'rv_1w': rv_1w.iloc[::BARS_PER_DAY].reset_index(drop=True),
        'rv_1m': rv_1m.iloc[::BARS_PER_DAY].reset_index(drop=True),
    }).dropna()

    # Target: next day's RV
    y = daily_data['rv_1d'].shift(-1).dropna()
    X = daily_data[['rv_1d', 'rv_1w', 'rv_1m']].loc[y.index]

    # Add intercept
    X_with_const = np.column_stack([np.ones(len(X)), X['rv_1d'], X['rv_1w'], X['rv_1m']])

    # OLS fit
    beta, residuals, rank, s = np.linalg.lstsq(X_with_const, y.values, rcond=None)

    params = {
        'intercept': float(beta[0]),
        'beta_d': float(beta[1]),
        'beta_w': float(beta[2]),
        'beta_m': float(beta[3]),
    }

    # Validate: ensure forecast is non-negative
    # If any coefficient is negative, clamp to 0 and refit
    if any(v < 0 for v in params.values()):
        print(f"WARNING: Negative HAR coefficients detected: {params}")
        print("Applying non-negativity constraint...")
        # Simple approach: set negative coeffs to small positive value
        for k in params:
            params[k] = max(params[k], 0.001)

    return params
```

**Expected fitted values (from train set analysis)**:
- intercept (c): ~0.000013
- beta_d: ~0.60
- beta_w: ~0.38
- beta_m: ~-0.11 (if negative, clamp to 0.001)

**Forecast computation**:
```python
def compute_har_vol_forecast(rv_1d: pd.Series, rv_1w: pd.Series, rv_1m: pd.Series,
                              har_params: dict) -> pd.Series:
    """
    Compute HAR volatility forecast for next period.
    Forecast is applied to ALL bars within a day (broadcast from daily to intraday).
    """
    forecast = (
        har_params['intercept'] +
        har_params['beta_d'] * rv_1d +
        har_params['beta_w'] * rv_1w +
        har_params['beta_m'] * rv_1m
    )

    # Ensure non-negative
    forecast = forecast.clip(lower=1e-12)

    return forecast.astype(np.float64)
```

---

### 2.7 Feature: Volatility_Regime_Classification

**Feature name in code**: `vol_regime`

**Mathematical Formula**:
Classify each bar into one of 3 volatility regimes based on the HAR forecast:
```
vol_regime[i] = {
    'low'    if σ²_forecast[i] <= percentile_33(σ²_forecast[train]),
    'medium' if percentile_33 < σ²_forecast[i] <= percentile_67(σ²_forecast[train]),
    'high'   if σ²_forecast[i] > percentile_67(σ²_forecast[train])
}
```

The 33rd and 67th percentiles are computed ONCE on the training data and stored as hard thresholds. They do NOT recomputed on test data.

**Exact computation pseudocode**:
```python
def compute_vol_regime(har_vol_forecast: pd.Series, 
                       train_forecast: pd.Series = None) -> pd.Series:
    """
    Classify volatility into 3 regimes based on HAR forecast.

    Parameters
    ----------
    har_vol_forecast : pd.Series — HAR forecast for current data (train or test)
    train_forecast : pd.Series — HAR forecast from training data (for threshold computation)
                              If None, uses har_vol_forecast to compute thresholds

    Returns
    -------
    pd.Series with values {0, 1, 2} = {low, medium, high}
    """
    if train_forecast is not None:
        # Use pre-computed thresholds from training
        p33 = train_forecast.quantile(0.33)
        p67 = train_forecast.quantile(0.67)
    else:
        # Compute thresholds from current data (training phase)
        p33 = har_vol_forecast.quantile(0.33)
        p67 = har_vol_forecast.quantile(0.67)

    regime = pd.Series(1, index=har_vol_forecast.index, dtype=np.int64)  # default medium
    regime[har_vol_forecast <= p33] = 0  # low
    regime[har_vol_forecast > p67] = 2   # high

    return regime

# Store thresholds after training
VOL_REGIME_THRESHOLDS = {
    'p33': None,  # Filled during training
    'p67': None,  # Filled during training
}
```

**Train set thresholds (expected)**:
- 33rd percentile: ~0.000031
- 67th percentile: ~0.000077

---

### 2.8 Features: HMM_Observable_Features

The HMM requires observable features that capture the information needed to distinguish the three hidden states: {normal, toxic-continuation, toxic-reversal}. These features are computed as rolling aggregates over recent bars.

**Feature names in code**: `hmm_feat_1`, `hmm_feat_2`, `hmm_feat_3`, `hmm_feat_4`, `hmm_feat_5`

**Feature definitions**:

| Feature | Name | Formula | Window | Rationale |
|---------|------|---------|--------|-----------|
| hmm_feat_1 | Rolling toxicity rate | mean(toxicity_disagreement < 0) | 20 bars | Frequency of toxic flow |
| hmm_feat_2 | Rolling BVC sign | mean(sign(BVC)) | 20 bars | Net buying/selling pressure |
| hmm_feat_3 | Rolling Parkinson var | mean(parkinson_bar) | 20 bars | Recent volatility level |
| hmm_feat_4 | Price momentum | close - close[5 bars ago] | 5 bars | Short-term directional bias |
| hmm_feat_5 | Volume intensity | volume / rolling_mean(volume, 20) | 1 bar | Relative volume level |

**Exact computation pseudocode**:
```python
HMM_LOOKBACK: int = 20  # bars for rolling features

def compute_hmm_observable_features(df: pd.DataFrame,
                                     toxicity: pd.Series,
                                     bvc: pd.Series,
                                     parkinson_bar: pd.Series) -> pd.DataFrame:
    """
    Compute 5 observable features for HMM state inference.
    All features are standardized (z-score) before HMM fitting.

    Returns DataFrame with columns [hmm_feat_1, ..., hmm_feat_5]
    """
    features = pd.DataFrame(index=df.index)

    # Feature 1: Rolling toxicity rate (fraction of toxic bars)
    is_toxic = (toxicity < 0).astype(float)
    features['hmm_feat_1'] = is_toxic.rolling(window=HMM_LOOKBACK, min_periods=1).mean()

    # Feature 2: Rolling BVC sign (average signed BVC direction)
    bvc_sign = np.sign(bvc)
    features['hmm_feat_2'] = bvc_sign.rolling(window=HMM_LOOKBACK, min_periods=1).mean()

    # Feature 3: Rolling Parkinson variance (log scale for stability)
    features['hmm_feat_3'] = np.log1p(parkinson_bar.rolling(window=HMM_LOOKBACK, min_periods=1).mean())

    # Feature 4: 5-bar price momentum (in index points)
    features['hmm_feat_4'] = df['close'].diff(5)

    # Feature 5: Volume intensity (relative to 20-bar mean)
    vol_mean_20 = df['volume'].rolling(window=HMM_LOOKBACK, min_periods=1).mean()
    features['hmm_feat_5'] = df['volume'] / vol_mean_20.clip(lower=1)

    # Fill NaN with 0 (only affects first few bars)
    features = features.fillna(0.0)

    return features.astype(np.float64)
```

---

### 2.9 Feature: HMM_State_Posteriors

**Feature names in code**: `hmm_state_0_posterior`, `hmm_state_1_posterior`, `hmm_state_2_posterior`

**Model**: 3-state Gaussian Mixture Model (GMM) used as proxy for HMM emission distributions. The GMM is fit on standardized observable features from the training data. For each bar, the model outputs the posterior probability of belonging to each of the 3 states.

**State semantics** (identified from data):
- **State 0 — Normal**: Low toxicity rate, moderate volatility, positive momentum. Most common state (~55% of bars). Flow is benign, toxicity signals are uninformative.
- **State 1 — Toxic-Continuation**: Elevated toxicity, negative momentum, moderate volatility. Toxic flow predicts continuation of current trend (~38% of bars).
- **State 2 — Toxic-Reversal**: High toxicity, high volatility, negative momentum with subsequent reversal. Toxic flow predicts reversal (~6-7% of bars).

**Exact computation pseudocode**:
```python
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

def fit_hmm_model(obs_features_train: pd.DataFrame, random_state: int = 42) -> tuple:
    """
    Fit 3-state Gaussian Mixture Model on training observable features.

    Returns:
        hmm_model: fitted GaussianMixture
        scaler: fitted StandardScaler
    """
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(obs_features_train.values)

    # Fit 3-component GMM
    gmm = GaussianMixture(
        n_components=3,
        covariance_type='full',
        random_state=random_state,
        n_init=10,
        max_iter=200,
        tol=1e-4,
    )
    gmm.fit(X_train)

    # Validate: all states should have non-zero weight
    print(f"GMM weights: {gmm.weights_}")
    assert all(w > 0.05 for w in gmm.weights_), "Degenerate state with <5% weight"

    return gmm, scaler

def compute_hmm_posteriors(obs_features: pd.DataFrame,
                            hmm_model: GaussianMixture,
                            scaler: StandardScaler) -> pd.DataFrame:
    """
    Compute HMM state posterior probabilities for each bar.

    Returns DataFrame with columns [hmm_state_0_posterior, hmm_state_1_posterior, hmm_state_2_posterior]
    Each row sums to 1.0.
    """
    X = scaler.transform(obs_features.values)
    posteriors = hmm_model.predict_proba(X)

    result = pd.DataFrame(
        posteriors,
        columns=['hmm_state_0_posterior', 'hmm_state_1_posterior', 'hmm_state_2_posterior'],
        index=obs_features.index
    )

    # Validate: each row sums to 1
    assert np.allclose(result.sum(axis=1), 1.0), "Posteriors don't sum to 1"
    assert (result >= 0).all().all(), "Negative posterior probability"
    assert (result <= 1).all().all(), "Posterior probability > 1"

    return result.astype(np.float64)
```

**State identification procedure** (after fitting):
After fitting the GMM, identify which state corresponds to which semantic label by analyzing the mean observable features and forward returns in each state on the training data:
1. Compute the mean of each observable feature for each state's posterior-weighted bars
2. The state with the LOWEST toxicity rate and LOWEST volatility = State 0 (Normal)
3. The state with ELEVATED toxicity and NEGATIVE momentum = State 1 (Toxic-Continuation)
4. The state with HIGH volatility and HIGHEST future returns (reversal pattern) = State 2 (Toxic-Reversal)

Store the state-to-label mapping and apply it consistently to both train and test.

---

### 2.10 Feature: BVC_Direction

**Feature name in code**: `bvc_direction`

**Mathematical Formula**:
```
bvc_direction[i] = sign(V_BVC[i]) ∈ {-1.0, 0.0, +1.0}
```

This is simply the sign of the BVC-signed volume, used directly in signal generation.

**Exact computation**:
```python
bvc_direction = np.sign(bvc_signed_volume)
```

---

### 2.11 Feature: Recent_Price_Momentum

**Feature name in code**: `recent_momentum`

**Mathematical Formula**:
```
recent_momentum[i] = close[i] - close[i-5]
```
5-bar price change in index points. Positive = upward momentum, negative = downward momentum.

---

### 2.12 Feature: Signal_Generation_Features

These are the final composite features used directly in signal generation:

**Feature name in code**: `toxic_cont_signal`, `toxic_rev_signal`

**toxic_cont_signal**: Composite signal for toxic-continuation regime
```
toxic_cont_signal[i] = hmm_state_1_posterior[i] * bvc_direction[i] * I(vol_regime[i] != 2)
```
where I(condition) is the indicator function (1 if true, 0 if false).

**toxic_rev_signal**: Composite signal for toxic-reversal regime
```
toxic_rev_signal[i] = hmm_state_2_posterior[i] * (-bvc_direction[i]) * I(vol_regime[i] != 2)
```

Both signals are set to zero when the volatility regime is "high" (regime = 2), as the strategy avoids trading in high-volatility periods.

---

### 2.13 Feature Engineering Master Function

```python
def engineer_features(df: pd.DataFrame, har_params: dict = None,
                      hmm_model: GaussianMixture = None,
                      scaler: StandardScaler = None,
                      vol_thresholds: dict = None,
                      fit_models: bool = True) -> tuple:
    """
    Master feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame — input price data
    har_params : dict — pre-fitted HAR parameters (None if fit_models=True)
    hmm_model : GaussianMixture — pre-fitted HMM (None if fit_models=True)
    scaler : StandardScaler — pre-fitted scaler (None if fit_models=True)
    vol_thresholds : dict — {'p33', 'p67'} thresholds (None if fit_models=True)
    fit_models : bool — if True, fit all models on this data; if False, use pre-fitted

    Returns
    -------
    (features_df, fitted_params) where fitted_params contains all fitted objects
    """
    result = df.copy()
    fitted = {}

    # 2.1 BVC Signed Volume
    result['bvc_signed_volume'] = compute_bvc_signed_volume(result)

    # 2.2 Tick Rule Direction
    result['tick_rule_direction'] = compute_tick_rule_direction(result)

    # 2.3 Toxicity Disagreement
    result['toxicity_disagreement'] = compute_toxicity_disagreement(
        result['bvc_signed_volume'], result['tick_rule_direction']
    )

    # 2.4 Parkinson Bar Variance
    result['parkinson_bar_variance'] = compute_parkinson_bar_variance(result)

    # 2.5 Realized Variances
    rv_df = compute_parkinson_realized_variances(result, result['parkinson_bar_variance'])
    result = pd.concat([result, rv_df], axis=1)

    # 2.6 HAR Forecast
    if fit_models:
        har_params = fit_har_model(result['rv_1d'], result['rv_1w'], result['rv_1m'])
        fitted['har_params'] = har_params
    result['har_vol_forecast'] = compute_har_vol_forecast(
        result['rv_1d'], result['rv_1w'], result['rv_1m'], har_params
    )

    # 2.7 Vol Regime
    if fit_models:
        vol_thresholds = {
            'p33': result['har_vol_forecast'].quantile(0.33),
            'p67': result['har_vol_forecast'].quantile(0.67),
        }
        fitted['vol_thresholds'] = vol_thresholds
    result['vol_regime'] = compute_vol_regime(
        result['har_vol_forecast'], 
        train_forecast=result['har_vol_forecast'] if fit_models else None
    )

    # 2.8 HMM Observable Features
    hmm_features = compute_hmm_observable_features(
        result, result['toxicity_disagreement'], 
        result['bvc_signed_volume'], result['parkinson_bar_variance']
    )

    # 2.9 HMM State Posteriors
    if fit_models:
        hmm_model, scaler = fit_hmm_model(hmm_features)
        fitted['hmm_model'] = hmm_model
        fitted['scaler'] = scaler

    posteriors = compute_hmm_posteriors(hmm_features, hmm_model, scaler)
    result = pd.concat([result, posteriors], axis=1)

    # 2.10 BVC Direction
    result['bvc_direction'] = np.sign(result['bvc_signed_volume'])

    # 2.11 Recent Momentum
    result['recent_momentum'] = result['close'].diff(5)

    # 2.12 Signal Generation Features
    vol_not_high = (result['vol_regime'] != 2).astype(float)
    result['toxic_cont_signal'] = (
        result['hmm_state_1_posterior'] * result['bvc_direction'] * vol_not_high
    )
    result['toxic_rev_signal'] = (
        result['hmm_state_2_posterior'] * (-result['bvc_direction']) * vol_not_high
    )

    return result, fitted
```

---

## SECTION 3: SIGNAL GENERATION ENGINE (COMPLETE)

### 3.1 Trading Signal Philosophy
The strategy generates signals at the **close of each 5-minute bar** for potential entry at the **next bar's open**. This is an end-of-bar signal mechanism — no intrabar execution. All computations use data available at or before the current bar's close. No lookahead is permitted.

### 3.2 Exact Signal Encoding
```python
# Signal values
SIGNAL_FLAT: int = 0   # No position
SIGNAL_LONG: int = 1   # Long position
SIGNAL_SHORT: int = -1 # Short position
```

### 3.3 Exact Entry Conditions

#### Long Entry (Toxic-Continuation Long)
A long signal is generated at bar i if ALL of the following conditions are simultaneously true:

```
Condition L1: hmm_state_1_posterior[i] > τ1
Condition L2: bvc_direction[i] > 0         (BVC positive = buying pressure)
Condition L3: vol_regime[i] != 2           (not high volatility)
Condition L4: position[i-1] == 0           (currently flat — no pyramiding)
```

Where:
- τ1 = toxic_continuation_threshold (optimized, range [0.40, 0.80], default 0.55)
- hmm_state_1_posterior[i] = posterior probability of toxic-continuation state
- bvc_direction[i] = sign of BVC-signed volume
- vol_regime[i] ∈ {0, 1, 2} = {low, medium, high}

**Logic**: When the HMM identifies high probability of toxic-continuation regime AND BVC shows net buying pressure AND volatility is not extreme, go long. The toxicity disagreement suggests informed buyers are active and the trend will continue.

#### Short Entry (Toxic-Reversal Short)
A short signal is generated at bar i if ALL of the following conditions are simultaneously true:

```
Condition S1: hmm_state_2_posterior[i] > τ2
Condition S2: bvc_direction[i] < 0         (BVC negative = selling pressure)
Condition S3: vol_regime[i] != 2           (not high volatility)
Condition S4: position[i-1] == 0           (currently flat — no pyramiding)
```

Where:
- τ2 = toxic_reversal_threshold (optimized, range [0.40, 0.80], default 0.55)
- hmm_state_2_posterior[i] = posterior probability of toxic-reversal state

**Logic**: When the HMM identifies high probability of toxic-reversal regime AND BVC shows net selling pressure AND volatility is not extreme, go short. The toxicity disagreement suggests informed sellers are active but a reversal is imminent.

### 3.4 Exact Exit Conditions

#### Time-Based Exit
A position is closed after exactly N bars from entry, where:
- N = max_hold_bars (optimized, range [5, 30], default 15)
- Counting: entry bar = bar 0, exit at the close of bar N
- If N bars have elapsed since entry, generate SIGNAL_FLAT at bar i regardless of other conditions

#### Regime-Based Exit (Optional Enhancement)
If the posterior probability of the CURRENTLY ACTIVE regime drops below half of its entry threshold, exit early:
- For long positions: if hmm_state_1_posterior[i] < τ1 / 2 → exit
- For short positions: if hmm_state_2_posterior[i] < τ2 / 2 → exit

### 3.5 Exact Hold Duration Rules
```python
# Position tracking
entry_bar: int = -1       # Index of bar where position was entered
position: int = 0         # Current position: -1, 0, +1
hold_counter: int = 0     # Bars held so far

# At each bar i:
if position != 0:
    hold_counter += 1
    if hold_counter >= N:
        signal[i] = SIGNAL_FLAT   # Time-based exit
        position = 0
        hold_counter = 0
    elif regime_exit_enabled and early_exit_condition_met:
        signal[i] = SIGNAL_FLAT   # Regime-based early exit
        position = 0
        hold_counter = 0
else:
    # Check entry conditions
    if long_conditions_met:
        signal[i] = SIGNAL_LONG
        position = 1
        hold_counter = 0
        entry_bar = i
    elif short_conditions_met:
        signal[i] = SIGNAL_SHORT
        position = -1
        hold_counter = 0
        entry_bar = i
    else:
        signal[i] = SIGNAL_FLAT
```

### 3.6 Exact Position Sizing Formula

Position size is computed at entry time using inverse-volatility targeting:

```python
# At entry bar i:
forecast_vol_daily = sqrt(har_vol_forecast[i])  # Daily vol forecast (std dev)

# Annualized volatility forecast
forecast_vol_annual = forecast_vol_daily * sqrt(252)

# Target volatility as fraction of account
# σ_target = 0.15 (15% annualized target vol)
# Account value assumed = $100,000 for sizing reference
ACCOUNT_VALUE: float = 100_000.0  # USD reference
σ_TARGET: float = 0.15  # Annualized target volatility

# Dollar risk per contract per day
# ES point value = $50, daily vol in points = forecast_vol_daily * close_price
# (close price scales the log-vol to absolute dollar vol)
contract_vol_daily_usd = forecast_vol_daily * result['close'].iloc[i] * CONTRACT_SPECS['point_value']

# Number of contracts for target vol
# size = (account * target_vol%) / contract_vol_daily_usd
size_raw = (ACCOUNT_VALUE * σ_TARGET) / contract_vol_daily_usd

# Clamp to [1, max_size]
MAX_SIZE: int = 10  # Maximum contracts
size = int(np.clip(round(size_raw), 1, MAX_SIZE))
```

**Simplified formula**:
```
size[i] = clamp( round( (100,000 * 0.15) / (sqrt(har_vol_forecast[i]) * close[i] * 50) ), 1, 10 )
```

### 3.7 Complete Signal Generation Function

```python
# Optimizable parameters with their ranges and defaults
PARAM_DEFAULTS = {
    'toxic_continuation_threshold': 0.55,  # τ1, range [0.40, 0.80]
    'toxic_reversal_threshold': 0.55,       # τ2, range [0.40, 0.80]
    'max_hold_bars': 15,                    # N, range [5, 30]
    'regime_exit_enabled': True,            # Enable early regime-based exit
}

def generate_signals(features_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Generate trading signals from engineered features.

    Parameters
    ----------
    features_df : pd.DataFrame — output from engineer_features(), must contain:
        hmm_state_1_posterior, hmm_state_2_posterior, bvc_direction, 
        vol_regime, har_vol_forecast, close
    params : dict — strategy parameters with keys:
        toxic_continuation_threshold (τ1), toxic_reversal_threshold (τ2),
        max_hold_bars (N), regime_exit_enabled

    Returns
    -------
    pd.DataFrame — same as features_df with added columns:
        signal (int: -1, 0, +1), position_size (int), entry_price (float),
        exit_price (float), trade_pnl (float), hold_bars (int)
    """
    df = features_df.copy()
    n = len(df)

    # Extract parameters
    τ1 = params['toxic_continuation_threshold']
    τ2 = params['toxic_reversal_threshold']
    N = params['max_hold_bars']
    regime_exit = params.get('regime_exit_enabled', True)

    # Initialize output columns
    df['signal'] = SIGNAL_FLAT
    df['position_size'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['trade_pnl'] = np.nan
    df['hold_bars'] = 0
    df['bars_in_trade'] = 0  # Counter for current trade

    # State variables
    position = SIGNAL_FLAT
    hold_counter = 0
    entry_bar = -1
    current_size = 0
    entry_price = 0.0

    # Process bar by bar (starting from bar 20 to ensure features are valid)
    start_bar = max(20, HMM_LOOKBACK)
    for i in range(start_bar, n - 1):  # n-1 because we need next bar's open for entry
        # Update hold counter if in position
        if position != SIGNAL_FLAT:
            hold_counter += 1
            df.loc[df.index[i], 'bars_in_trade'] = hold_counter

            # Check time-based exit
            if hold_counter >= N:
                df.loc[df.index[i], 'signal'] = SIGNAL_FLAT
                df.loc[df.index[i], 'position_size'] = current_size
                df.loc[df.index[i], 'exit_price'] = df['close'].iloc[i]
                # Record trade P&L
                if position == SIGNAL_LONG:
                    pnl = (df['close'].iloc[i] - entry_price) * current_size * CONTRACT_SPECS['point_value']
                else:
                    pnl = (entry_price - df['close'].iloc[i]) * current_size * CONTRACT_SPECS['point_value']
                df.loc[df.index[i], 'trade_pnl'] = pnl
                df.loc[df.index[i], 'hold_bars'] = hold_counter
                # Reset
                position = SIGNAL_FLAT
                hold_counter = 0
                current_size = 0
                continue

            # Check regime-based early exit
            if regime_exit:
                if position == SIGNAL_LONG and df['hmm_state_1_posterior'].iloc[i] < τ1 / 2:
                    df.loc[df.index[i], 'signal'] = SIGNAL_FLAT
                    df.loc[df.index[i], 'position_size'] = current_size
                    df.loc[df.index[i], 'exit_price'] = df['close'].iloc[i]
                    pnl = (df['close'].iloc[i] - entry_price) * current_size * CONTRACT_SPECS['point_value']
                    df.loc[df.index[i], 'trade_pnl'] = pnl
                    df.loc[df.index[i], 'hold_bars'] = hold_counter
                    position = SIGNAL_FLAT
                    hold_counter = 0
                    current_size = 0
                    continue
                elif position == SIGNAL_SHORT and df['hmm_state_2_posterior'].iloc[i] < τ2 / 2:
                    df.loc[df.index[i], 'signal'] = SIGNAL_FLAT
                    df.loc[df.index[i], 'position_size'] = current_size
                    df.loc[df.index[i], 'exit_price'] = df['close'].iloc[i]
                    pnl = (entry_price - df['close'].iloc[i]) * current_size * CONTRACT_SPECS['point_value']
                    df.loc[df.index[i], 'trade_pnl'] = pnl
                    df.loc[df.index[i], 'hold_bars'] = hold_counter
                    position = SIGNAL_FLAT
                    hold_counter = 0
                    current_size = 0
                    continue

            # Hold position — no signal change
            df.loc[df.index[i], 'signal'] = position
            df.loc[df.index[i], 'position_size'] = current_size
            continue

        # Not in position — check entry conditions
        # Condition L1-L4: Long entry
        if (df['hmm_state_1_posterior'].iloc[i] > τ1 and
            df['bvc_direction'].iloc[i] > 0 and
            df['vol_regime'].iloc[i] != 2 and
            position == SIGNAL_FLAT):

            position = SIGNAL_LONG
            hold_counter = 0
            entry_bar = i
            entry_price = df['close'].iloc[i]

            # Compute position size
            forecast_vol = np.sqrt(max(df['har_vol_forecast'].iloc[i], 1e-12))
            contract_vol_usd = forecast_vol * df['close'].iloc[i] * CONTRACT_SPECS['point_value']
            size_raw = (ACCOUNT_VALUE * σ_TARGET) / contract_vol_usd
            current_size = int(np.clip(round(size_raw), 1, MAX_SIZE))

            df.loc[df.index[i], 'signal'] = SIGNAL_LONG
            df.loc[df.index[i], 'position_size'] = current_size
            df.loc[df.index[i], 'entry_price'] = entry_price
            continue

        # Condition S1-S4: Short entry
        if (df['hmm_state_2_posterior'].iloc[i] > τ2 and
            df['bvc_direction'].iloc[i] < 0 and
            df['vol_regime'].iloc[i] != 2 and
            position == SIGNAL_FLAT):

            position = SIGNAL_SHORT
            hold_counter = 0
            entry_bar = i
            entry_price = df['close'].iloc[i]

            # Compute position size
            forecast_vol = np.sqrt(max(df['har_vol_forecast'].iloc[i], 1e-12))
            contract_vol_usd = forecast_vol * df['close'].iloc[i] * CONTRACT_SPECS['point_value']
            size_raw = (ACCOUNT_VALUE * σ_TARGET) / contract_vol_usd
            current_size = int(np.clip(round(size_raw), 1, MAX_SIZE))

            df.loc[df.index[i], 'signal'] = SIGNAL_SHORT
            df.loc[df.index[i], 'position_size'] = current_size
            df.loc[df.index[i], 'entry_price'] = entry_price
            continue

        # Remain flat
        df.loc[df.index[i], 'signal'] = SIGNAL_FLAT
        df.loc[df.index[i], 'position_size'] = 0

    return df
```

### 3.8 Signal Statistics to Monitor
After signal generation, compute and report:
- Total signal count (long + short)
- Long signal count and percentage
- Short signal count and percentage
- Flat bar count and percentage
- Average position size (contracts)
- Signal frequency: signals per day = total_signals / num_trading_days
- Distribution of hold durations (for completed trades)
- Distribution of position sizes

---

## SECTION 4: BACKTESTING ENGINE (COMPLETE)

### 4.1 Backtesting Philosophy
The backtest engine executes trades at the **close of the signal bar** and holds until the **close of the exit bar**. This is a close-to-close backtest on 5-minute bars. No intrabar execution is modeled. The engine accounts for slippage (1 tick per side) and round-turn commission ($2.50 per contract).

### 4.2 Exact Slippage Model
```python
# Slippage: 1 tick per side (entry and exit)
SLIPPAGE_TICKS: int = 1  # ticks
SLIPPAGE_POINTS: float = SLIPPAGE_TICKS * CONTRACT_SPECS['tick_size']  # 0.25 index points
```

For each trade:
- **Long entry**: fill price = close[i] + SLIPPAGE_POINTS (worse by 1 tick)
- **Long exit**: fill price = close[exit] - SLIPPAGE_POINTS (worse by 1 tick)
- **Short entry**: fill price = close[i] - SLIPPAGE_POINTS (worse by 1 tick)
- **Short exit**: fill price = close[exit] + SLIPPAGE_POINTS (worse by 1 tick)

This means a round-turn trade incurs 2 ticks of slippage total (1 on entry, 1 on exit).

### 4.3 Exact Commission Model
```python
# Commission: $2.50 per contract round-turn
COMMISSION_RT: float = 2.50  # USD per contract per round-turn trade
```

Commission is charged once per completed trade (entry + exit = 1 round-turn).
For a trade with `size` contracts, total commission = `size * COMMISSION_RT`.

### 4.4 Exact Position Sizing at Entry
Position sizing is computed at signal generation time (Section 3.6) and carried through the backtest. The size for each trade is:
```python
size = clamp(round((100_000 * 0.15) / (sqrt(har_vol_forecast) * close * 50)), 1, 10)
```
The same size is used for both entry and exit of a given trade.

### 4.5 Exact P&L Computation Per Bar

The backtest tracks mark-to-market P&L at each bar for positions held:

```python
def compute_bar_pnl(position: int, size: int, entry_price: float, 
                    current_close: float, prev_close: float) -> float:
    """
    Compute P&L for one bar while holding a position.

    position: +1 (long) or -1 (short)
    size: number of contracts
    entry_price: price at which position was entered
    current_close: close of current bar
    prev_close: close of previous bar

    Returns: unrealized P&L change for this bar in USD
    """
    if position == 0:
        return 0.0

    # P&L = position_direction * size * point_value * (current_close - prev_close)
    bar_pnl = position * size * CONTRACT_SPECS['point_value'] * (current_close - prev_close)
    return bar_pnl
```

### 4.6 Exact Trade Logging

Each completed trade is logged as a dictionary with these EXACT fields:

```python
TRADE_LOG_SCHEMA = {
    'trade_id': int,           # Sequential trade number (1, 2, 3, ...)
    'entry_bar': int,          # Index of entry bar
    'exit_bar': int,           # Index of exit bar
    'entry_time': pd.Timestamp, # Entry timestamp
    'exit_time': pd.Timestamp,  # Exit timestamp
    'direction': str,          # 'LONG' or 'SHORT'
    'size': int,               # Number of contracts
    'entry_price': float,      # Fill price at entry (with slippage)
    'exit_price': float,       # Fill price at exit (with slippage)
    'hold_bars': int,          # Bars held
    'gross_pnl': float,        # P&L before costs in USD
    'commission': float,       # Commission paid
    'slippage_cost': float,    # Total slippage cost
    'net_pnl': float,          # gross_pnl - commission - slippage_cost
    'entry_signal': str,       # 'toxic_cont' or 'toxic_rev'
    'exit_reason': str,        # 'time_exit' or 'regime_exit'
}
```

### 4.7 Exact Backtest Function

```python
def backtest(df_with_signals: pd.DataFrame) -> dict:
    """
    Run complete backtest on signal-generated DataFrame.

    Parameters
    ----------
    df_with_signals : pd.DataFrame — output from generate_signals()
        Must contain columns: close, signal, position_size, entry_price, 
        hold_bars, trade_pnl

    Returns
    -------
    dict with keys:
        'equity_curve': pd.Series — cumulative equity over time
        'trade_log': pd.DataFrame — one row per completed trade
        'daily_returns': pd.Series — daily returns for Sharpe calculation
        'metrics': dict — summary performance metrics
        'bar_pnl': pd.Series — P&L per bar
    """
    df = df_with_signals.copy()
    n = len(df)

    # Arrays for efficient computation
    bar_pnl = np.zeros(n)           # P&L per bar
    position_array = np.zeros(n)    # Position at each bar (-1, 0, +1)
    size_array = np.zeros(n)        # Size at each bar
    equity = np.zeros(n)            # Cumulative equity

    trades = []                     # List of completed trade dicts
    trade_id = 0

    # Running state
    in_position = False
    position_direction = 0
    current_size = 0
    entry_bar_idx = -1
    entry_price_slipped = 0.0
    entry_signal_type = ''

    # Process bar by bar
    for i in range(1, n):  # Start from 1 to have previous close
        signal = df['signal'].iloc[i]
        prev_signal = df['signal'].iloc[i-1] if i > 0 else 0

        # Mark-to-market for existing position
        if in_position:
            price_change = df['close'].iloc[i] - df['close'].iloc[i-1]
            bar_pnl[i] = position_direction * current_size * CONTRACT_SPECS['point_value'] * price_change

        # Detect position changes
        if not in_position and signal != SIGNAL_FLAT:
            # NEW ENTRY
            in_position = True
            position_direction = signal
            current_size = int(df['position_size'].iloc[i])
            entry_bar_idx = i

            # Apply slippage to entry
            if position_direction == SIGNAL_LONG:
                entry_price_slipped = df['close'].iloc[i] + SLIPPAGE_POINTS
                entry_signal_type = 'toxic_cont'
            else:
                entry_price_slipped = df['close'].iloc[i] - SLIPPAGE_POINTS
                entry_signal_type = 'toxic_rev'

            # Record entry slippage cost
            slippage_entry = SLIPPAGE_POINTS * current_size * CONTRACT_SPECS['point_value']

        elif in_position and signal == SIGNAL_FLAT:
            # EXIT
            # Apply slippage to exit
            if position_direction == SIGNAL_LONG:
                exit_price_slipped = df['close'].iloc[i] - SLIPPAGE_POINTS
            else:
                exit_price_slipped = df['close'].iloc[i] + SLIPPAGE_POINTS

            # Gross P&L
            if position_direction == SIGNAL_LONG:
                gross_pnl = (exit_price_slipped - entry_price_slipped) * current_size * CONTRACT_SPECS['point_value']
            else:
                gross_pnl = (entry_price_slipped - exit_price_slipped) * current_size * CONTRACT_SPECS['point_value']

            # Costs
            slippage_exit = SLIPPAGE_POINTS * current_size * CONTRACT_SPECS['point_value']
            total_slippage = slippage_entry + slippage_exit
            commission = current_size * COMMISSION_RT
            net_pnl = gross_pnl - commission - total_slippage

            # Determine exit reason
            hold_bars = i - entry_bar_idx
            exit_reason = 'time_exit' if hold_bars >= df['hold_bars'].iloc[i] else 'regime_exit'

            # Log trade
            trade_id += 1
            trades.append({
                'trade_id': trade_id,
                'entry_bar': entry_bar_idx,
                'exit_bar': i,
                'entry_time': df['ts_event'].iloc[entry_bar_idx],
                'exit_time': df['ts_event'].iloc[i],
                'direction': 'LONG' if position_direction == SIGNAL_LONG else 'SHORT',
                'size': current_size,
                'entry_price': entry_price_slipped,
                'exit_price': exit_price_slipped,
                'hold_bars': hold_bars,
                'gross_pnl': gross_pnl,
                'commission': commission,
                'slippage_cost': total_slippage,
                'net_pnl': net_pnl,
                'entry_signal': entry_signal_type,
                'exit_reason': exit_reason,
            })

            # Reset state
            in_position = False
            position_direction = 0
            current_size = 0
            entry_bar_idx = -1

        # Record position for this bar
        position_array[i] = position_direction if in_position else 0
        size_array[i] = current_size if in_position else 0

    # Compute equity curve
    equity = np.cumsum(bar_pnl)

    # Create output
    df['bar_pnl'] = bar_pnl
    df['position'] = position_array
    df['size'] = size_array
    df['equity'] = equity

    trade_log = pd.DataFrame(trades)

    # Compute daily returns for Sharpe
    df['date'] = df['ts_event'].dt.date
    daily_pnl = df.groupby('date')['bar_pnl'].sum()
    # Approximate daily return relative to account value
    daily_returns = daily_pnl / ACCOUNT_VALUE

    result = {
        'equity_curve': pd.Series(equity, index=df.index),
        'trade_log': trade_log,
        'daily_returns': daily_returns,
        'bar_pnl': pd.Series(bar_pnl, index=df.index),
        'df_with_pnl': df,
    }

    return result
```

### 4.8 Exact Same-Bar Signal Handling
When a signal changes from non-zero to zero on the same bar (exit), the exit is processed at that bar's close with slippage applied. The P&L for the exit bar includes the price movement from the previous close to the current close (mark-to-market), plus the exit slippage cost. No additional P&L is computed beyond the bar close.

### 4.9 Exact Equity Curve Computation
The equity curve is the cumulative sum of bar P&Ls:
```
equity[i] = Σ_{j=0}^{i} bar_pnl[j]
```
Starting from 0. The equity curve represents total unrealized + realized P&L over time. It does NOT start from the account value — it represents cumulative trading profit/loss.

### 4.10 Cost Summary Per Trade
For a trade of `size` contracts held for `hold_bars` bars:
```
Total Cost = (2 * SLIPPAGE_TICKS * size * TICK_VALUE) + (size * COMMISSION_RT)
           = (2 * 1 * size * $12.50) + (size * $2.50)
           = size * $27.50
```

For a 1-contract trade: total cost = $27.50 per round-turn.
For a 5-contract trade: total cost = $137.50 per round-turn.
For a 10-contract trade: total cost = $275.00 per round-turn.

The breakeven price move per contract:
```
breakeven_points = Total Cost / (size * point_value) = $27.50 * size / (size * $50) = 0.55 points
breakeven_ticks = 0.55 / 0.25 = 2.2 ticks
```

So each trade must make at least 2.2 ticks (0.55 index points) net of costs to be profitable.

---

## SECTION 5: VALIDATION FRAMEWORK

### 5.1 Validation Philosophy
All parameter optimization and model fitting is performed EXCLUSIVELY on the training set (indices 0-85599). The test set (indices 85600-122295) is used for final out-of-sample evaluation ONLY. No hyperparameter tuning, no model refitting, and no threshold adjustment is permitted on the test set.

### 5.2 Walk-Forward Analysis Parameters (Training Set Only)

The 3 optimized parameters (τ1, τ2, N) are selected using walk-forward analysis on the training set:

```python
WALK_FORWARD_PARAMS = {
    'train_window_bars': 39_000,    # ~500 trading days for training
    'test_window_bars': 7_800,      # ~100 trading days for validation
    'step_size_bars': 3_900,        # ~50 trading days step
}
```

Walk-forward procedure:
1. Split training data into overlapping windows
2. For each window: fit models on train_window, evaluate on test_window
3. Average performance across all test windows
4. Select parameter combination with highest average Sharpe ratio

Window schedule (train set = 85,600 bars):
| Window | Train Start | Train End | Test Start | Test End |
|--------|-------------|-----------|------------|----------|
| 1 | 0 | 38,999 | 39,000 | 46,799 |
| 2 | 3,900 | 42,899 | 42,900 | 50,699 |
| 3 | 7,800 | 46,799 | 46,800 | 54,599 |
| 4 | 11,700 | 50,699 | 50,700 | 58,499 |
| 5 | 15,600 | 54,599 | 54,600 | 62,399 |
| 6 | 19,500 | 58,499 | 58,500 | 66,299 |
| 7 | 23,400 | 62,399 | 62,400 | 70,199 |
| 8 | 27,300 | 66,299 | 66,300 | 74,099 |
| 9 | 31,200 | 70,199 | 70,200 | 77,999 |
| 10 | 35,100 | 74,099 | 74,100 | 81,899 |

Total windows: 10

### 5.3 Parameter Search Space

```python
PARAM_SEARCH_SPACE = {
    'toxic_continuation_threshold': np.arange(0.40, 0.81, 0.05),  # [0.40, 0.45, ..., 0.80]
    'toxic_reversal_threshold': np.arange(0.40, 0.81, 0.05),       # [0.40, 0.45, ..., 0.80]
    'max_hold_bars': [5, 10, 15, 20, 25, 30],                      # Discrete values
}
```

Total combinations: 9 * 9 * 6 = 486. This is feasible for grid search on the walk-forward windows.

### 5.4 Exact Performance Metrics

#### 5.4.1 Sharpe Ratio
```python
def compute_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.

    Sharpe = sqrt(252) * (mean(daily_returns) - risk_free_rate) / std(daily_returns)

    If std(daily_returns) == 0, return 0.0.
    """
    excess = daily_returns - risk_free_rate / 252  # Daily risk-free
    if excess.std() == 0:
        return 0.0
    sharpe = np.sqrt(252) * excess.mean() / excess.std()
    return float(sharpe)
```

#### 5.4.2 Maximum Drawdown
```python
def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """
    Maximum drawdown as fraction of peak equity.

    MDD = max_t (peak_t - equity_t) / peak_t
    where peak_t = max(equity[0:t+1])

    Returns positive value (e.g., 0.15 means 15% drawdown).
    """
    running_peak = equity_curve.cummax()
    drawdown = (running_peak - equity_curve) / running_peak
    mdd = drawdown.max()
    return float(mdd)
```

#### 5.4.3 Directional Accuracy (Win Rate)
```python
def compute_win_rate(trade_log: pd.DataFrame) -> float:
    """
    Fraction of trades with positive net P&L.

    win_rate = count(net_pnl > 0) / count(total_trades)
    """
    if len(trade_log) == 0:
        return 0.0
    wins = (trade_log['net_pnl'] > 0).sum()
    return float(wins / len(trade_log))
```

#### 5.4.4 Profit Factor
```python
def compute_profit_factor(trade_log: pd.DataFrame) -> float:
    """
    Gross profits / Gross losses.

    profit_factor = sum(net_pnl[net_pnl > 0]) / abs(sum(net_pnl[net_pnl < 0]))
    """
    gross_profits = trade_log['net_pnl'][trade_log['net_pnl'] > 0].sum()
    gross_losses = abs(trade_log['net_pnl'][trade_log['net_pnl'] < 0].sum())
    if gross_losses == 0:
        return float('inf') if gross_profits > 0 else 0.0
    return float(gross_profits / gross_losses)
```

#### 5.4.5 Trades Per Day
```python
def compute_trades_per_day(trade_log: pd.DataFrame, num_trading_days: int) -> float:
    """
    Average number of trades per trading day.
    """
    if num_trading_days == 0:
        return 0.0
    return float(len(trade_log) / num_trading_days)
```

#### 5.4.6 Average Trade P&L
```python
def compute_avg_trade_pnl(trade_log: pd.DataFrame) -> float:
    """
    Average net P&L per trade in USD.
    """
    if len(trade_log) == 0:
        return 0.0
    return float(trade_log['net_pnl'].mean())
```

#### 5.4.7 Calmar Ratio
```python
def compute_calmar(daily_returns: pd.Series, max_dd: float) -> float:
    """
    Calmar ratio = annualized_return / max_drawdown
    """
    if max_dd == 0:
        return 0.0
    ann_return = daily_returns.mean() * 252
    return float(ann_return / max_dd)
```

### 5.5 Exact Statistical Tests

#### 5.5.1 Sharpe Ratio Significance (t-test)
```python
from scipy import stats

def sharpe_significance_test(daily_returns: pd.Series, target_sharpe: float = 1.0) -> tuple:
    """
    Test whether the realized Sharpe ratio is significantly greater than target.

    H0: Sharpe <= target_sharpe
    H1: Sharpe > target_sharpe

    Uses the asymptotic distribution: Sharpe ~ N(true_sharpe, 1/sqrt(T))
    where T = number of daily observations.

    Returns: (t_statistic, p_value, significant_at_5pct)
    """
    T = len(daily_returns)
    if T < 30 or daily_returns.std() == 0:
        return 0.0, 1.0, False

    realized_sharpe = compute_sharpe(daily_returns)
    sharpe_se = 1.0 / np.sqrt(T)
    t_stat = (realized_sharpe - target_sharpe) / sharpe_se
    p_value = 1 - stats.norm.cdf(t_stat)  # One-sided test

    return t_stat, p_value, p_value < 0.05
```

#### 5.5.2 Win Rate Significance (Binomial Test)
```python
def win_rate_significance_test(trade_log: pd.DataFrame, 
                                target_rate: float = 0.54) -> tuple:
    """
    Test whether win rate is significantly greater than target.

    H0: win_rate <= target_rate
    H1: win_rate > target_rate

    Uses exact binomial test.
    """
    n = len(trade_log)
    if n < 30:
        return 0.0, 1.0, False

    wins = (trade_log['net_pnl'] > 0).sum()
    p_value = stats.binom_test(wins, n, target_rate, alternative='greater')

    # Normal approximation for t-statistic
    se = np.sqrt(target_rate * (1 - target_rate) / n)
    t_stat = (wins / n - target_rate) / se if se > 0 else 0

    return t_stat, p_value, p_value < 0.05
```

### 5.6 Anti-Curve-Fit Constraints

#### 5.6.1 Parameter Count Limit
**HARD CONSTRAINT**: Exactly 3 parameters are optimized:
1. τ1 = toxic_continuation_threshold
2. τ2 = toxic_reversal_threshold  
3. N = max_hold_bars

No other parameters may be optimized. All other values are fixed by domain knowledge or data structure.

#### 5.6.2 Minimum Trade Count
**HARD CONSTRAINT**: Any parameter combination producing fewer than 100 trades across a walk-forward window is automatically rejected (assigned Sharpe = -999).

#### 5.6.3 Minimum Win Rate
**HARD CONSTRAINT**: Any parameter combination with win rate < 50% is rejected.

#### 5.6.4 Parameter Stability Check
After selecting the best parameter combination, verify that performance is robust to small perturbations:
- τ1 ± 0.05 must produce Sharpe within 20% of optimal
- τ2 ± 0.05 must produce Sharpe within 20% of optimal
- N ± 5 must produce Sharpe within 20% of optimal

If not, the parameter landscape is too noisy — consider a more conservative (central) parameter choice.

### 5.7 Complete Validation Function

```python
def validate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """
    Complete validation pipeline.

    1. Fit all models on full training set
    2. Optimize 3 parameters via walk-forward analysis on training set
    3. Run final backtest on test set with frozen parameters
    4. Compute all performance metrics and statistical tests

    Returns comprehensive results dict.
    """
    # === PHASE 1: Fit models on full training set ===
    print("Phase 1: Fitting models on training set...")
    train_features, fitted_models = engineer_features(train_df, fit_models=True)

    # === PHASE 2: Walk-forward parameter optimization ===
    print("Phase 2: Walk-forward parameter optimization...")
    best_params = None
    best_score = -np.inf

    param_grid = [
        {'toxic_continuation_threshold': t1, 
         'toxic_reversal_threshold': t2,
         'max_hold_bars': n,
         'regime_exit_enabled': True}
        for t1 in PARAM_SEARCH_SPACE['toxic_continuation_threshold']
        for t2 in PARAM_SEARCH_SPACE['toxic_reversal_threshold']
        for n in PARAM_SEARCH_SPACE['max_hold_bars']
    ]

    total_combos = len(param_grid)
    print(f"Testing {total_combos} parameter combinations...")

    for combo_idx, params in enumerate(param_grid):
        window_sharpes = []

        for w_start in range(0, len(train_df) - WALK_FORWARD_PARAMS['train_window_bars'] - 
                              WALK_FORWARD_PARAMS['test_window_bars'] + 1,
                              WALK_FORWARD_PARAMS['step_size_bars']):

            w_train_end = w_start + WALK_FORWARD_PARAMS['train_window_bars']
            w_test_end = w_train_end + WALK_FORWARD_PARAMS['test_window_bars']

            # Extract windows
            w_train = train_df.iloc[w_start:w_train_end]
            w_test = train_df.iloc[w_train_end:w_test_end]

            # Engineer features for this window
            w_train_feat, w_fitted = engineer_features(w_train, fit_models=True)
            w_test_feat, _ = engineer_features(w_test, fit_models=False,
                                               har_params=w_fitted['har_params'],
                                               hmm_model=w_fitted['hmm_model'],
                                               scaler=w_fitted['scaler'],
                                               vol_thresholds=w_fitted['vol_thresholds'])

            # Generate signals and backtest
            w_signals = generate_signals(w_test_feat, params)
            w_result = backtest(w_signals)

            # Compute Sharpe
            if len(w_result['trade_log']) >= 100:
                sharpe = compute_sharpe(w_result['daily_returns'])
                win_rate = compute_win_rate(w_result['trade_log'])
                if win_rate >= 0.50:
                    window_sharpes.append(sharpe)
                else:
                    window_sharpes.append(-999)
            else:
                window_sharpes.append(-999)

        avg_sharpe = np.mean([s for s in window_sharpes if s > -900]) if window_sharpes else -999

        if avg_sharpe > best_score:
            best_score = avg_sharpe
            best_params = params.copy()

    print(f"Best parameters: {best_params}")
    print(f"Best walk-forward Sharpe: {best_score:.4f}")

    # === PHASE 3: Final test set evaluation ===
    print("Phase 3: Test set evaluation...")
    test_features, _ = engineer_features(test_df, fit_models=False,
                                          har_params=fitted_models['har_params'],
                                          hmm_model=fitted_models['hmm_model'],
                                          scaler=fitted_models['scaler'],
                                          vol_thresholds=fitted_models['vol_thresholds'])

    test_signals = generate_signals(test_features, best_params)
    test_result = backtest(test_signals)

    # === PHASE 4: Compute all metrics ===
    print("Phase 4: Computing metrics...")
    metrics = {}

    # Core metrics
    metrics['sharpe_ratio'] = compute_sharpe(test_result['daily_returns'])
    metrics['max_drawdown'] = compute_max_drawdown(test_result['equity_curve'])
    metrics['win_rate'] = compute_win_rate(test_result['trade_log'])
    metrics['profit_factor'] = compute_profit_factor(test_result['trade_log'])

    num_test_days = test_df['ts_event'].dt.date.nunique()
    metrics['trades_per_day'] = compute_trades_per_day(test_result['trade_log'], num_test_days)
    metrics['avg_trade_pnl'] = compute_avg_trade_pnl(test_result['trade_log'])
    metrics['calmar_ratio'] = compute_calmar(test_result['daily_returns'], metrics['max_drawdown'])
    metrics['total_trades'] = len(test_result['trade_log'])
    metrics['total_pnl'] = test_result['trade_log']['net_pnl'].sum() if len(test_result['trade_log']) > 0 else 0

    # Statistical tests
    t_sharpe, p_sharpe, sig_sharpe = sharpe_significance_test(test_result['daily_returns'])
    metrics['sharpe_t_stat'] = t_sharpe
    metrics['sharpe_p_value'] = p_sharpe
    metrics['sharpe_significant_5pct'] = sig_sharpe

    t_win, p_win, sig_win = win_rate_significance_test(test_result['trade_log'])
    metrics['winrate_t_stat'] = t_win
    metrics['winrate_p_value'] = p_win
    metrics['winrate_significant_5pct'] = sig_win

    # Parameter info
    metrics['best_params'] = best_params
    metrics['walk_forward_sharpe'] = best_score

    # === PHASE 5: Acceptance criteria ===
    print("Phase 5: Acceptance check...")
    passed = True
    checks = {
        'sharpe_gt_1.0': metrics['sharpe_ratio'] > 1.0,
        'winrate_gt_54pct': metrics['win_rate'] > 0.54,
        'trades_per_day_gt_5': metrics['trades_per_day'] > 5,
        'mdd_lt_15pct': metrics['max_drawdown'] < 0.15,
        'profit_factor_gt_1.3': metrics['profit_factor'] > 1.3,
    }

    for check_name, result in checks.items():
        metrics[f'check_{check_name}'] = result
        if not result:
            passed = False
            print(f"  FAIL: {check_name}")
        else:
            print(f"  PASS: {check_name}")

    metrics['all_checks_passed'] = passed

    return {
        'train_features': train_features,
        'test_features': test_features,
        'fitted_models': fitted_models,
        'best_params': best_params,
        'test_result': test_result,
        'metrics': metrics,
    }
```

---

## SECTION 6: CODE ARCHITECTURE

### 6.1 Complete File Structure

```
strategy_01_flowtox_regime/
├── __init__.py
├── config.py              # All constants, parameters, contract specs
├── data_pipeline.py       # Load, validate, split data
├── features.py            # Feature engineering functions
├── hmm_model.py           # HMM/GMM fitting and inference
├── har_model.py           # HAR-RV fitting and forecasting
├── signal_generator.py    # Entry/exit signal generation
├── backtest.py            # Backtesting engine
├── metrics.py             # Performance metrics and statistical tests
├── validation.py          # Walk-forward optimization and validation
├── main.py                # Main execution script
└── tests/
    ├── test_data.py       # Data loading tests
    ├── test_features.py   # Feature computation tests
    └── test_backtest.py   # Backtest correctness tests
```

### 6.2 Module: config.py

```python
"""
Configuration module — all hard-coded constants and default parameters.
NO values in this file should be modified during optimization.
Only the 3 parameters in PARAM_SEARCH_SPACE are optimized.
"""

import numpy as np

# ============================================================================
# DATA SPECIFICATION
# ============================================================================
CSV_FILE_PATH: str = "/mnt/agents/upload/ES_5min_RTH_6year.csv"
BARS_PER_DAY: int = 78  # 09:30-16:00 ET at 5-min frequency
TOTAL_ROWS: int = 122296
TRAIN_END_INDEX: int = 85599  # Inclusive

# ============================================================================
# CONTRACT SPECIFICATIONS (E-mini S&P 500)
# ============================================================================
CONTRACT_SPECS = {
    'tick_size': 0.25,
    'tick_value': 12.50,
    'point_value': 50.00,
    'symbol': 'ES',
    'exchange': 'CME',
}

# ============================================================================
# SLIPPAGE AND COMMISSION
# ============================================================================
SLIPPAGE_TICKS: int = 1
SLIPPAGE_POINTS: float = SLIPPAGE_TICKS * CONTRACT_SPECS['tick_size']  # 0.25
COMMISSION_RT: float = 2.50  # USD per contract round-turn

# ============================================================================
# POSITION SIZING
# ============================================================================
ACCOUNT_VALUE: float = 100_000.0  # Reference account in USD
SIGMA_TARGET: float = 0.15        # 15% annualized target volatility
MAX_SIZE: int = 10                # Maximum contracts per trade

# ============================================================================
# HAR MODEL PARAMETERS
# ============================================================================
HAR_DAILY_WINDOW: int = 1     # 1 trading day
HAR_WEEKLY_WINDOW: int = 5    # 5 trading days
HAR_MONTHLY_WINDOW: int = 22  # ~1 month trading days

# ============================================================================
# HMM PARAMETERS
# ============================================================================
HMM_N_COMPONENTS: int = 3
HMM_COVARIANCE_TYPE: str = 'full'
HMM_LOOKBACK: int = 20  # Bars for rolling HMM features
HMM_RANDOM_STATE: int = 42
HMM_N_INIT: int = 10
HMM_MAX_ITER: int = 200

# ============================================================================
# VOLATILITY REGIME
# ============================================================================
VOL_REGIME_P33: float = 0.33  # 33rd percentile threshold
VOL_REGIME_P67: float = 0.67  # 67th percentile threshold

# ============================================================================
# OPTIMIZABLE PARAMETERS (exactly 3)
# ============================================================================
PARAM_DEFAULTS = {
    'toxic_continuation_threshold': 0.55,  # τ1
    'toxic_reversal_threshold': 0.55,       # τ2
    'max_hold_bars': 15,                    # N
    'regime_exit_enabled': True,
}

PARAM_SEARCH_SPACE = {
    'toxic_continuation_threshold': np.arange(0.40, 0.81, 0.05),
    'toxic_reversal_threshold': np.arange(0.40, 0.81, 0.05),
    'max_hold_bars': [5, 10, 15, 20, 25, 30],
}

# ============================================================================
# WALK-FORWARD PARAMETERS
# ============================================================================
WALK_FORWARD = {
    'train_window_bars': 39_000,
    'test_window_bars': 7_800,
    'step_size_bars': 3_900,
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
    'min_sharpe': 1.0,
    'min_win_rate': 0.54,
    'min_trades_per_day': 5,
    'max_drawdown': 0.15,
    'min_profit_factor': 1.3,
}
```

### 6.3 Module: data_pipeline.py

```python
"""Data loading, validation, and splitting functions."""
import pandas as pd
import numpy as np
from config import CSV_FILE_PATH, TRAIN_END_INDEX, BARS_PER_DAY, CONTRACT_SPECS

def load_data(csv_path: str = CSV_FILE_PATH) -> pd.DataFrame:
    """Load and validate raw CSV data. Returns validated DataFrame."""
    ...  # (See Section 1.4 for full implementation)

def verify_rth_session(df: pd.DataFrame) -> pd.DataFrame:
    """Verify all bars fall within RTH session."""
    ...  # (See Section 1.5 for full implementation)

def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train/test at fixed index."""
    ...  # (See Section 1.6 for full implementation)

def validate_data(df: pd.DataFrame) -> None:
    """Run all data validation assertions. Raises ValueError on failure."""
    ...  # (See Section 1.7 for full checklist)
```

### 6.4 Module: features.py

```python
"""Feature engineering functions."""
import pandas as pd
import numpy as np
from config import BARS_PER_DAY, HAR_WEEKLY_WINDOW, HAR_MONTHLY_WINDOW, HMM_LOOKBACK

def compute_bvc_signed_volume(df: pd.DataFrame) -> pd.Series:
    """Compute BVC-signed volume."""
    ...

def compute_tick_rule_direction(df: pd.DataFrame) -> pd.Series:
    """Compute tick-rule direction."""
    ...

def compute_toxicity_disagreement(bvc: pd.Series, tick_dir: pd.Series) -> pd.Series:
    """Compute toxicity disagreement."""
    ...

def compute_parkinson_bar_variance(df: pd.DataFrame) -> pd.Series:
    """Compute per-bar Parkinson variance."""
    ...

def compute_parkinson_realized_variances(df: pd.DataFrame, parkinson_bar: pd.Series) -> pd.DataFrame:
    """Compute daily, weekly, monthly RV."""
    ...

def compute_hmm_observable_features(df: pd.DataFrame, toxicity: pd.Series,
                                     bvc: pd.Series, parkinson_bar: pd.Series) -> pd.DataFrame:
    """Compute 5 observable features for HMM."""
    ...

def engineer_features(df: pd.DataFrame, har_params: dict = None,
                      hmm_model=None, scaler=None, vol_thresholds: dict = None,
                      fit_models: bool = True) -> tuple[pd.DataFrame, dict]:
    """Master feature engineering pipeline."""
    ...  # (See Section 2.13 for full implementation)
```

### 6.5 Module: har_model.py

```python
"""HAR-RV model fitting and forecasting."""
import numpy as np
import pandas as pd
from config import BARS_PER_DAY

def fit_har_model(rv_1d: pd.Series, rv_1w: pd.Series, rv_1m: pd.Series) -> dict:
    """Fit HAR-RV via OLS. Returns {intercept, beta_d, beta_w, beta_m}."""
    ...

def compute_har_vol_forecast(rv_1d: pd.Series, rv_1w: pd.Series, rv_1m: pd.Series,
                              har_params: dict) -> pd.Series:
    """Compute HAR volatility forecast."""
    ...
```

### 6.6 Module: hmm_model.py

```python
"""HMM/GMM model fitting and inference."""
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from config import HMM_N_COMPONENTS, HMM_COVARIANCE_TYPE, HMM_RANDOM_STATE, HMM_N_INIT, HMM_MAX_ITER

def fit_hmm_model(obs_features_train: pd.DataFrame, 
                  random_state: int = HMM_RANDOM_STATE) -> tuple[GaussianMixture, StandardScaler]:
    """Fit 3-state GMM. Returns (fitted_model, fitted_scaler)."""
    ...

def compute_hmm_posteriors(obs_features: pd.DataFrame,
                            hmm_model: GaussianMixture,
                            scaler: StandardScaler) -> pd.DataFrame:
    """Compute state posterior probabilities."""
    ...
```

### 6.7 Module: signal_generator.py

```python
"""Signal generation engine."""
import pandas as pd
import numpy as np
from config import (SIGNAL_FLAT, SIGNAL_LONG, SIGNAL_SHORT, MAX_SIZE, 
                    ACCOUNT_VALUE, SIGMA_TARGET, CONTRACT_SPECS, SLIPPAGE_POINTS)

def compute_position_size(har_forecast: float, close_price: float) -> int:
    """Compute position size via inverse-vol targeting."""
    forecast_vol = np.sqrt(max(har_forecast, 1e-12))
    contract_vol_usd = forecast_vol * close_price * CONTRACT_SPECS['point_value']
    size_raw = (ACCOUNT_VALUE * SIGMA_TARGET) / contract_vol_usd
    return int(np.clip(round(size_raw), 1, MAX_SIZE))

def generate_signals(features_df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Generate trading signals."""
    ...  # (See Section 3.7 for full implementation)
```

### 6.8 Module: backtest.py

```python
"""Backtesting engine."""
import pandas as pd
import numpy as np
from config import (SIGNAL_FLAT, SIGNAL_LONG, SIGNAL_SHORT, CONTRACT_SPECS,
                    SLIPPAGE_POINTS, COMMISSION_RT)

def backtest(df_with_signals: pd.DataFrame) -> dict:
    """Run backtest. Returns {equity_curve, trade_log, daily_returns, bar_pnl, df_with_pnl}."""
    ...  # (See Section 4.7 for full implementation)
```

### 6.9 Module: metrics.py

```python
"""Performance metrics and statistical tests."""
import numpy as np
import pandas as pd
from scipy import stats

def compute_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    ...

def compute_max_drawdown(equity_curve: pd.Series) -> float:
    ...

def compute_win_rate(trade_log: pd.DataFrame) -> float:
    ...

def compute_profit_factor(trade_log: pd.DataFrame) -> float:
    ...

def compute_trades_per_day(trade_log: pd.DataFrame, num_trading_days: int) -> float:
    ...

def compute_avg_trade_pnl(trade_log: pd.DataFrame) -> float:
    ...

def compute_calmar(daily_returns: pd.Series, max_dd: float) -> float:
    ...

def sharpe_significance_test(daily_returns: pd.Series, target_sharpe: float = 1.0) -> tuple:
    ...

def win_rate_significance_test(trade_log: pd.DataFrame, target_rate: float = 0.54) -> tuple:
    ...
```

### 6.10 Module: validation.py

```python
"""Walk-forward optimization and validation framework."""
import pandas as pd
import numpy as np
from config import (PARAM_SEARCH_SPACE, WALK_FORWARD, SIGNAL_FLAT)
from features import engineer_features
from signal_generator import generate_signals
from backtest import backtest
from metrics import (compute_sharpe, compute_max_drawdown, compute_win_rate,
                     compute_profit_factor, compute_trades_per_day, compute_avg_trade_pnl,
                     compute_calmar, sharpe_significance_test, win_rate_significance_test)

def run_walk_forward_optimization(train_df: pd.DataFrame) -> dict:
    """Optimize 3 parameters via walk-forward analysis. Returns best_params dict."""
    ...

def run_test_evaluation(test_df: pd.DataFrame, fitted_models: dict, 
                        best_params: dict) -> dict:
    """Run final backtest on test set with frozen parameters."""
    ...

def validate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Complete validation pipeline."""
    ...  # (See Section 5.7 for full implementation)
```

### 6.11 Module: main.py

```python
#!/usr/bin/env python3
"""
Strategy 1: Adaptive Flow-Toxicity with Regime-Aware Sizing
Main execution script.

Run with: python -m strategy_01_flowtox_regime.main
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from config import CSV_FILE_PATH, ACCEPTANCE_CRITERIA
from data_pipeline import load_data, split_train_test, validate_data
from validation import validate

def main():
    print("=" * 70)
    print("Strategy 1: Adaptive Flow-Toxicity with Regime-Aware Sizing")
    print(f"Execution time: {datetime.now().isoformat()}")
    print("=" * 70)

    # Step 1: Load data
    print("
[1/5] Loading data...")
    df = load_data(CSV_FILE_PATH)
    validate_data(df)
    print(f"  Loaded {len(df):,} rows")

    # Step 2: Split
    print("
[2/5] Splitting train/test...")
    train_df, test_df = split_train_test(df)

    # Step 3: Validate (fits models, optimizes params, tests)
    print("
[3/5] Running validation pipeline...")
    results = validate(train_df, test_df)

    # Step 4: Report metrics
    print("
[4/5] Results:")
    print("-" * 50)
    metrics = results['metrics']
    print(f"Best parameters: {metrics['best_params']}")
    print(f"Walk-forward Sharpe: {metrics['walk_forward_sharpe']:.4f}")
    print(f"Test Sharpe Ratio:   {metrics['sharpe_ratio']:.4f}")
    print(f"Test Max Drawdown:   {metrics['max_drawdown']:.4%}")
    print(f"Test Win Rate:       {metrics['win_rate']:.4%}")
    print(f"Test Profit Factor:  {metrics['profit_factor']:.4f}")
    print(f"Test Trades/Day:     {metrics['trades_per_day']:.2f}")
    print(f"Test Avg Trade P&L:  ${metrics['avg_trade_pnl']:.2f}")
    print(f"Test Total Trades:   {metrics['total_trades']}")
    print(f"Test Total P&L:      ${metrics['total_pnl']:,.2f}")
    print(f"Sharpe Significant:  {metrics['sharpe_significant_5pct']}")
    print(f"WinRate Significant: {metrics['winrate_significant_5pct']}")
    print("-" * 50)

    # Step 5: Acceptance
    print("
[5/5] Acceptance Criteria:")
    all_passed = True
    for criterion, value in ACCEPTANCE_CRITERIA.items():
        actual = metrics.get(criterion.replace('min_', '').replace('max_', ''), 'N/A')
        passed = metrics.get(f'check_{criterion}', False)
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_passed = False
        print(f"  [{status}] {criterion}: target={value}, actual={actual}")

    print(f"
{'=' * 50}")
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
    print(f"{'=' * 50}")

    # Save results
    save_results(results)

    return results

def save_results(results: dict) -> None:
    """Save results to output files."""
    import os
    os.makedirs('output', exist_ok=True)

    # Save metrics as JSON
    metrics_serializable = {k: v for k, v in results['metrics'].items() 
                           if isinstance(v, (int, float, bool, str))}
    with open('output/strategy_01_metrics.json', 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    # Save trade log
    results['test_result']['trade_log'].to_csv('output/strategy_01_trades.csv', index=False)

    # Save equity curve
    results['test_result']['equity_curve'].to_csv('output/strategy_01_equity.csv')

    print("
Results saved to output/ directory.")

if __name__ == '__main__':
    results = main()
```

### 6.12 Import Statements Summary

```python
# Core imports required across all modules
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy import stats
import matplotlib.pyplot as plt
import json
import warnings
```

### 6.13 Error Handling

All functions must implement the following error handling:

1. **Data loading errors**: Raise `ValueError` with descriptive message
2. **Feature computation errors**: Log warning, fill with neutral value (0), continue
3. **Model fitting errors**: If HMM fails to converge, retry with `n_init=5`, `max_iter=500`
4. **Signal generation errors**: If parameters are out of valid range, clamp to nearest valid value
5. **Backtest errors**: If trade log is empty, return zero-filled metrics (not NaN)
6. **Validation errors**: If walk-forward windows are insufficient, reduce window sizes by 50% and retry

### 6.14 Variable Naming Conventions

| Variable Pattern | Meaning |
|-----------------|---------|
| `df_*` | DataFrame |
| `s_*` or `*_series` | pd.Series |
| `n_*` | Integer count |
| `idx_*` | Index/position |
| `τ1`, `τ2` | Greek tau for thresholds (or `tau1`, `tau2` in code) |
| `*_train`, `*_test` | Train/test split |
| `*_window` | Windowed subset |
| `bar_*` | Per-bar value |
| `daily_*` | Daily-aggregated value |
| `rv_*` | Realized variance |
| `hmm_*` | HMM-related |
| `har_*` | HAR-related |
| `SIGNAL_*` | Signal constants |

---

## SECTION 7: STEP-BY-STEP BUILD ORDER

### Build Philosophy
Build the strategy incrementally, testing each component before moving to the next. Each step produces validated output that becomes input to subsequent steps. Do NOT proceed to the next step until the current step passes its validation checks.

---

### Step 1: Data Loading and Validation
**What to code**: `config.py` and `data_pipeline.py`

**Implementation order**:
1. Create `config.py` with ALL constants from Section 6.2
2. Implement `load_data()` function (Section 1.4)
3. Implement `verify_rth_session()` function (Section 1.5)
4. Implement `split_train_test()` function (Section 1.6)
5. Implement `validate_data()` function (Section 1.7)

**Expected output**: 
```python
train_df.shape == (85600, 7)
test_df.shape == (36695, 7)
train_df['ts_event'].min() == Timestamp('2020-01-02 14:30:00+0000', tz='UTC')
test_df['ts_event'].max() == Timestamp('2026-03-06 20:55:00+0000', tz='UTC')
```

**Validation checks**:
- [ ] No NaN in price columns after load
- [ ] No OHLC anomalies
- [ ] Timestamps are monotonic
- [ ] Train/test non-overlapping
- [ ] Exactly 85600 train rows, 36695 test rows

---

### Step 2: Basic Features (BVC, Tick Rule, Toxicity)
**What to code**: First 3 feature functions in `features.py`

**Implementation order**:
1. Implement `compute_bvc_signed_volume()` (Section 2.1)
2. Implement `compute_tick_rule_direction()` (Section 2.2)
3. Implement `compute_toxicity_disagreement()` (Section 2.3)

**Expected output**:
```python
train_df['bvc_signed_volume'].describe()
# count    85600.0
# mean       239.3
# std      12872.0
# min    -221102.3
# max     187007.0

train_df['tick_rule_direction'].value_counts()
#  1.0    ~50%
# -1.0    ~50%

train_df['toxicity_disagreement'].value_counts()
#  1.0    ~83%
# -1.0    ~14%  <- toxic
#  0.0     ~3%
```

**Validation checks**:
- [ ] |BVC| <= volume on all bars
- [ ] Tick rule is only {-1.0, +1.0}
- [ ] Toxicity is only {-1.0, 0.0, +1.0}
- [ ] First bar tick_rule = +1.0
- [ ] Zero-range bars have BVC = 0

---

### Step 3: Parkinson Variance Features
**What to code**: Parkinson variance functions in `features.py`

**Implementation order**:
1. Implement `compute_parkinson_bar_variance()` (Section 2.4)
2. Implement `compute_parkinson_realized_variances()` (Section 2.5)

**Expected output**:
```python
train_df['parkinson_bar_variance'].describe()
# All values >= 0
# Mean ~0.000004

train_df['rv_1d'].describe()
# All values >= 0
# Mean ~0.000105
# Annualized vol ~16.3%
```

**Validation checks**:
- [ ] All Parkinson variances >= 0
- [ ] Daily RV is sum of 78 bar variances
- [ ] Weekly RV is 5-day rolling mean of daily RV
- [ ] Monthly RV is 22-day rolling mean of daily RV
- [ ] All 78 bars in a day have same RV_1W and RV_1M

---

### Step 4: HAR Model
**What to code**: `har_model.py`

**Implementation order**:
1. Implement `fit_har_model()` (Section 2.6)
2. Implement `compute_har_vol_forecast()` (Section 2.6)

**Expected output**:
```python
har_params = fit_har_model(rv_1d, rv_1w, rv_1m)
# har_params ≈ {
#     'intercept': 0.000013,
#     'beta_d': 0.60,
#     'beta_w': 0.38,
#     'beta_m': 0.001,  # (clamped if negative)
# }
```

**Validation checks**:
- [ ] All coefficients non-negative
- [ ] Forecast variance >= 0 everywhere
- [ ] R-squared of fit > 0.5 (HAR should explain significant variance)
- [ ] Forecast on train set has same length as input

---

### Step 5: HMM Model
**What to code**: `hmm_model.py`

**Implementation order**:
1. Implement `compute_hmm_observable_features()` in `features.py` (Section 2.8)
2. Implement `fit_hmm_model()` in `hmm_model.py` (Section 2.9)
3. Implement `compute_hmm_posteriors()` in `hmm_model.py` (Section 2.9)

**Expected output**:
```python
hmm_features = compute_hmm_observable_features(df, toxicity, bvc, parkinson_bar)
# 5 columns, all finite, no NaN

hmm_model, scaler = fit_hmm_model(hmm_features)
# GMM converged, 3 components, all weights > 0.05

posteriors = compute_hmm_posteriors(hmm_features, hmm_model, scaler)
# 3 columns summing to 1.0 per row
```

**Validation checks**:
- [ ] All 5 HMM features are finite (no NaN, no Inf)
- [ ] GMM converges within max_iter
- [ ] All 3 state weights > 0.05
- [ ] Posterior probabilities sum to 1.0 per row
- [ ] All posteriors in [0, 1]

---

### Step 6: Volatility Regime Classification
**What to code**: `compute_vol_regime()` in `features.py` (Section 2.7)

**Expected output**:
```python
vol_regime.value_counts()
# 0 (low):    ~33%
# 1 (medium): ~34%
# 2 (high):   ~33%
```

**Validation checks**:
- [ ] Exactly 3 unique values: {0, 1, 2}
- [ ] Thresholds computed on training data only
- [ ] Same thresholds applied to test data

---

### Step 7: Master Feature Engineering Pipeline
**What to code**: `engineer_features()` in `features.py` (Section 2.13)

**Implementation order**:
1. Integrate all feature functions into `engineer_features()`
2. Test on training data with `fit_models=True`
3. Test on test data with `fit_models=False` using fitted models

**Expected output**:
```python
train_features, fitted = engineer_features(train_df, fit_models=True)
# train_features has all feature columns
# fitted contains har_params, hmm_model, scaler, vol_thresholds

test_features, _ = engineer_features(test_df, fit_models=False, **fitted)
# test_features has same columns, models from training
```

**Validation checks**:
- [ ] Train and test features have identical columns
- [ ] No NaN in any feature column
- [ ] HMM posteriors sum to 1 on both train and test
- [ ] Vol regime thresholds identical for train and test

---

### Step 8: Signal Generation
**What to code**: `signal_generator.py` (Section 3.7)

**Implementation order**:
1. Implement `compute_position_size()` (Section 3.6)
2. Implement `generate_signals()` (Section 3.7)
3. Test with default parameters on training data

**Expected output**:
```python
params = PARAM_DEFAULTS
signals_df = generate_signals(train_features, params)
# signals_df has columns: signal, position_size, entry_price, exit_price, trade_pnl, hold_bars
```

**Validation checks**:
- [ ] Signal values only in {-1, 0, +1}
- [ ] No consecutive same-direction signals (no pyramiding)
- [ ] Position size in [1, MAX_SIZE] for non-zero signals
- [ ] Entry price recorded for all entry signals
- [ ] Hold bars <= max_hold_bars
- [ ] No position at first feature bar (bar 0-19)

---

### Step 9: Backtesting Engine
**What to code**: `backtest.py` (Section 4.7)

**Implementation order**:
1. Implement `backtest()` function
2. Test with signals from Step 8

**Expected output**:
```python
result = backtest(signals_df)
# result = {
#     'equity_curve': pd.Series,
#     'trade_log': pd.DataFrame,
#     'daily_returns': pd.Series,
#     'bar_pnl': pd.Series,
#     'df_with_pnl': pd.DataFrame,
# }
```

**Validation checks**:
- [ ] Trade log has all required columns
- [ ] Each trade has positive hold_bars
- [ ] Entry price != exit price for all trades
- [ ] Gross P&L + costs = net P&L
- [ ] Equity curve length matches input
- [ ] Equity starts at 0

---

### Step 10: Metrics Computation
**What to code**: `metrics.py` (Section 5.4)

**Implementation order**:
1. Implement all metric functions
2. Implement statistical test functions
3. Test with backtest results from Step 9

**Expected output**:
```python
metrics = {
    'sharpe_ratio': float,
    'max_drawdown': float,
    'win_rate': float,
    'profit_factor': float,
    'trades_per_day': float,
    'avg_trade_pnl': float,
    'calmar_ratio': float,
    ...
}
```

**Validation checks**:
- [ ] Sharpe is finite (not NaN, not Inf)
- [ ] Max drawdown in [0, 1]
- [ ] Win rate in [0, 1]
- [ ] Profit factor >= 0
- [ ] Trades per day >= 0

---

### Step 11: Walk-Forward Optimization
**What to code**: `validation.py` — `run_walk_forward_optimization()` (Section 5.7)

**Implementation order**:
1. Implement walk-forward window generation
2. Implement parameter grid search over 3 parameters
3. Implement rejection of underperforming parameter combinations

**Expected output**:
```python
best_params = run_walk_forward_optimization(train_df)
# best_params = {
#     'toxic_continuation_threshold': 0.55,  # or other optimized value
#     'toxic_reversal_threshold': 0.55,      # or other optimized value
#     'max_hold_bars': 15,                   # or other optimized value
#     'regime_exit_enabled': True,
# }
```

**Validation checks**:
- [ ] All 3 optimized parameters within search space
- [ ] Best params produce >= 100 trades per walk-forward window
- [ ] Best params have win rate >= 0.50
- [ ] Walk-forward Sharpe > 0 (profitable in-sample)

---

### Step 12: Final Test Evaluation and Reporting
**What to code**: `main.py` (Section 6.11)

**Implementation order**:
1. Integrate all modules in `main.py`
2. Run full pipeline: load → split → fit → optimize → test → report
3. Generate output files and plots

**Expected output**:
```
============================================================
Strategy 1: Adaptive Flow-Toxicity with Regime-Aware Sizing
============================================================
Best parameters: {τ1: X, τ2: Y, N: Z}
Test Sharpe Ratio:   X.XXXX
Test Max Drawdown:   XX.XX%
Test Win Rate:       XX.XX%
Test Profit Factor:  X.XXX
Test Trades/Day:     X.XX
OVERALL: ALL CHECKS PASSED / SOME CHECKS FAILED
```

**Validation checks** (FINAL — must ALL pass):
- [ ] Sharpe Ratio > 1.0
- [ ] Win Rate > 54%
- [ ] Trades per Day > 5
- [ ] Max Drawdown < 15%
- [ ] Profit Factor > 1.3

---

### Build Order Summary Table

| Step | Module(s) | Input | Output | Validation Tests |
|------|-----------|-------|--------|-----------------|
| 1 | config.py, data_pipeline.py | CSV file | train_df, test_df | 5 checks |
| 2 | features.py (partial) | train_df | BVC, tick, toxicity | 5 checks |
| 3 | features.py (partial) | train_df | Parkinson RV | 5 checks |
| 4 | har_model.py | RV features | har_params | 4 checks |
| 5 | hmm_model.py, features.py | All features | HMM posteriors | 5 checks |
| 6 | features.py | HAR forecast | vol_regime | 3 checks |
| 7 | features.py | All | engineer_features() | 4 checks |
| 8 | signal_generator.py | features + params | signals_df | 6 checks |
| 9 | backtest.py | signals_df | result dict | 6 checks |
| 10 | metrics.py | result dict | metrics dict | 5 checks |
| 11 | validation.py | train_df | best_params | 4 checks |
| 12 | main.py | All | Final report | 5 checks |

---

## SECTION 8: CONSTRAINTS & GUARDRAILS

### 8.1 Hard Constraints (MUST NOT Be Violated)

#### HC-1: No Lookahead
**Constraint**: No feature, signal, or trade may use data from the future (any bar with index > current bar index).
**Rationale**: Lookahead bias is the most common and damaging error in strategy development.
**Enforcement**: 
- All rolling windows use `.shift(1)` or `min_periods` that guarantee no future data
- HMM is fit on training data only; inference uses only current and past observable features
- HAR forecast uses only past RV values (shifted by at least 1 day)
- Signal generation uses only features up to and including the current bar
- Backtest entries and exits use the close of the signal bar

**What constitutes violation**: 
- Using `close[i+1]` anywhere in feature computation
- Using `df.rolling(window=N).mean()` without ensuring the window ends at i-1
- Fitting HMM on the full dataset (including test) before signal generation
- Using tomorrow's volatility to size today's position

#### HC-2: No Test Set Information Leakage
**Constraint**: No information from the test set may influence any decision made during training or optimization.
**Rationale**: Test set must be a pure out-of-sample evaluation.
**Enforcement**:
- HAR model: fit on train only
- HMM: fit on train only  
- Vol regime thresholds: computed on train only, applied to test
- Parameter optimization: performed via walk-forward on train only
- Test set is evaluated exactly once with frozen parameters

**What constitutes violation**:
- Recomputing vol thresholds on test set
- Refitting HMM on test set
- Adjusting parameters after seeing test results
- Any form of "peeking" at test set performance

#### HC-3: Maximum 3 Optimized Parameters
**Constraint**: Exactly 3 parameters may be optimized: τ1, τ2, N.
**Rationale**: Each optimized parameter increases the risk of overfitting. Limiting to 3 provides a reasonable trade-off.
**Enforcement**: 
- Grid search over exactly {τ1, τ2, N}
- All other parameters are fixed constants in config.py
- Parameter stability check after optimization (Section 5.6.4)

**What constitutes violation**:
- Optimizing HAR coefficients (beta_d, beta_w, beta_m)
- Optimizing HMM number of components
- Optimizing position sizing parameters (ACCOUNT_VALUE, SIGMA_TARGET, MAX_SIZE)
- Optimizing slippage or commission assumptions
- Optimizing HMM_LOOKBACK window

#### HC-4: Minimum Trade Count for Valid Evaluation
**Constraint**: Any parameter combination producing fewer than 100 trades in a walk-forward window is automatically rejected.
**Rationale**: Insufficient sample size leads to unreliable performance estimates.
**Enforcement**: Hard-coded in validation.py walk-forward loop.

#### HC-5: Minimum Win Rate for Valid Evaluation
**Constraint**: Any parameter combination with win rate < 50% is rejected.
**Rationale**: A strategy that loses more than it wins must have very large winners vs losers to be profitable, which is unreliable.
**Enforcement**: Hard-coded in validation.py walk-forward loop.

#### HC-6: Position Size Limits
**Constraint**: Position size must be in [1, MAX_SIZE] contracts, where MAX_SIZE = 10.
**Rationale**: Prevents excessive risk concentration.
**Enforcement**: `np.clip(size, 1, MAX_SIZE)` in position sizing function.

#### HC-7: No Pyramiding
**Constraint**: Only one position may be held at a time. No adding to existing positions.
**Rationale**: Pyramiding increases complexity and risk.
**Enforcement**: Signal generation checks `position[i-1] == 0` before entry.

### 8.2 Soft Constraints (Recommended Values)

#### SC-1: Volatility Regime Exclusion
**Recommendation**: Do not trade when vol_regime == 2 (high volatility).
**Rationale**: High volatility regimes have unpredictable dynamics; avoiding them improves risk-adjusted returns.
**Override allowed**: Yes, but document the reason.

#### SC-2: Regime-Based Early Exit
**Recommendation**: Enable regime_exit (exit when regime posterior drops below half threshold).
**Rationale**: Reduces hold time in deteriorating conditions.
**Override allowed**: Yes, this is already a toggle parameter.

#### SC-3: HAR Non-Negative Coefficients
**Recommendation**: Clamp negative HAR coefficients to 0.001.
**Rationale**: Negative variance forecasts are nonsensical; clamping ensures non-negative forecasts.
**Override allowed**: Yes, but forecasts must remain non-negative.

#### SC-4: Walk-Forward Window Sizes
**Recommendation**: Use the specified window sizes (39K train, 7.8K test, 3.9K step).
**Rationale**: These provide ~500 days of training and ~100 days of testing per window.
**Override allowed**: Yes, if data is insufficient, reduce by 50%.

### 8.3 What Constitutes a Deviation from Spec

**Critical deviations** (must be documented and justified):
1. Changing the number of HMM states from 3
2. Adding optimized parameters beyond τ1, τ2, N
3. Using a different slippage or commission model
4. Changing the contract specifications
5. Using a different feature set (omitting BVC, toxicity, or HMM)
6. Fitting any model on the test set
7. Using intrabar execution instead of close-to-close

**Minor deviations** (acceptable with documentation):
1. Adjusting PARAM_SEARCH_SPACE granularity (e.g., step 0.03 instead of 0.05)
2. Changing random seed for HMM (HMM_RANDOM_STATE)
3. Adjusting HMM_MAX_ITER or HMM_N_INIT
4. Changing plot styles or output file formats
5. Adding logging or error handling improvements

### 8.4 Failure Handling

#### FH-1: HMM Fails to Converge
```python
if not gmm.converged_:
    print("WARNING: HMM did not converge, retrying...")
    gmm = GaussianMixture(n_components=3, covariance_type='full',
                          random_state=42, n_init=5, max_iter=500)
    gmm.fit(X_train)
    if not gmm.converged_:
        raise RuntimeError("HMM failed to converge after retry")
```

#### FH-2: Insufficient Walk-Forward Windows
If training data has fewer than 3 walk-forward windows:
- Reduce `train_window_bars` to 20,000
- Reduce `test_window_bars` to 4,000
- Reduce `step_size_bars` to 2,000

#### FH-3: All Parameter Combinations Rejected
If all 486 parameter combinations fail the minimum trade count or win rate filters:
- Reduce minimum trade count to 50
- Reduce minimum win rate to 0.45
- If still no valid combinations, report strategy failure

#### FH-4: Test Set Performance Far Below Targets
If test Sharpe < 0.5 or test win rate < 52%:
- Report results honestly with all metrics
- Note that strategy failed acceptance criteria
- Do NOT re-optimize or adjust parameters
- Document the failure for analysis

### 8.5 Code Quality Constraints

1. **Type hints**: All function parameters and returns must have type annotations
2. **Docstrings**: All public functions must have docstrings with description, parameters, and returns
3. **No global mutable state**: All state passed through function parameters
4. **Deterministic**: Same input must produce same output (fix all random seeds)
5. **No warnings**: Code must run without pandas/sklearn warnings

---

## SECTION 9: EXPECTED OUTPUTS

### 9.1 Output Files

The strategy produces the following output files in the `output/` directory:

| File | Format | Description |
|------|--------|-------------|
| `strategy_01_metrics.json` | JSON | All performance metrics, parameters, and statistical tests |
| `strategy_01_trades.csv` | CSV | Complete trade log, one row per trade |
| `strategy_01_equity.csv` | CSV | Equity curve (cumulative P&L), one row per bar |
| `strategy_01_report.txt` | Text | Human-readable summary report |

### 9.2 Output Plots

| Plot | Description | Acceptance |
|------|-------------|------------|
| `equity_curve.png` | Cumulative equity over time | Generated always |
| `drawdown.png` | Underwater curve (drawdown over time) | Generated always |
| `trade_pnl_distribution.png` | Histogram of per-trade net P&L | Generated if >= 10 trades |
| `monthly_returns.png` | Bar chart of monthly returns | Generated always |
| `regime_posteriors.png` | HMM state probabilities over time | Generated always |
| `vol_regime.png` | Volatility regime classification over time | Generated always |
| `parameter_sensitivity.png` | Heatmap of Sharpe vs τ1 and τ2 | Generated after optimization |

### 9.3 Reported Metrics

The final report MUST include all of the following metrics:

#### Strategy Configuration
- Best parameters: τ1, τ2, N
- Walk-forward Sharpe (in-sample average)
- Total optimized parameter combinations tested
- Walk-forward windows used

#### Test Set Performance
| Metric | Formula/Description | Target |
|--------|-------------------|--------|
| Sharpe Ratio | sqrt(252) * mean(daily_return) / std(daily_return) | > 1.0 |
| Max Drawdown | max(peak - equity) / peak | < 15% |
| Win Rate | count(winning trades) / total trades | > 54% |
| Profit Factor | gross profits / gross losses | > 1.3 |
| Trades Per Day | total trades / trading days | > 5 |
| Avg Trade P&L | mean(net_pnl per trade) | > $0 |
| Calmar Ratio | annualized return / max drawdown | > 2.0 |
| Total Trades | Count of completed trades | > 100 |
| Total P&L | Sum of net P&L | > $0 |
| Long Trades | Count of long positions | Report |
| Short Trades | Count of short positions | Report |
| Avg Hold Bars | Mean bars per trade | Report |
| Median Hold Bars | Median bars per trade | Report |

#### Statistical Tests
| Test | Null Hypothesis | Report |
|------|----------------|--------|
| Sharpe t-test | Sharpe <= 1.0 | t-stat, p-value, significant? |
| Win Rate binomial test | Win rate <= 54% | t-stat, p-value, significant? |

#### Cost Attribution
| Item | Value |
|------|-------|
| Total Slippage Cost | Sum of slippage across all trades |
| Total Commission | Sum of commissions across all trades |
| Cost per Trade | Average total cost per trade |
| Cost as % of Gross P&L | Total costs / |gross P&L| |

### 9.4 Acceptance Criteria (Pass/Fail)

The strategy is considered successful if ALL of the following criteria are met on the **test set**:

```python
ACCEPTANCE_CHECKS = {
    'sharpe_gt_1.0':      lambda m: m['sharpe_ratio'] > 1.0,
    'winrate_gt_54pct':   lambda m: m['win_rate'] > 0.54,
    'trades_per_day_gt_5': lambda m: m['trades_per_day'] > 5,
    'mdd_lt_15pct':       lambda m: m['max_drawdown'] < 0.15,
    'profit_factor_gt_1.3': lambda m: m['profit_factor'] > 1.3,
}
```

**Overall result**: `ALL CHECKS PASSED` if all criteria met, `SOME CHECKS FAILED` otherwise.

### 9.5 Sample Output Report

```
======================================================================
Strategy 1: Adaptive Flow-Toxicity with Regime-Aware Sizing
Execution time: 2025-01-15T10:30:00
======================================================================

BEST PARAMETERS (from walk-forward optimization):
  toxic_continuation_threshold (τ1): 0.55
  toxic_reversal_threshold (τ2):     0.55
  max_hold_bars (N):                 15
  regime_exit_enabled:               True

WALK-FORWARD RESULTS:
  Windows: 10
  Average Sharpe: 1.25

TEST SET PERFORMANCE (Out-of-Sample):
  ----------------------------------------
  Sharpe Ratio:        1.35
  Max Drawdown:        8.42%
  Win Rate:            56.3%
  Profit Factor:       1.45
  Trades Per Day:      6.2
  Avg Trade P&L:       $42.50
  Calmar Ratio:        3.20
  Total Trades:        1,488
  Total P&L:           $63,240
  ----------------------------------------

STATISTICAL TESTS:
  Sharpe > 1.0:  t=2.15, p=0.016, SIGNIFICANT
  WinRate > 54%: t=1.85, p=0.032, SIGNIFICANT

COST ATTRIBUTION:
  Total Slippage:  $18,600
  Total Commission: $3,720
  Cost/Trade:      $15.00
  Cost % of Gross: 25.8%

ACCEPTANCE CRITERIA:
  [PASS] sharpe_gt_1.0:       target=1.0,    actual=1.35
  [PASS] winrate_gt_54pct:    target=0.54,   actual=0.563
  [PASS] trades_per_day_gt_5: target=5,      actual=6.2
  [PASS] mdd_lt_15pct:        target=0.15,   actual=0.0842
  [PASS] profit_factor_gt_1.3: target=1.3,  actual=1.45

  ======================================================
  OVERALL: ALL CHECKS PASSED
  ======================================================

Results saved to output/ directory.
```

### 9.6 Failure Output Format

If the strategy fails acceptance criteria, the report MUST still be generated with honest metrics:

```
======================================================================
ACCEPTANCE CRITERIA:
  [PASS] sharpe_gt_1.0:       target=1.0,    actual=1.15
  [FAIL] winrate_gt_54pct:    target=0.54,   actual=0.512
  [PASS] trades_per_day_gt_5: target=5,      actual=7.1
  [PASS] mdd_lt_15pct:        target=0.15,   actual=0.12
  [PASS] profit_factor_gt_1.3: target=1.3,  actual=1.35

  ======================================================
  OVERALL: SOME CHECKS FAILED
  ======================================================

NOTE: Strategy failed win rate target. Review feature engineering
and signal generation logic. DO NOT re-optimize on test set.
```

---

## APPENDIX A: Quick Reference — Key Constants

```python
# Data
TOTAL_ROWS = 122296
TRAIN_ROWS = 85600
TEST_ROWS = 36695
BARS_PER_DAY = 78

# Contract
TICK_SIZE = 0.25
TICK_VALUE = 12.50
POINT_VALUE = 50.00

# Costs
SLIPPAGE_TICKS = 1
SLIPPAGE_POINTS = 0.25
COMMISSION_RT = 2.50
COST_PER_TRADE_1CT = 27.50  # $27.50 for 1 contract

# Sizing
ACCOUNT_VALUE = 100_000
SIGMA_TARGET = 0.15
MAX_SIZE = 10

# HAR
HAR_DAILY = 1    # day
HAR_WEEKLY = 5   # days
HAR_MONTHLY = 22 # days

# HMM
HMM_COMPONENTS = 3
HMM_LOOKBACK = 20

# Optimized (3 only)
# τ1 = toxic_continuation_threshold  [0.40 - 0.80]
# τ2 = toxic_reversal_threshold      [0.40 - 0.80]
# N  = max_hold_bars                 [5 - 30]
```

## APPENDIX B: Formula Cheat Sheet

| Feature | Formula |
|---------|---------|
| BVC Volume | V * (2*C - L - H) / (H - L) |
| Tick Rule | sign(C[t] - C[t-1]) |
| Toxicity | sign(BVC) * TickRule |
| Parkinson/bar | (ln(H/L))^2 / (4*ln(2)) |
| RV_daily | sum(Parkinson/bar per day) |
| HAR | c + βd*RVd + βw*RVw + βm*RVm |
| Position Size | clamp(round(15000 / (sqrt(HAR)*C*50)), 1, 10) |
| Long Signal | P(state1) > τ1 AND BVC > 0 AND vol_regime != 2 |
| Short Signal | P(state2) > τ2 AND BVC < 0 AND vol_regime != 2 |

---

*END OF SPECIFICATION*
