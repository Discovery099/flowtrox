"""Overfitting / robustness diagnostics for the strategy.

Implements:
- Probabilistic Sharpe Ratio (PSR) and Deflated Sharpe Ratio (DSR), accounting
  for non-normal returns and the number of optimization trials (Bailey & Lopez
  de Prado).
- Probability of Backtest Overfitting (PBO) via Combinatorially Symmetric
  Cross-Validation (CSCV) on the walk-forward (window x combo) Sharpe matrix.
- Bootstrap confidence interval for the test Sharpe ratio.

These are intentionally honest tools: they quantify how much of the in-sample
edge is likely to be overfitting noise.
"""

import math
from itertools import combinations

import numpy as np
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def _sharpe_daily(returns: np.ndarray) -> float:
    sd = returns.std(ddof=1)
    if sd == 0 or np.isnan(sd) or len(returns) < 2:
        return 0.0
    return float(returns.mean() / sd)


def probabilistic_sharpe_ratio(returns: np.ndarray, benchmark_sr_daily: float = 0.0) -> float:
    """P(true Sharpe > benchmark) given the sample (non-normality adjusted)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 8:
        return float("nan")
    sr = _sharpe_daily(r)
    skew = float(stats.skew(r))
    kurt = float(stats.kurtosis(r, fisher=False))  # non-excess kurtosis
    denom = 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr
    if denom <= 0:
        return float("nan")
    z = (sr - benchmark_sr_daily) * math.sqrt(n - 1) / math.sqrt(denom)
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(returns: np.ndarray, trial_sharpes_annual: np.ndarray) -> dict:
    """Deflated Sharpe Ratio: PSR against the expected-max-Sharpe under the null
    of ``N`` trials, using the cross-sectional variance of trial Sharpes.

    ``trial_sharpes_annual`` are the (annualized) Sharpe ratios of all tested
    parameter combinations.
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    ts = np.asarray(trial_sharpes_annual, dtype=float)
    ts = ts[np.isfinite(ts)]
    n_trials = len(ts)
    if len(r) < 8 or n_trials < 2:
        return {"dsr": float("nan"), "sr0_annual": float("nan"), "n_trials": int(n_trials)}

    # Convert annual trial Sharpes -> per-observation (daily) scale.
    var_daily = float(np.var(ts, ddof=1)) / 252.0
    sd_daily = math.sqrt(max(var_daily, 1e-18))

    # Expected maximum Sharpe under the null (Bailey & Lopez de Prado).
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * math.e))
    sr0_daily = sd_daily * ((1.0 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)

    dsr = probabilistic_sharpe_ratio(r, benchmark_sr_daily=sr0_daily)
    return {
        "dsr": dsr,
        "sr0_annual": float(sr0_daily * math.sqrt(252)),
        "n_trials": int(n_trials),
    }


def pbo_cscv(window_combo_matrix: list, max_splits: int = 400) -> dict:
    """Probability of Backtest Overfitting via CSCV.

    ``window_combo_matrix`` is shape (S windows, C combos) of per-window Sharpe.
    Splits the S windows into equal IS/OOS halves over all combinations; for each
    split picks the IS-best combo and records its OOS relative rank (logit).
    PBO = fraction of splits where the IS-best combo is OOS-below-median.
    """
    M = np.asarray(window_combo_matrix, dtype=float)
    if M.ndim != 2 or M.shape[0] < 4 or M.shape[1] < 4:
        return {"pbo": float("nan"), "n_splits": 0, "median_logit": float("nan")}

    S, C = M.shape
    half = S // 2
    if half == 0:
        return {"pbo": float("nan"), "n_splits": 0, "median_logit": float("nan")}

    all_idx = list(range(S))
    combos = list(combinations(all_idx, half))
    # CSCV uses complementary partitions; cap the count for runtime.
    if len(combos) > max_splits:
        rng = np.random.default_rng(42)
        sel = rng.choice(len(combos), size=max_splits, replace=False)
        combos = [combos[i] for i in sel]

    logits = []
    overfit = 0
    for is_idx in combos:
        oos_idx = [i for i in all_idx if i not in is_idx]
        is_perf = M[list(is_idx), :].mean(axis=0)
        oos_perf = M[oos_idx, :].mean(axis=0)
        best = int(np.argmax(is_perf))
        # Relative rank of the IS-best combo in the OOS distribution.
        rank = float((oos_perf < oos_perf[best]).sum() + 1) / (C + 1)
        rank = min(max(rank, 1e-6), 1 - 1e-6)
        lam = math.log(rank / (1.0 - rank))
        logits.append(lam)
        if lam <= 0:
            overfit += 1

    n = len(logits)
    return {
        "pbo": float(overfit / n) if n else float("nan"),
        "n_splits": int(n),
        "median_logit": float(np.median(logits)) if n else float("nan"),
    }


def bootstrap_sharpe_ci(returns: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    """Bootstrap CI for the annualized Sharpe + P(Sharpe <= 0)."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < 8:
        return {"sharpe_ci_low": float("nan"), "sharpe_ci_high": float("nan"),
                "bootstrap_p_value": float("nan")}
    rng = np.random.default_rng(seed)
    ann = math.sqrt(252)
    sims = np.empty(n_boot)
    for i in range(n_boot):
        sample = r[rng.integers(0, n, n)]
        sd = sample.std(ddof=1)
        sims[i] = (sample.mean() / sd * ann) if sd > 0 else 0.0
    return {
        "sharpe_ci_low": float(np.percentile(sims, 5)),
        "sharpe_ci_high": float(np.percentile(sims, 95)),
        "bootstrap_p_value": float(np.mean(sims <= 0.0)),  # P(no edge)
    }


def compute_diagnostics(daily_returns, trial_sharpes_annual=None, window_combo_matrix=None) -> dict:
    """Bundle all robustness diagnostics into one JSON-friendly dict."""
    r = np.asarray(daily_returns, dtype=float)
    out = {}
    out["psr"] = probabilistic_sharpe_ratio(r, 0.0)
    out.update(bootstrap_sharpe_ci(r))
    if trial_sharpes_annual is not None:
        out.update(deflated_sharpe_ratio(r, trial_sharpes_annual))
    if window_combo_matrix is not None:
        out.update(pbo_cscv(window_combo_matrix))

    def _clean(x):
        if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
            return None
        return x
    return {k: _clean(v) for k, v in out.items()}
