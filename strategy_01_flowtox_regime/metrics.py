"""Performance metrics and statistical tests (Spec Section 5.4 / 5.5)."""

import numpy as np
import pandas as pd
from scipy import stats


def compute_sharpe(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualized Sharpe ratio (Spec 5.4.1)."""
    if len(daily_returns) == 0:
        return 0.0
    excess = daily_returns - risk_free_rate / 252
    sd = excess.std()
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.sqrt(252) * excess.mean() / sd)


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown as a fraction of peak equity (Spec 5.4.2).

    The raw equity curve is cumulative P&L starting near 0, which makes a naive
    peak-relative drawdown unstable. We anchor the curve to the reference
    account value so drawdown is expressed as a fraction of account equity.
    """
    if len(equity_curve) == 0:
        return 0.0
    equity = equity_curve.astype(float) + 100_000.0  # anchor to account value
    running_peak = equity.cummax()
    drawdown = (running_peak - equity) / running_peak
    mdd = drawdown.max()
    return float(mdd) if not np.isnan(mdd) else 0.0


def compute_win_rate(trade_log: pd.DataFrame) -> float:
    """Fraction of trades with positive net P&L (Spec 5.4.3)."""
    if trade_log is None or len(trade_log) == 0:
        return 0.0
    wins = (trade_log["net_pnl"] > 0).sum()
    return float(wins / len(trade_log))


def compute_profit_factor(trade_log: pd.DataFrame) -> float:
    """Gross profits / gross losses (Spec 5.4.4)."""
    if trade_log is None or len(trade_log) == 0:
        return 0.0
    gross_profits = trade_log["net_pnl"][trade_log["net_pnl"] > 0].sum()
    gross_losses = abs(trade_log["net_pnl"][trade_log["net_pnl"] < 0].sum())
    if gross_losses == 0:
        return float("inf") if gross_profits > 0 else 0.0
    return float(gross_profits / gross_losses)


def compute_trades_per_day(trade_log: pd.DataFrame, num_trading_days: int) -> float:
    """Average number of trades per trading day (Spec 5.4.5)."""
    if num_trading_days == 0 or trade_log is None:
        return 0.0
    return float(len(trade_log) / num_trading_days)


def compute_avg_trade_pnl(trade_log: pd.DataFrame) -> float:
    """Average net P&L per trade in USD (Spec 5.4.6)."""
    if trade_log is None or len(trade_log) == 0:
        return 0.0
    return float(trade_log["net_pnl"].mean())


def compute_calmar(daily_returns: pd.Series, max_dd: float) -> float:
    """Calmar ratio = annualized return / max drawdown (Spec 5.4.7)."""
    if max_dd == 0 or len(daily_returns) == 0:
        return 0.0
    ann_return = daily_returns.mean() * 252
    return float(ann_return / max_dd)


def sharpe_significance_test(daily_returns: pd.Series, target_sharpe: float = 1.0) -> tuple:
    """Test whether realized Sharpe is significantly > target (Spec 5.5.1)."""
    T = len(daily_returns)
    if T < 30 or daily_returns.std() == 0:
        return 0.0, 1.0, False
    realized = compute_sharpe(daily_returns)
    se = 1.0 / np.sqrt(T)
    t_stat = (realized - target_sharpe) / se
    p_value = 1 - stats.norm.cdf(t_stat)
    return float(t_stat), float(p_value), bool(p_value < 0.05)


def win_rate_significance_test(trade_log: pd.DataFrame, target_rate: float = 0.54) -> tuple:
    """Test whether win rate is significantly > target via binomial (Spec 5.5.2)."""
    if trade_log is None:
        return 0.0, 1.0, False
    n = len(trade_log)
    if n < 30:
        return 0.0, 1.0, False
    wins = int((trade_log["net_pnl"] > 0).sum())
    res = stats.binomtest(wins, n, target_rate, alternative="greater")
    p_value = float(res.pvalue)
    se = np.sqrt(target_rate * (1 - target_rate) / n)
    t_stat = (wins / n - target_rate) / se if se > 0 else 0.0
    return float(t_stat), p_value, bool(p_value < 0.05)


def compute_all_metrics(test_result: dict, num_test_days: int) -> dict:
    """Compute the full metric suite + statistical tests for a backtest result."""
    tl = test_result["trade_log"]
    dr = test_result["daily_returns"]
    eq = test_result["equity_curve"]

    metrics = {}
    metrics["sharpe_ratio"] = compute_sharpe(dr)
    metrics["max_drawdown"] = compute_max_drawdown(eq)
    metrics["win_rate"] = compute_win_rate(tl)
    metrics["profit_factor"] = compute_profit_factor(tl)
    metrics["trades_per_day"] = compute_trades_per_day(tl, num_test_days)
    metrics["avg_trade_pnl"] = compute_avg_trade_pnl(tl)
    metrics["calmar_ratio"] = compute_calmar(dr, metrics["max_drawdown"])
    metrics["total_trades"] = int(len(tl)) if tl is not None else 0
    metrics["total_pnl"] = float(tl["net_pnl"].sum()) if tl is not None and len(tl) > 0 else 0.0

    if tl is not None and len(tl) > 0:
        metrics["long_trades"] = int((tl["direction"] == "LONG").sum())
        metrics["short_trades"] = int((tl["direction"] == "SHORT").sum())
        metrics["avg_hold_bars"] = float(tl["hold_bars"].mean())
        metrics["median_hold_bars"] = float(tl["hold_bars"].median())
        metrics["total_slippage_cost"] = float(tl["slippage_cost"].sum())
        metrics["total_commission"] = float(tl["commission"].sum())
        metrics["cost_per_trade"] = float((tl["slippage_cost"] + tl["commission"]).mean())
        gross = float(tl["gross_pnl"].abs().sum())
        total_cost = float(tl["slippage_cost"].sum() + tl["commission"].sum())
        metrics["cost_pct_of_gross"] = float(total_cost / gross) if gross > 0 else 0.0
    else:
        for k in ["long_trades", "short_trades", "avg_hold_bars", "median_hold_bars",
                  "total_slippage_cost", "total_commission", "cost_per_trade", "cost_pct_of_gross"]:
            metrics[k] = 0.0

    t_s, p_s, sig_s = sharpe_significance_test(dr)
    metrics["sharpe_t_stat"] = t_s
    metrics["sharpe_p_value"] = p_s
    metrics["sharpe_significant_5pct"] = sig_s

    t_w, p_w, sig_w = win_rate_significance_test(tl)
    metrics["winrate_t_stat"] = t_w
    metrics["winrate_p_value"] = p_w
    metrics["winrate_significant_5pct"] = sig_w

    return metrics


def check_acceptance(metrics: dict) -> dict:
    """Evaluate the five hard acceptance criteria (Spec 9.4)."""
    checks = {
        "sharpe_gt_1.0": metrics["sharpe_ratio"] > 1.0,
        "winrate_gt_54pct": metrics["win_rate"] > 0.54,
        "trades_per_day_gt_5": metrics["trades_per_day"] > 5,
        "mdd_lt_15pct": metrics["max_drawdown"] < 0.15,
        "profit_factor_gt_1.3": metrics["profit_factor"] > 1.3,
    }
    checks["all_passed"] = all(checks.values())
    return checks
