#!/usr/bin/env python3
"""CLI entry point for FLOWTOX_REGIME_01 (Spec Section 6.11).

Runs the full pipeline (load -> walk-forward optimize -> test evaluation),
prints a human-readable report, and writes output files.

Usage: python -m strategy_01_flowtox_regime.main
"""

import json
import os
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")

from .config import ACCEPTANCE_CRITERIA
from .pipeline import run_full_pipeline


def _progress(ev: dict) -> None:
    print(f"  [{ev.get('pct', 0):3d}%] {ev.get('message', '')}")


def main(symbol: str = "ES", out_dir: str = "output"):
    print("=" * 70)
    print("Strategy 1: Adaptive Flow-Toxicity with Regime-Aware Sizing")
    print(f"Execution time: {datetime.now().isoformat()}")
    print("=" * 70)

    results = run_full_pipeline(symbol, progress_cb=_progress)
    metrics = results["metrics"]
    checks = results["checks"]
    bp = results["best_params"]

    print("\nBEST PARAMETERS (walk-forward):")
    print(f"  tau1 = {bp['toxic_continuation_threshold']}")
    print(f"  tau2 = {bp['toxic_reversal_threshold']}")
    print(f"  N    = {bp['max_hold_bars']}")
    print(f"  walk-forward Sharpe: {results['walk_forward']['best_score']:.4f}")

    print("\nTEST SET PERFORMANCE (out-of-sample):")
    print(f"  Sharpe Ratio:   {metrics['sharpe_ratio']:.4f}")
    print(f"  Max Drawdown:   {metrics['max_drawdown']:.4%}")
    print(f"  Win Rate:       {metrics['win_rate']:.4%}")
    print(f"  Profit Factor:  {metrics['profit_factor']:.4f}")
    print(f"  Trades/Day:     {metrics['trades_per_day']:.2f}")
    print(f"  Avg Trade P&L:  ${metrics['avg_trade_pnl']:.2f}")
    print(f"  Total Trades:   {metrics['total_trades']}")
    print(f"  Total P&L:      ${metrics['total_pnl']:,.2f}")

    print("\nACCEPTANCE CRITERIA:")
    for k, v in checks.items():
        if k == "all_passed":
            continue
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\nOVERALL: {'ALL CHECKS PASSED' if checks['all_passed'] else 'SOME CHECKS FAILED'}")

    _save_results(results, out_dir)
    return results


def _save_results(results: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    metrics_serializable = {
        k: v for k, v in results["metrics"].items()
        if isinstance(v, (int, float, bool, str))
    }
    metrics_serializable["best_params"] = results["best_params"]
    with open(os.path.join(out_dir, "strategy_01_metrics.json"), "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    results["result"]["trade_log"].to_csv(
        os.path.join(out_dir, "strategy_01_trades.csv"), index=False
    )
    results["result"]["equity_curve"].to_csv(
        os.path.join(out_dir, "strategy_01_equity.csv")
    )
    print(f"\nResults saved to {out_dir}/ directory.")


if __name__ == "__main__":
    main()
