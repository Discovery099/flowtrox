"""Validate multi-instrument support after refactor (date-based RV + per-instrument costs)."""
import sys, time, warnings, traceback
warnings.filterwarnings("ignore")
sys.path.insert(0, "/app")
from strategy_01_flowtox_regime.pipeline import load_instrument_data, _fit_train_models, _evaluate_on_test, costs_from_inst
from strategy_01_flowtox_regime.config import INSTRUMENTS, PARAM_DEFAULTS

def run(symbol):
    t0 = time.time()
    train_df, test_df, inst = load_instrument_data(symbol)
    costs = costs_from_inst(inst)
    fitted = _fit_train_models(train_df, hmm_n_init=3)
    evald = _evaluate_on_test(test_df, fitted, dict(PARAM_DEFAULTS), costs=costs)
    m = evald["metrics"]
    har = fitted["har_params"]
    dt = time.time() - t0
    print(f"{symbol:4s} | train={len(train_df):6d} test={len(test_df):6d} "
          f"| tick={inst['contract_specs']['tick_size']} pv={inst['contract_specs']['point_value']} "
          f"slip_pts={costs['slippage_points']:.3f} "
          f"| HAR R2={har['r_squared']:.3f} "
          f"| trades={m['total_trades']:5d} sharpe={m['sharpe_ratio']:.3f} "
          f"pnl=${m['total_pnl']:,.0f} | {dt:.0f}s")
    assert m["total_trades"] > 0, f"{symbol}: no trades"
    assert -1e-9 <= har["r_squared"] <= 1.0001, f"{symbol}: bad R2"

if __name__ == "__main__":
    print("Instruments registered:", list(INSTRUMENTS.keys()))
    for s in ["ES", "MES", "MNQ", "M2K", "MGC", "MCL"]:
        try:
            run(s)
        except Exception:
            print(f"{s}: FAILED")
            traceback.print_exc()
    print("DONE")
