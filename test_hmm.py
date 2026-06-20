"""Validate real-HMM path (filtered, no-lookahead posteriors) vs GMM."""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/app")
import numpy as np
from strategy_01_flowtox_regime.pipeline import load_instrument_data, _fit_train_models, _evaluate_on_test, costs_from_inst
from strategy_01_flowtox_regime.config import PARAM_DEFAULTS

train_df, test_df, inst = load_instrument_data("ES")
costs = costs_from_inst(inst)

for rm in ["gmm", "hmm"]:
    t0 = time.time()
    fitted = _fit_train_models(train_df, hmm_n_init=3, regime_model=rm)
    ev = _evaluate_on_test(test_df, fitted, dict(PARAM_DEFAULTS), costs=costs)
    feat = ev["features"]
    p = feat[["hmm_state_0_posterior","hmm_state_1_posterior","hmm_state_2_posterior"]].values
    psum = p.sum(axis=1)
    m = ev["metrics"]
    extra = ""
    if hasattr(fitted["hmm_model"], "transmat_"):
        A = np.asarray(fitted["hmm_model"].transmat_)
        extra = f"diag(A)={np.round(np.diag(A),3).tolist()} (persistence)"
    print(f"[{rm}] {time.time()-t0:.0f}s | posteriors sum to 1: {np.allclose(psum,1.0,atol=1e-6)} "
          f"| trades={m['total_trades']} sharpe={m['sharpe_ratio']:.3f} winrate={m['win_rate']:.3f} {extra}")

print("DONE")
