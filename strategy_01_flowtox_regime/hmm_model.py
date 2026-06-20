"""Regime model fitting and inference (Spec Section 2.9).

Two backends are supported:

- "gmm": a 3-state Gaussian Mixture (the spec's HMM *proxy*). Per-bar
  responsibilities are independent across bars (no temporal dependence).
- "hmm": a real 3-state Gaussian Hidden Markov Model (``hmmlearn``) with a
  learned transition matrix. To remain valid for trading we compute **filtered
  (online) forward posteriors** P(state_t | x_1..t) — NOT hmmlearn's
  ``predict_proba`` smoothed posteriors, which peek at future bars within the
  sequence and would introduce look-ahead bias.

After fitting, components are deterministically relabelled to semantic states
{0: Normal, 1: Toxic-Continuation, 2: Toxic-Reversal}.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import (
    HMM_N_COMPONENTS,
    HMM_COVARIANCE_TYPE,
    HMM_RANDOM_STATE,
    HMM_N_INIT,
    HMM_MAX_ITER,
)

# Observable feature column order (see features.compute_hmm_observable_features):
#   0 toxicity rate, 1 BVC sign, 2 log Parkinson var, 3 momentum, 4 vol intensity


def _derive_label_map(means: np.ndarray) -> list:
    """Map component index -> semantic state via standardized component means.

    Returns ``label_map`` with label_map[k] = component index for semantic state
    k (0=Normal, 1=Continuation, 2=Reversal).
    """
    tox = means[:, 0]   # toxicity rate
    vol = means[:, 2]   # (log) Parkinson variance
    comps = list(range(means.shape[0]))

    normal = int(np.argmin(tox + vol))
    remaining = [c for c in comps if c != normal]
    reversal = int(remaining[int(np.argmax(vol[remaining]))])
    continuation = [c for c in remaining if c != reversal][0]
    return [normal, continuation, reversal]


def fit_hmm_model(
    obs_features_train: pd.DataFrame,
    random_state: int = HMM_RANDOM_STATE,
    n_init: int = HMM_N_INIT,
    model_type: str = "gmm",
) -> Tuple[object, StandardScaler, list]:
    """Fit the regime model on training observable features (Spec 2.9).

    Returns (fitted_model, fitted_scaler, label_map). ``fitted_model`` is either
    a GaussianMixture ("gmm") or a GaussianHMM ("hmm").
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(obs_features_train.values)

    if model_type == "hmm":
        from hmmlearn.hmm import GaussianHMM

        model = GaussianHMM(
            n_components=HMM_N_COMPONENTS,
            covariance_type=HMM_COVARIANCE_TYPE,
            n_iter=max(HMM_MAX_ITER, 200),
            tol=1e-4,
            random_state=random_state,
            init_params="stmc",
        )
        model.fit(X_train)
        label_map = _derive_label_map(np.asarray(model.means_))
        return model, scaler, label_map

    # Default: Gaussian Mixture proxy.
    gmm = GaussianMixture(
        n_components=HMM_N_COMPONENTS,
        covariance_type=HMM_COVARIANCE_TYPE,
        random_state=random_state,
        n_init=n_init,
        max_iter=HMM_MAX_ITER,
        tol=1e-4,
    )
    gmm.fit(X_train)
    if not gmm.converged_:
        gmm = GaussianMixture(
            n_components=HMM_N_COMPONENTS,
            covariance_type=HMM_COVARIANCE_TYPE,
            random_state=random_state,
            n_init=max(n_init, 5),
            max_iter=500,
            tol=1e-4,
        )
        gmm.fit(X_train)
        if not gmm.converged_:
            raise RuntimeError("GMM failed to converge after retry")

    label_map = _derive_label_map(gmm.means_)
    return gmm, scaler, label_map


def _emission_logprob(model, X: np.ndarray) -> np.ndarray:
    """(n, K) Gaussian emission log-likelihoods for a fitted HMM."""
    K = model.n_components
    means = np.asarray(model.means_)
    covars = model.covars_  # (K, d, d) for 'full'
    out = np.empty((X.shape[0], K))
    for k in range(K):
        cov = covars[k]
        cov = cov + np.eye(cov.shape[0]) * 1e-6  # numerical floor
        out[:, k] = multivariate_normal.logpdf(X, mean=means[k], cov=cov, allow_singular=True)
    return out


def _filtered_posteriors(model, X: np.ndarray) -> np.ndarray:
    """Online FILTERED posteriors P(state_t | x_1..t) via the forward algorithm.

    No look-ahead: each bar uses only observations up to and including itself.
    """
    n = X.shape[0]
    K = model.n_components
    log_start = np.log(np.asarray(model.startprob_) + 1e-300)
    log_A = np.log(np.asarray(model.transmat_) + 1e-300)
    emis = _emission_logprob(model, X)

    filt = np.empty((n, K))
    # t = 0
    la = log_start + emis[0]
    la -= logsumexp(la)
    filt[0] = np.exp(la)
    # t >= 1: predict then update, renormalising each step (filtered, not smoothed)
    for t in range(1, n):
        # predicted log prob for each state: logsumexp_j(filt_{t-1}(j) * A[j,k])
        pred = logsumexp(la[:, None] + log_A, axis=0)
        la = pred + emis[t]
        la -= logsumexp(la)
        filt[t] = np.exp(la)
    return filt


def compute_hmm_posteriors(
    obs_features: pd.DataFrame,
    hmm_model,
    scaler: StandardScaler,
    label_map: list,
    model_type: str = "gmm",
) -> pd.DataFrame:
    """Compute regime posteriors per bar, relabelled to semantic states.

    Columns: hmm_state_0_posterior (Normal), _1_ (Continuation), _2_ (Reversal);
    each row sums to 1.0.
    """
    X = scaler.transform(obs_features.values)

    if model_type == "hmm":
        posteriors = _filtered_posteriors(hmm_model, X)
    else:
        posteriors = hmm_model.predict_proba(X)

    ordered = posteriors[:, label_map]
    result = pd.DataFrame(
        ordered,
        columns=[
            "hmm_state_0_posterior",
            "hmm_state_1_posterior",
            "hmm_state_2_posterior",
        ],
        index=obs_features.index,
    )
    return result.astype(np.float64)
