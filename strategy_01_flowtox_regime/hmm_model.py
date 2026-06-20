"""HMM/GMM model fitting and inference (Spec Section 2.9).

A 3-state Gaussian Mixture Model is used as a proxy for the HMM emission
distributions. After fitting, GMM components are deterministically relabelled
to the semantic states {0: Normal, 1: Toxic-Continuation, 2: Toxic-Reversal}
so that downstream signal logic is stable across train/test.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import (
    HMM_N_COMPONENTS,
    HMM_COVARIANCE_TYPE,
    HMM_RANDOM_STATE,
    HMM_N_INIT,
    HMM_MAX_ITER,
)

# Observable feature column order (must match features.compute_hmm_observable_features):
#   idx 0 = hmm_feat_1 = rolling toxicity rate
#   idx 1 = hmm_feat_2 = rolling BVC sign
#   idx 2 = hmm_feat_3 = rolling (log) Parkinson variance
#   idx 3 = hmm_feat_4 = 5-bar price momentum
#   idx 4 = hmm_feat_5 = volume intensity


def _derive_label_map(gmm: GaussianMixture) -> list:
    """Map GMM component index -> semantic state index using component means.

    Returns ``label_map`` such that ``label_map[k]`` is the GMM component index
    that represents semantic state ``k`` (0=Normal, 1=Continuation, 2=Reversal).
    Means are in standardized space; relative ordering per feature is preserved.
    """
    means = gmm.means_  # shape (3, 5)
    tox = means[:, 0]   # toxicity rate
    vol = means[:, 2]   # (log) Parkinson variance

    comps = list(range(means.shape[0]))

    # Normal = lowest combined toxicity + volatility.
    normal = int(np.argmin(tox + vol))

    remaining = [c for c in comps if c != normal]
    # Reversal = highest volatility among the remaining components.
    reversal = int(remaining[int(np.argmax(vol[remaining]))])

    continuation = [c for c in remaining if c != reversal][0]

    return [normal, continuation, reversal]


def fit_hmm_model(
    obs_features_train: pd.DataFrame,
    random_state: int = HMM_RANDOM_STATE,
    n_init: int = HMM_N_INIT,
) -> Tuple[GaussianMixture, StandardScaler, list]:
    """Fit a 3-state GMM on training observable features (Spec 2.9).

    Returns (fitted_gmm, fitted_scaler, label_map).
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(obs_features_train.values)

    gmm = GaussianMixture(
        n_components=HMM_N_COMPONENTS,
        covariance_type=HMM_COVARIANCE_TYPE,
        random_state=random_state,
        n_init=n_init,
        max_iter=HMM_MAX_ITER,
        tol=1e-4,
    )
    gmm.fit(X_train)

    # Failure handling FH-1: retry if not converged.
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
            raise RuntimeError("HMM failed to converge after retry")

    label_map = _derive_label_map(gmm)
    return gmm, scaler, label_map


def compute_hmm_posteriors(
    obs_features: pd.DataFrame,
    hmm_model: GaussianMixture,
    scaler: StandardScaler,
    label_map: list,
) -> pd.DataFrame:
    """Compute HMM state posterior probabilities for each bar (Spec 2.9).

    Columns are relabelled via ``label_map`` so that:
        hmm_state_0_posterior = Normal
        hmm_state_1_posterior = Toxic-Continuation
        hmm_state_2_posterior = Toxic-Reversal
    Each row sums to 1.0.
    """
    X = scaler.transform(obs_features.values)
    posteriors = hmm_model.predict_proba(X)  # (n, 3) in GMM component order

    # Reorder columns into semantic-state order.
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
