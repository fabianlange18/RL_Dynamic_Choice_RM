import numpy as np

from src.demand.mnl import mnl_probabilities

def mmnl_probabilities(action_binary, segment_betas, segment_weights=None):
    """Compute MMNL probabilities as a finite mixture of MNL models.

    Applied formulation:
        P(i | A) = sum_{s=1}^S w_s P_MNL(i | A, beta_s),
        beta_s in {0.6, 0.8, 1.0, 1.2, 1.4} * beta by default,
        sum_s w_s = 1, w_s >= 0.
    """

    segment_betas = np.asarray(segment_betas, dtype=float)

    if segment_weights is None:
        segment_weights = np.full(segment_betas.size, 1.0 / segment_betas.size, dtype=float)
    else:
        segment_weights = np.asarray(segment_weights, dtype=float)
        if segment_weights.shape != segment_betas.shape:
            raise ValueError("segment_weights must have the same shape as segment_betas")
        if np.any(segment_weights < 0):
            raise ValueError("segment_weights must be non-negative")
        weight_sum = float(np.sum(segment_weights))
        if weight_sum <= 0:
            raise ValueError("segment_weights must sum to a positive value")
        segment_weights = segment_weights / weight_sum

    component_probabilities = np.array(
        [mnl_probabilities(action_binary, b) for b in segment_betas],
        dtype=float,
    )
    return np.average(component_probabilities, axis=0, weights=segment_weights)