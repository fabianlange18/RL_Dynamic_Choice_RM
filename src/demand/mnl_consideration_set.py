import numpy as np
from functools import lru_cache

import src.constants as C


# Static arrays derived from constants to avoid recomputing them on every call.
_PRICES = np.asarray(C.r, dtype=float)
_N_PRODUCTS = len(_PRICES)
_CONSIDERATION_PROB = 1.0 / (1.0 + np.exp(_PRICES * -C.MNL_CONSIDERATION_LOGIT_SLOPE))


@lru_cache(maxsize=C.MNL_CONSIDERATION_SUBSET_CACHE_SIZE)
def subset_membership_matrix(k):
    """Return a boolean matrix with all subset memberships for k items."""
    masks = np.arange(1 << k, dtype=np.uint32)
    bits = np.arange(k, dtype=np.uint32)
    return ((masks[:, None] >> bits[None, :]) & 1).astype(bool)


def mnl_consideration_set_probabilities(action_binary, beta=None, n_draws=None, seed=None):
    """Compute MNL probabilities with latent consideration sets.

    Applied formulation:
        Z_i ~ Bernoulli(q_i), independently for i in A,
        q_i = 1 / (1 + exp(-0.0125 * r_i)),
        C = {i in A : Z_i = 1},
        P(i | A) = sum_{C subseteq A} P(C | A) * P_MNL(i | C).

    For large offer sets, a simple Monte Carlo estimator samples latent
    consideration sets directly and averages the resulting MNL probabilities.
    """
    action_binary = np.asarray(action_binary, dtype=int)

    offered_indices = np.where(action_binary == 1)[0]
    probabilities = np.zeros(_N_PRODUCTS + 1, dtype=float)

    if len(offered_indices) == 0:
        probabilities[-1] = 1.0
        return probabilities

    k = len(offered_indices)
    q = _CONSIDERATION_PROB[offered_indices]
    one_minus_q = 1.0 - q
    exp_utility_offered = np.exp(beta * _PRICES[offered_indices])

    if k <= C.MNL_CONSIDERATION_EXACT_MAX_PRODUCTS:
        subset_matrix = subset_membership_matrix(k)
        subset_matrix_float = subset_matrix.astype(float)

        set_probabilities = np.prod(
            np.where(subset_matrix, q[None, :], one_minus_q[None, :]),
            axis=1,
        )

        denominators = 1.0 + subset_matrix_float.dot(exp_utility_offered)
        weights = set_probabilities / denominators

        offered_probabilities = exp_utility_offered * subset_matrix_float.T.dot(weights)
        probabilities[offered_indices] = offered_probabilities
        probabilities[-1] = float(np.sum(weights))
    else:
        n_mc_draws = int(C.MNL_CONSIDERATION_MONTE_CARLO_DRAWS if n_draws is None else n_draws)
        rng = np.random.default_rng() if seed is None else np.random.default_rng(seed)

        # Vectorized Monte Carlo over latent consideration draws.
        considered_matrix = (rng.random((n_mc_draws, k)) < q[None, :]).astype(float)
        weighted_utilities = considered_matrix * exp_utility_offered[None, :]
        denominators = 1.0 + np.sum(weighted_utilities, axis=1)
        draw_probabilities = weighted_utilities / denominators[:, None]

        probabilities[offered_indices] = np.mean(draw_probabilities, axis=0)
        probabilities[-1] = float(np.mean(1.0 / denominators))

        total_prob = float(np.sum(probabilities))
        if total_prob > 0.0 and abs(total_prob - 1.0) > C.MNL_CONSIDERATION_NORMALIZATION_TOLERANCE:
            probabilities /= total_prob

    return probabilities