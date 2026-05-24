import numpy as np
from functools import lru_cache

import src.constants as C


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
    prices = np.asarray(C.r, dtype=float)
    action_binary = np.asarray(action_binary, dtype=int)
    n_products = len(prices)

    offered_indices = np.where(action_binary == 1)[0]
    probabilities = np.zeros(n_products + 1, dtype=float)

    if len(offered_indices) == 0:
        probabilities[-1] = 1.0
        return probabilities

    consideration_prob = 1.0 / (1.0 + np.exp(prices * -C.MNL_CONSIDERATION_LOGIT_SLOPE))
    k = len(offered_indices)
    q = consideration_prob[offered_indices]
    one_minus_q = 1.0 - q
    exp_utility_offered = np.exp(beta * prices[offered_indices])

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
        probability_sums = np.zeros(n_products + 1, dtype=float)

        for _ in range(n_mc_draws):
            considered_mask = rng.random(k) < q
            considered_indices = offered_indices[considered_mask]

            if len(considered_indices) == 0:
                probability_sums[-1] += 1.0
                continue

            exp_utility = np.exp(beta * prices[considered_indices])
            denominator = 1.0 + np.sum(exp_utility)

            probability_sums[considered_indices] += exp_utility / denominator
            probability_sums[-1] += 1.0 / denominator

        probabilities = probability_sums / float(n_mc_draws)
        total_prob = float(np.sum(probabilities))
        if total_prob > 0.0 and abs(total_prob - 1.0) > C.MNL_CONSIDERATION_NORMALIZATION_TOLERANCE:
            probabilities /= total_prob

    return probabilities