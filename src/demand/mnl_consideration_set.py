import numpy as np
from functools import lru_cache

import src.constants as C

PRICES = np.asarray(C.r, dtype=float)
CONSIDERATION_PROB = 1.0 / (1.0 + np.exp(PRICES * -C.MNL_CONSIDERATION_LOGIT_SLOPE))
ONE_MINUS_CONSIDERATION_PROB = 1.0 - CONSIDERATION_PROB


@lru_cache(maxsize=C.MNL_CONSIDERATION_SUBSET_CACHE_SIZE)
def subset_membership_matrix(k):
    """Return a boolean matrix with all subset memberships for k items."""
    masks = np.arange(1 << k, dtype=np.uint32)
    bits = np.arange(k, dtype=np.uint32)
    return ((masks[:, None] >> bits[None, :]) & 1).astype(bool)


@lru_cache(maxsize=C.MNL_CONSIDERATION_QUADRATURE_CACHE_SIZE)
def unit_interval_legendre_quadrature(n_points):
    """Return Gauss-Legendre nodes/weights mapped from [-1,1] to [0,1]."""
    x, w = np.polynomial.legendre.leggauss(int(n_points))
    return 0.5 * (x + 1.0), 0.5 * w


@lru_cache(maxsize=C.MNL_CONSIDERATION_QUADRATURE_CACHE_SIZE)
def _exp_utility(beta):
    return np.exp(beta * PRICES)


@lru_cache(maxsize=C.MNL_CONSIDERATION_QUADRATURE_CACHE_SIZE)
def _mnl_consideration_set_probabilities(action_tuple, beta):
    action_binary = np.asarray(action_tuple, dtype=np.uint8)
    n_products = len(PRICES)

    offered_indices = np.where(action_binary == 1)[0]
    probabilities = np.zeros(n_products + 1, dtype=float)

    if len(offered_indices) == 0:
        probabilities[-1] = 1.0
        return probabilities

    k = len(offered_indices)
    q = CONSIDERATION_PROB[offered_indices]
    one_minus_q = ONE_MINUS_CONSIDERATION_PROB[offered_indices]
    exp_utility_offered = _exp_utility(beta)[offered_indices]

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
        t_nodes, t_weights = unit_interval_legendre_quadrature(C.MNL_CONSIDERATION_QUADRATURE_POINTS)
        t_pow_v = t_nodes[:, None] ** exp_utility_offered[None, :]
        factors = one_minus_q[None, :] + q[None, :] * t_pow_v
        prod_all = np.prod(factors, axis=1)

        outside_prob = float(np.dot(t_weights, prod_all))
        prod_excluding_i = prod_all[:, None] / factors
        offered_integrand = (
            q[None, :]
            * exp_utility_offered[None, :]
            * t_pow_v
            * prod_excluding_i
        )
        offered_probabilities = np.dot(t_weights, offered_integrand)

        probabilities[offered_indices] = offered_probabilities
        probabilities[-1] = outside_prob

        total_prob = float(np.sum(probabilities))
        if total_prob > 0.0 and abs(total_prob - 1.0) > C.MNL_CONSIDERATION_NORMALIZATION_TOLERANCE:
            probabilities /= total_prob

    return probabilities


def mnl_consideration_set_probabilities(action_binary, beta=None):
    """Compute MNL probabilities with latent consideration sets.

    Applied formulation:
        Z_i ~ Bernoulli(q_i), independently for i in A,
        q_i = 1 / (1 + exp(-0.0125 * r_i)),
        C = {i in A : Z_i = 1},
        P(i | A) = sum_{C subseteq A} P(C | A) * P_MNL(i | C).

    For large offer sets, the equivalent integral
        1 / (1 + s) = int_0^1 t^s dt
    is used to avoid explicit subset enumeration.
    """
    action_tuple = tuple(np.asarray(action_binary, dtype=np.uint8).tolist())
    return _mnl_consideration_set_probabilities(action_tuple, float(beta))
