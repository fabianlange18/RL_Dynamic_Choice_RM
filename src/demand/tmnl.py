import numpy as np

import src.constants as C


def tmnl_probabilities(action_binary, beta, delta=C.TMNL_DEFAULT_DELTA):
    """Compute threshold-MNL probabilities.

    Applied formulation:
        u_i = beta * r_i,  u_0 = -1,
        Psi(A; delta) = {i in A : u_i >= max(u_0, max_{j in A} u_j) - delta},
        P(i | A) = exp(u_i) / sum_{j in Psi(A; delta) union {0 if considered}} exp(u_j)
    over the thresholded consideration set.
    """
    prices = np.asarray(C.r, dtype=float)
    action_binary = np.asarray(action_binary, dtype=bool)

    offered_indices = np.where(action_binary)[0]
    probabilities = np.zeros(len(prices) + 1, dtype=float)

    if len(offered_indices) == 0:
        probabilities[-1] = 1.0
        return probabilities

    u_products = beta * prices
    u_outside = C.TMNL_OUTSIDE_UTILITY
    u_max = max(float(np.max(u_products[offered_indices])), u_outside)
    threshold = u_max - delta

    in_consideration = action_binary & (u_products >= threshold)
    outside_considered = u_outside >= threshold

    exp_u_considered = np.where(in_consideration, np.exp(u_products), 0.0)
    denominator = np.sum(exp_u_considered)
    if outside_considered:
        denominator += np.exp(u_outside)

    if denominator == 0.0:
        probabilities[-1] = 1.0
        return probabilities

    probabilities[:-1] = exp_u_considered / denominator
    probabilities[-1] = np.exp(u_outside) / denominator if outside_considered else 0.0
    return probabilities