import numpy as np

import src.constants as C


def nested_logit_probabilities(action_binary, beta=None):
    """Compute nested-logit probabilities with two nests.

    Applied formulation:
        D_m = sum_{i in A cap m} exp((beta * r_i) / mu_m),
        G_m = D_m^{mu_m},
        P(m | A) = G_m / (1 + G_A + G_B),
        P(i | m, A) = exp((beta * r_i) / mu_m) / D_m,
        P(i | A) = P(m | A) * P(i | m, A),
        P(0 | A) = 1 / (1 + G_A + G_B).
    """
    prices = np.asarray(C.r, dtype=float)
    action_binary = np.asarray(action_binary, dtype=int)
    n_products = len(prices)

    nest_a = np.arange(n_products // 2)
    nest_b = np.arange(n_products // 2, n_products)
    mu_a = C.NESTED_LOGIT_MU_A
    mu_b = C.NESTED_LOGIT_MU_B

    def _nest_terms(indices, mu):
        active = indices[action_binary[indices] == 1]
        if len(active) == 0:
            return 0.0, active, np.array([], dtype=float)

        scaled_utilities = np.exp((beta * prices[active]) / mu)
        return float(np.sum(scaled_utilities)), active, scaled_utilities

    d_a, active_a, scaled_a = _nest_terms(nest_a, mu_a)
    d_b, active_b, scaled_b = _nest_terms(nest_b, mu_b)

    g_a = d_a ** mu_a
    g_b = d_b ** mu_b
    denominator = 1.0 + g_a + g_b

    probabilities = np.zeros(n_products + 1, dtype=float)

    if d_a > 0:
        p_a = g_a / denominator
        probabilities[active_a] = p_a * (scaled_a / d_a)

    if d_b > 0:
        p_b = g_b / denominator
        probabilities[active_b] = p_b * (scaled_b / d_b)

    probabilities[-1] = 1.0 / denominator
    return probabilities