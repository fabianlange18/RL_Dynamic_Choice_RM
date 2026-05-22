import numpy as np

import src.constants as C


def mnl_reference_price_probabilities(action_binary, beta, reference_price=None):
    """Compute MNL probabilities with a reference-price utility adjustment.

    Applied formulation:
        U_i = beta * r_i + beta_ref * (r_ref - r_i),
        P(i | A) = exp(U_i) / (1 + sum_{j in A} exp(U_j)),
        P(0 | A) = 1 / (1 + sum_{j in A} exp(U_j)).
    """
    prices = np.asarray(C.r, dtype=float)
    action_binary = np.asarray(action_binary, dtype=bool)

    if reference_price is None:
        reference_price = float(np.mean(prices))

    beta_ref = C.MNL_REFERENCE_PRICE_BETA
    ref_adjustment = beta_ref * (reference_price - prices)
    utilities = np.where(action_binary, np.exp(beta * prices + ref_adjustment), 0.0)

    denominator = 1.0 + np.sum(utilities)
    probabilities = np.zeros(len(prices) + 1, dtype=float)
    probabilities[:-1] = utilities / denominator
    probabilities[-1] = 1.0 / denominator
    return probabilities