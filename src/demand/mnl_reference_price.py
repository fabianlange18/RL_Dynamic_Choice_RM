import numpy as np

import src.constants as C


def mnl_reference_price_probabilities(action_binary, beta, reference_price=None):
    """Compute MNL probabilities with asymmetric reference-price effects.

    Utility per offered product j:
        u_j = beta * r_j
              + beta_ref_plus * max(0, p_ref - r_j)
              - beta_ref_minus * max(0, r_j - p_ref)
    with outside-option utility fixed at 0.
    """
    prices = np.asarray(C.r, dtype=float)
    action_binary = np.asarray(action_binary, dtype=bool)

    reference_price = float(np.mean(prices)) if reference_price is None else float(reference_price)

    gains = np.maximum(0.0, reference_price - prices)
    losses = np.maximum(0.0, prices - reference_price)

    deterministic_utility = (
        beta * prices
        + C.MNL_REFERENCE_PRICE_BETA_GAIN * gains
        - C.MNL_REFERENCE_PRICE_BETA_LOSS * losses
    )
    offered_exp_utility = np.where(action_binary, np.exp(deterministic_utility), 0.0)

    denominator = 1.0 + np.sum(offered_exp_utility)
    probabilities = np.zeros(len(prices) + 1, dtype=float)
    probabilities[:-1] = offered_exp_utility / denominator
    probabilities[-1] = 1.0 / denominator
    return probabilities