import numpy as np

import src.constants as C


def probit_probabilities(action_binary, beta, seed=None):
    """Return one-draw Probit outcome probabilities.

    Applied formulation:
        U_i = beta * r_i + epsilon_i,  epsilon_i ~ N(0, 1),
        U_0 = epsilon_0,  epsilon_0 ~ N(0, 1),
        choose argmax among offered products and the outside option.

    This implementation intentionally uses exactly one utility draw.
    """
    prices = np.asarray(C.r, dtype=float)
    action_binary = np.asarray(action_binary)
    active_indices = np.where(action_binary == 1)[0]

    counts = np.zeros(len(prices) + 1, dtype=float)
    if len(active_indices) == 0:
        counts[-1] = 1.0
        return counts

    rng = np.random.default_rng() if seed is None else np.random.default_rng(int(seed))
    outside_utility = float(rng.normal(0.0, 1.0))
    utilities = np.full(len(prices), -np.inf, dtype=float)
    utilities[active_indices] = beta * prices[active_indices] + rng.normal(0.0, 1.0, size=len(active_indices))

    chosen = int(np.argmax(utilities))
    if utilities[chosen] > outside_utility:
        counts[chosen] = 1.0
    else:
        counts[-1] = 1.0

    return counts