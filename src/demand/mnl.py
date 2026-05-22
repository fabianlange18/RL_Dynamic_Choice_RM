import numpy as np

import src.constants as C


def mnl_probabilities(action_binary, beta):
    """Compute MNL probabilities with an outside option.

    Applied formulation:
        U_i = beta * r_i for i in A,  U_0 = 0,
        P(i | A) = exp(U_i) / (1 + sum_{j in A} exp(U_j)),
        P(0 | A) = 1 / (1 + sum_{j in A} exp(U_j)).
    """
    prices = np.asarray(C.r, dtype=float)
    action_binary = np.asarray(action_binary, dtype=bool)
    utilities = np.where(action_binary, np.exp(beta * prices), 0.0)

    denominator = 1.0 + np.sum(utilities)
    probabilities = np.zeros(len(prices) + 1, dtype=float)
    probabilities[:-1] = utilities / denominator
    probabilities[-1] = 1.0 / denominator
    return probabilities