import numpy as np
from functools import lru_cache

import src.constants as C


_PRICES = np.asarray(C.r, dtype=float)
_J = len(_PRICES)
_GLOBAL_RNG = np.random.default_rng()


# def probit_probabilities(action_binary, beta, seed=None):
#     """Return one-draw Probit outcome probabilities.

#     Applied formulation:
#         U_i = beta * r_i + epsilon_i,  epsilon_i ~ N(0, 1),
#         U_0 = epsilon_0,  epsilon_0 ~ N(0, 1),
#         choose argmax among offered products and the outside option.

#     This implementation intentionally uses exactly one utility draw.
#     """
#     prices = np.asarray(C.r, dtype=float)
#     action_binary = np.asarray(action_binary)
#     active_indices = np.where(action_binary == 1)[0]

#     counts = np.zeros(len(prices) + 1, dtype=float)
#     if len(active_indices) == 0:
#         counts[-1] = 1.0
#         return counts

#     rng = np.random.default_rng() if seed is None else np.random.default_rng(int(seed))
#     outside_utility = float(rng.normal(0.0, 1.0))
#     utilities = np.full(len(prices), -np.inf, dtype=float)
#     utilities[active_indices] = beta * prices[active_indices] + rng.normal(0.0, 1.0, size=len(active_indices))

#     chosen = int(np.argmax(utilities))
#     if utilities[chosen] > outside_utility:
#         counts[chosen] = 1.0
#     else:
#         counts[-1] = 1.0

#     return counts


def _default_block_sigma(n_products):
    """Build a default block-structured covariance for product + outside errors."""
    n_blocks = 2
    rho_within = 0.35
    rho_between = 0.10
    rho_outside = 0.05

    n_alt = int(n_products) + 1
    sigma = np.full((n_alt, n_alt), rho_between, dtype=float)
    np.fill_diagonal(sigma, 1.0)

    # Two price blocks for product alternatives.
    block_edges = np.linspace(0, int(n_products), n_blocks + 1, dtype=int)
    for block_idx in range(n_blocks):
        start = int(block_edges[block_idx])
        end = int(block_edges[block_idx + 1])
        if end <= start:
            continue
        sigma[start:end, start:end] = rho_within
        np.fill_diagonal(sigma[start:end, start:end], 1.0)

    # Outside option weakly correlated with all inside alternatives.
    sigma[:-1, -1] = rho_outside
    sigma[-1, :-1] = rho_outside
    sigma[-1, -1] = 1.0

    # Ensure numerical PSD and unit variances.
    eigvals, eigvecs = np.linalg.eigh(sigma)
    eigvals = np.clip(eigvals, 1e-8, None)
    sigma = (eigvecs * eigvals) @ eigvecs.T
    std = np.sqrt(np.clip(np.diag(sigma), 1e-12, None))
    sigma = sigma / np.outer(std, std)
    np.fill_diagonal(sigma, 1.0)
    return sigma


@lru_cache(maxsize=4)
def _default_block_cholesky(n_products):
    """Cached Cholesky factor for the default block covariance."""
    sigma = _default_block_sigma(n_products)
    return np.linalg.cholesky(sigma)


def probit_probabilities(action_binary, beta, seed=None):
    """Return one-draw multinomial probit outcome with MVN error terms.

    If Sigma is None, a default block-structured covariance is used.
    Sigma must be shaped (J+1, J+1), including outside option covariance.
    """
    action_binary = np.asarray(action_binary)
    active_indices = np.flatnonzero(action_binary)

    J = _J
    counts = np.zeros(J + 1, dtype=float)

    if len(active_indices) == 0:
        counts[-1] = 1.0
        return counts

    L = _default_block_cholesky(J)
    rng = _GLOBAL_RNG if seed is None else np.random.default_rng(seed)

    # One MVN draw via cached Cholesky factor: eps ~ N(0, Sigma).
    eps = L @ rng.normal(0.0, 1.0, size=J + 1)

    # split
    eps_inside = eps[:J]
    eps_outside = eps[-1]

    utilities = np.full(J, -np.inf, dtype=float)
    utilities[active_indices] = beta * _PRICES[active_indices] + eps_inside[active_indices]

    chosen = int(np.argmax(utilities))

    if utilities[chosen] > eps_outside:
        counts[chosen] = 1.0
    else:
        counts[-1] = 1.0

    return counts
