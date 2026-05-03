import numpy as np

import constants as C
from buying_probabilities import get_buying_probabilities_by_model


def _precompute_actions(n: int):
    """
    Returns:
        action_ints: shape (A,)
        action_binary: shape (A, n)
    """
    A = 2 ** n
    action_ints = np.arange(A, dtype=np.int32)

    # Bit-decode all actions at once
    action_binary = ((action_ints[:, None] >> np.arange(n)) & 1).astype(np.int8)

    return action_ints, action_binary


def _precompute_action_stats(
    action_binary,
    estimated_beta,
    model="MNL",
    segment_betas=None,
    segment_weights=None,
    mu_b=None,
    sigma_b=None,
):
    """
    Precompute for every action:
        reward[a]        = expected immediate reward
        purchase_prob[a] = probability any purchase occurs
    """
    A = action_binary.shape[0]

    reward = np.zeros(A, dtype=np.float32)
    purchase_prob = np.zeros(A, dtype=np.float32)

    for a in range(A):
        probs = get_buying_probabilities_by_model(
            action_binary=action_binary[a],
            beta=estimated_beta,
            model=model,
            segment_betas=segment_betas,
            segment_weights=segment_weights,
            mu_b=mu_b,
            sigma_b=sigma_b,
        )

        probs = np.asarray(probs, dtype=np.float64)

        reward[a] = np.dot(C.r, probs)
        purchase_prob[a] = probs.sum()

    return reward, purchase_prob


def solve_by_dp(
    estimated_beta,
    estimated_lambda,
    efficient_sets=None,
    model="MNL",
    segment_betas=None,
    segment_weights=None,
    mu_b=None,
    sigma_b=None,
):
    """
    Faster DP solver:
      - precomputes all actions once
      - precomputes reward + purchase probs once
      - vectorized action argmax at each state
    """

    arrival_prob = float(
        C.ARRIVAL_PROB
        if estimated_lambda is None
        else np.clip(estimated_lambda, 0.0, 1.0)
    )
    no_arrival_prob = 1.0 - arrival_prob

    # ----------------------------
    # Build action space
    # ----------------------------
    if efficient_sets is not None:
        # Decode only the provided action integers – avoids 2**n enumeration.
        idx = np.asarray(list(efficient_sets), dtype=object)
        action_binary = ((idx[:, None] >> np.arange(C.n)) & 1).astype(np.int8)
    else:
        _, action_binary = _precompute_actions(C.n)

    # ----------------------------
    # Precompute action stats ONCE
    # ----------------------------
    reward, purchase_prob = _precompute_action_stats(
        action_binary=action_binary,
        estimated_beta=estimated_beta,
        model=model,
        segment_betas=segment_betas,
        segment_weights=segment_weights,
        mu_b=mu_b,
        sigma_b=sigma_b,
    )

    # ----------------------------
    # DP arrays
    # ----------------------------
    v = np.zeros((C.T + 1, C.C + 1), dtype=np.float32)
    pi = np.zeros((C.T, C.C + 1), dtype=np.int32)

    # ----------------------------
    # Backward recursion
    # ----------------------------
    for t in range(C.T - 1, -1, -1):

        next_v = v[t + 1]

        for x in range(1, C.C + 1):

            stay_value = next_v[x]
            buy_value = next_v[x - 1]

            # value for all actions simultaneously
            vals = (
                no_arrival_prob * stay_value
                + arrival_prob
                * (
                    reward
                    + purchase_prob * buy_value
                    + (1.0 - purchase_prob) * stay_value
                )
            )

            best_idx = np.argmax(vals)

            v[t, x] = vals[best_idx]
            pi[t, x] = best_idx

    return v, pi