import numpy as np

import constants as C
from buying_probabilities import get_buying_probabilities_by_model


def _action_int_to_binary(action_int):
    return np.array([(int(action_int) >> i) & 1 for i in range(C.n)], dtype=int)


def solve_by_dp(estimated_beta, estimated_lambda, efficient_sets=None, model="MNL", segment_betas=None):
    arrival_prob = float(C.ARRIVAL_PROB if estimated_lambda is None else np.clip(estimated_lambda, 0.0, 1.0))
    no_arrival_prob = 1.0 - arrival_prob

    v = np.zeros((C.T + 1, C.C + 1))
    pi = np.zeros((C.T, C.C + 1), dtype=int)

    for t in range(C.T - 1, -1, -1):
        for x in range(C.C + 1):
            if x <= 0:
                continue

            best_value = 0.0
            best_action = 0

            for action_idx, action_int in enumerate(efficient_sets if efficient_sets is not None else range(2 ** C.n)):
                action_binary = _action_int_to_binary(action_int)
                expected_value = no_arrival_prob * v[t + 1, x]

                if arrival_prob > 0:
                    buying_probabilities = get_buying_probabilities_by_model(
                        action_binary=action_binary,
                        prices=C.r,
                        beta=estimated_beta,
                        model=model,
                        segment_betas=segment_betas,
                    )

                    immediate_reward = float(np.dot(C.r, buying_probabilities))
                    purchase_prob = float(np.sum(buying_probabilities))

                    expected_future_value = 0.0
                    if purchase_prob > 0:
                        expected_future_value += purchase_prob * v[t + 1, x - 1]

                    expected_future_value += (1.0 - purchase_prob) * v[t + 1, x]
                    expected_value += arrival_prob * (immediate_reward + expected_future_value)

                if expected_value > best_value:
                    best_value = expected_value
                    best_action = action_idx

            v[t, x] = best_value
            pi[t, x] = best_action

    return v, pi
