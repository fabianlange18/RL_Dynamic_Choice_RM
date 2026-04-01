import numpy as np
from functools import lru_cache

import config as c
import constants as C
from buying_probabilities import get_buying_probabilities_by_model


def _action_int_to_binary(action_int):
    return np.array([(action_int >> i) & 1 for i in range(C.n)], dtype=int)


@lru_cache(maxsize=8)
def _identify_efficient_sets(model, beta=None, segment_betas=None):
    beta = None if beta is None else float(beta)
    segment_betas = None if segment_betas is None else tuple(float(b) for b in segment_betas)
    offer_set_metrics = []

    for action_int in range(2 ** C.n):
        action_binary = _action_int_to_binary(action_int)
        buying_probabilities = np.asarray(
            get_buying_probabilities_by_model(
                action_binary,
                C.r,
                beta,
                model=model,
                segment_betas=segment_betas if model == "MMNL" else None,
            ),
            dtype=float,
        )
        q_value = float(np.sum(buying_probabilities))
        r_value = float(np.dot(C.r, buying_probabilities))
        offer_set_metrics.append({"action": action_int, "Q": q_value, "R": r_value})

    tol = 1e-12
    efficient_sequence = [0]
    current = offer_set_metrics[0]

    while True:
        best_candidate = None
        best_ratio = -np.inf

        for candidate in offer_set_metrics:
            if candidate["action"] in efficient_sequence:
                continue

            if candidate["Q"] + tol < current["Q"] or candidate["R"] + tol < current["R"]:
                continue

            delta_q = candidate["Q"] - current["Q"]
            delta_r = candidate["R"] - current["R"]

            if delta_q <= tol:
                continue

            marginal_ratio = delta_r / delta_q
            if (
                best_candidate is None
                or marginal_ratio > best_ratio + tol
                or (
                    abs(marginal_ratio - best_ratio) <= tol
                    and (
                        candidate["R"] > best_candidate["R"] + tol
                        or (
                            abs(candidate["R"] - best_candidate["R"]) <= tol
                            and candidate["Q"] > best_candidate["Q"] + tol
                        )
                    )
                )
            ):
                best_ratio = marginal_ratio
                best_candidate = candidate

        if best_candidate is None:
            break

        efficient_sequence.append(best_candidate["action"])
        current = best_candidate

    return tuple(efficient_sequence)


def compute_efficient_sets(model, beta=None, segment_betas=None):
    if beta is None and segment_betas is None:
        beta = C.SENSITIVITY_BETA_GT["high"] if c.HIGH_SENSITIVITY else C.SENSITIVITY_BETA_GT["low"]

    # Convert segment_betas to tuple for lru_cache hashability
    if segment_betas is not None and isinstance(segment_betas, list):
        segment_betas = tuple(segment_betas)

    return _identify_efficient_sets(model=model, beta=beta, segment_betas=segment_betas)