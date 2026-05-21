import numpy as np

import constants as C
from buying_probabilities import get_buying_probabilities_by_model


def _decode_actions(action_ints, n):
    """Decode integer actions to binary offer vectors."""
    return ((action_ints[:, None] >> np.arange(n)) & 1).astype(np.int8)


def _sample_actions(n, max_actions, rng):
    """Sample a bounded action set when 2**n is too large to enumerate."""
    action_binary = rng.integers(0, 2, size=(max_actions, n), dtype=np.int8)
    powers = np.asarray([1 << i for i in range(n)], dtype=object)
    action_ints = np.sum(action_binary.astype(object) * powers, axis=1)
    return np.asarray(action_ints, dtype=object), action_binary


def _build_action_space(efficient_sets, n, max_actions, rng):
    """
    Returns:
        action_ids: integer action ids used when policy should output action integers
        action_binary: binary matrix shape (A, n)
        policy_returns_index: whether policy should store argmax index instead of action id
    """
    if efficient_sets is not None:
        idx = np.asarray(list(efficient_sets), dtype=object)
        return idx, _decode_actions(idx, n), True

    total_actions = 2 ** n
    if total_actions <= max_actions:
        action_ints = np.arange(total_actions, dtype=np.int64)
        return action_ints, _decode_actions(action_ints, n), False

    action_ints, action_binary = _sample_actions(n=n, max_actions=max_actions, rng=rng)
    return action_ints, action_binary, False


def _precompute_action_stats(
    action_binary,
    estimated_beta,
    model="MNL",
    segment_betas=None,
    segment_weights=None,
    mu_b=None,
    sigma_b=None,
):
    """Precompute expected immediate reward and purchase probability for each action."""
    n_actions = action_binary.shape[0]
    reward = np.zeros(n_actions, dtype=np.float32)
    purchase_prob = np.zeros(n_actions, dtype=np.float32)

    for a in range(n_actions):
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


def _fit_quadratic_value(x, y, ridge_lambda):
    """Fit y ~= a0 + a1*x + a2*x^2 using ridge regularization."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    design = np.column_stack([np.ones_like(x), x, x * x])
    gram = design.T @ design
    ridge = ridge_lambda * np.eye(3, dtype=np.float64)
    coeffs = np.linalg.solve(gram + ridge, design.T @ y)
    return coeffs


def _predict_quadratic_value(coeffs, capacity_max):
    """Evaluate quadratic approximation on all capacities 0..C."""
    x = np.arange(capacity_max + 1, dtype=np.float64)
    vals = coeffs[0] + coeffs[1] * x + coeffs[2] * x * x
    return vals.astype(np.float32)


def solve_by_adp(
    estimated_beta,
    estimated_lambda,
    efficient_sets=None,
    model="MNL",
    segment_betas=None,
    segment_weights=None,
    mu_b=None,
    sigma_b=None,
    max_actions=4096,
    n_state_samples=64,
    ridge_lambda=1e-6,
    random_seed=0,
):
    """
    Approximate Dynamic Programming (ADP) solver using fitted value iteration.

    Key approximations:
      - Sample capacities each time step instead of evaluating all states for fitting.
      - Fit a quadratic value function V_t(x) over capacity x.
      - Optionally sample action space when 2**n is too large.

    Returns:
        v: value table with shape (T + 1, C + 1)
        pi: policy table with shape (T, C + 1)
            - If efficient_sets is provided, entries are indices into list(efficient_sets).
            - Otherwise, entries are action integers.
    """
    rng = np.random.default_rng(random_seed)

    arrival_prob = float(
        C.ARRIVAL_PROB if estimated_lambda is None else np.clip(estimated_lambda, 0.0, 1.0)
    )
    no_arrival_prob = 1.0 - arrival_prob

    action_ids, action_binary, policy_returns_index = _build_action_space(
        efficient_sets=efficient_sets,
        n=C.n,
        max_actions=max_actions,
        rng=rng,
    )

    reward, purchase_prob = _precompute_action_stats(
        action_binary=action_binary,
        estimated_beta=estimated_beta,
        model=model,
        segment_betas=segment_betas,
        segment_weights=segment_weights,
        mu_b=mu_b,
        sigma_b=sigma_b,
    )

    v = np.zeros((C.T + 1, C.C + 1), dtype=np.float32)
    pi = np.zeros((C.T, C.C + 1), dtype=object if not policy_returns_index else np.int32)

    all_caps = np.arange(1, C.C + 1, dtype=np.int32)

    for t in range(C.T - 1, -1, -1):
        next_v = v[t + 1]

        if len(all_caps) <= n_state_samples:
            sampled_caps = all_caps
        else:
            sampled_caps = np.sort(rng.choice(all_caps, size=n_state_samples, replace=False))

        sampled_values = np.zeros_like(sampled_caps, dtype=np.float64)

        for i, x in enumerate(sampled_caps):
            stay_value = next_v[x]
            buy_value = next_v[x - 1]

            vals = (
                no_arrival_prob * stay_value
                + arrival_prob
                * (
                    reward
                    + purchase_prob * buy_value
                    + (1.0 - purchase_prob) * stay_value
                )
            )
            sampled_values[i] = float(np.max(vals))

        # Enforce terminal and zero-capacity anchors in the regression.
        fit_x = np.concatenate(([0], sampled_caps, [C.C]))
        fit_y = np.concatenate(([0.0], sampled_values, [sampled_values[-1] if sampled_values.size > 0 else 0.0]))

        coeffs = _fit_quadratic_value(x=fit_x, y=fit_y, ridge_lambda=ridge_lambda)
        vt = _predict_quadratic_value(coeffs=coeffs, capacity_max=C.C)

        # Structural cleanups: non-negative and non-decreasing in capacity.
        vt = np.maximum(vt, 0.0)
        vt[0] = 0.0
        vt = np.maximum.accumulate(vt)
        v[t] = vt

        for x in range(1, C.C + 1):
            stay_value = next_v[x]
            buy_value = next_v[x - 1]

            vals = (
                no_arrival_prob * stay_value
                + arrival_prob
                * (
                    reward
                    + purchase_prob * buy_value
                    + (1.0 - purchase_prob) * stay_value
                )
            )
            best_idx = int(np.argmax(vals))
            pi[t, x] = best_idx if policy_returns_index else action_ids[best_idx]

    return v, pi
