import os
import numpy as np
from scipy.optimize import minimize

import src.constants as C

def _reward_to_purchase_index(reward):
    if reward <= 0:
        return None
    matches = np.where(np.isclose(C.r, reward))[0]
    return int(matches[0]) if len(matches) else None


def collect_transaction_data(env, n_episodes=C.N_ESTIMATION_EPISODES):
    observations = []
    for _ in range(int(n_episodes)):
        env.reset()
        while True:
            current_t = int(env.s[0])
            sampled_action = env.action_space.sample()
            action_binary = env._action_to_binary(sampled_action)
            arrival_flag = int(env.arrival_xi[current_t])
            _, reward, done, truncated, _ = env.step(sampled_action)
            observations.append(
                {
                    "action_binary": action_binary,
                    "arrival_flag": arrival_flag,
                    "purchase_index": _reward_to_purchase_index(reward),
                }
            )
            if done or truncated:
                break
    return observations


def _normalize_simplex(weights):
    weights = np.asarray(weights, dtype=float)
    return weights / np.sum(weights)


class ScipyEstimator:
    """Direct maximum-likelihood estimator for MNL and finite-support MMNL."""

    @staticmethod
    def _estimate_lambda(observations):
        return float(np.mean([obs["arrival_flag"] for obs in observations]))

    def __init__(self, observations):
        self.observations = observations
        self.n = len(C.r)
        self.lambda_val = self._estimate_lambda(observations)
        self._beta_bounds = tuple(C.ESTIMATION_BETA_BOUNDS)
        self._optimizer_method = os.getenv("SCIPY_ESTIMATION_METHOD", "L-BFGS-B")
        self._maxiter = int(os.getenv("SCIPY_ESTIMATION_MAXITER", "500"))
        self._random_seed = int(os.getenv("SCIPY_ESTIMATION_SEED", "42"))
        self._mnl_restarts = int(os.getenv("SCIPY_MNL_RESTARTS", "3"))
        self._mmnl_restarts = int(os.getenv("SCIPY_MMNL_RESTARTS", "7"))
        self._mmnl2_min_weight = float(os.getenv("SCIPY_MMNL2_MIN_WEIGHT", "0.2"))
        self._mmnl5_min_weight = float(os.getenv("SCIPY_MMNL5_MIN_WEIGHT", "0.1"))
        self._mmnl_min_weight_floor = float(os.getenv("SCIPY_MMNL_MIN_WEIGHT_FLOOR", "1e-8"))

        self._prices = None
        self._available = None
        self._chosen = None
        self._chosen_is_outside = None
        self._chosen_product_index = None
        self._n_obs = 0
        self._arrival_purchase_rate = 0.0
        self._build_estimation_arrays()

    def _build_estimation_arrays(self):
        prices = np.asarray(C.r, dtype=float)
        arrival_obs = [obs for obs in self.observations if obs["arrival_flag"]]

        if not arrival_obs:
            raise ValueError("No arrival observations available for SciPy estimation.")

        self._n_obs = len(arrival_obs)
        self._prices = prices
        self._available = np.asarray(
            [np.asarray(obs["action_binary"], dtype=float) for obs in arrival_obs],
            dtype=float,
        )
        self._chosen = np.asarray(
            [self.n if obs["purchase_index"] is None else int(obs["purchase_index"]) for obs in arrival_obs],
            dtype=np.int32,
        )
        self._chosen_is_outside = self._chosen == self.n
        self._chosen_product_index = np.where(self._chosen_is_outside, 0, self._chosen)
        self._arrival_purchase_rate = float(1.0 - np.mean(self._chosen_is_outside))

        total_bytes = (
            self._prices.nbytes
            + self._available.nbytes
            + self._chosen.nbytes
            + self._chosen_is_outside.nbytes
            + self._chosen_product_index.nbytes
        )
        print(
            f"scipy estimation data: obs={self._n_obs}, alts={self.n + 1}, memory={total_bytes / (1024 ** 2):.2f} MiB"
        )
        print(
            f"scipy estimation mode: raw prices, beta_bounds={self._beta_bounds}, purchase_rate|arrival={self._arrival_purchase_rate:.4f}"
        )

    @staticmethod
    def _fit_stats(log_likelihood, n_obs, n_parameters):
        aic = 2.0 * float(n_parameters) - 2.0 * float(log_likelihood)
        bic = float(n_parameters) * np.log(int(n_obs)) - 2.0 * float(log_likelihood)
        return {
            "final_log_likelihood": float(log_likelihood),
            "aic": float(aic),
            "bic": float(bic),
        }

    def _segment_choice_probabilities(self, betas):
        betas = np.asarray(betas, dtype=float)
        exp_utilities = np.exp(np.outer(betas, self._prices))
        denominators = 1.0 + self._available @ exp_utilities.T

        chosen_exp = exp_utilities[:, self._chosen_product_index].T
        chosen_exp[self._chosen_is_outside] = 1.0

        return chosen_exp / denominators

    def _neg_log_likelihood(self, params, n_segments, min_weight):
        betas = np.asarray(params[:n_segments], dtype=float)
        segment_choice_probabilities = self._segment_choice_probabilities(betas)

        if int(n_segments) == 1:
            mixture_probabilities = segment_choice_probabilities[:, 0]
        else:
            free_weights = np.asarray(params[n_segments:], dtype=float)
            last_weight = 1.0 - np.sum(free_weights)
            if last_weight < float(min_weight):
                return 1e100
            weights = np.concatenate([free_weights, np.asarray([last_weight], dtype=float)])
            mixture_probabilities = segment_choice_probabilities @ weights

        return -float(np.sum(np.log(np.clip(mixture_probabilities, 1e-300, None))))

    def _initial_params(self, n_segments, restart_index, rng, min_weight):
        beta_low, beta_high = self._beta_bounds
        base_betas = np.linspace(beta_low, beta_high, int(n_segments) + 2, dtype=float)[1:-1]

        if restart_index > 0:
            jitter_scale = 0.1 * (beta_high - beta_low)
            base_betas = np.clip(
                base_betas + rng.normal(scale=jitter_scale, size=int(n_segments)),
                beta_low,
                beta_high,
            )

        if int(n_segments) == 1:
            return base_betas

        if int(n_segments) * float(min_weight) >= 1.0:
            raise ValueError(
                f"Infeasible minimum weight {min_weight} for {n_segments} segments. "
                "Need n_segments * min_weight < 1."
            )

        if restart_index == 0:
            init_weights = np.full(int(n_segments), 1.0 / int(n_segments), dtype=float)
        else:
            alpha = np.full(int(n_segments), 1.0, dtype=float)
            init_weights = rng.dirichlet(alpha)

        init_weights = np.maximum(init_weights, float(min_weight))
        init_weights = _normalize_simplex(init_weights)

        free_weights = init_weights[:-1]
        return np.concatenate([base_betas, free_weights])

    def _fit_latent_class_model(self, n_segments, min_weight):
        n_segments = int(n_segments)
        if n_segments < 1:
            raise ValueError("Number of segments must be at least 1.")

        rng = np.random.default_rng(self._random_seed)
        restart_count = self._mnl_restarts if n_segments == 1 else self._mmnl_restarts
        beta_bounds = [tuple(self._beta_bounds)] * n_segments
        weight_bounds = [(float(min_weight), 1.0 - float(min_weight))] * max(0, n_segments - 1)
        bounds = beta_bounds + weight_bounds

        best_result = None
        for restart_index in range(int(restart_count)):
            initial_params = self._initial_params(n_segments, restart_index, rng, min_weight)
            result = minimize(
                self._neg_log_likelihood,
                x0=initial_params,
                args=(n_segments, min_weight),
                method=self._optimizer_method,
                bounds=bounds,
                options={"maxiter": self._maxiter},
            )
            if best_result is None or result.fun < best_result.fun:
                best_result = result

        if best_result is None or not best_result.success:
            message = "No optimization result returned." if best_result is None else str(best_result.message)
            raise RuntimeError(f"SciPy estimation failed for {n_segments} segment(s): {message}")

        # Guard against pathological edge solutions that can collapse efficient sets to only the empty set.
        beta_low = float(self._beta_bounds[0])
        edge_tol = 1e-6
        fitted_betas = np.asarray(best_result.x[:n_segments], dtype=float)
        all_on_lower_bound = bool(np.all(np.abs(fitted_betas - beta_low) <= edge_tol))
        if all_on_lower_bound and self._arrival_purchase_rate > 0.5 and beta_low < -0.01:
            fallback_low = -0.01
            fallback_bounds = [(fallback_low, float(self._beta_bounds[1]))] * n_segments + weight_bounds
            fallback_result = None
            for restart_index in range(int(restart_count)):
                initial_params = self._initial_params(n_segments, restart_index, rng, min_weight)
                result = minimize(
                    self._neg_log_likelihood,
                    x0=initial_params,
                    args=(n_segments, min_weight),
                    method=self._optimizer_method,
                    bounds=fallback_bounds,
                    options={"maxiter": self._maxiter},
                )
                if fallback_result is None or result.fun < fallback_result.fun:
                    fallback_result = result

            if fallback_result is not None and fallback_result.success and fallback_result.fun <= best_result.fun + 1e-6:
                print(
                    "scipy estimation fallback: moved beta lower bound from "
                    f"{beta_low} to {fallback_low} to avoid degenerate lower-bound solution"
                )
                best_result = fallback_result

        betas = np.asarray(best_result.x[:n_segments], dtype=float)
        if n_segments == 1:
            weights = np.asarray([1.0], dtype=float)
        else:
            free_weights = np.asarray(best_result.x[n_segments:], dtype=float)
            last_weight = 1.0 - np.sum(free_weights)
            if last_weight < float(min_weight):
                raise RuntimeError(
                    f"SciPy estimation produced infeasible last segment weight {last_weight:.6g}."
                )
            weights = np.concatenate([free_weights, np.asarray([last_weight], dtype=float)])

        order = np.argsort(betas)
        betas = betas[order]
        weights = weights[order]
        weights = _normalize_simplex(weights)

        log_likelihood = -float(best_result.fun)
        return {
            "betas": betas,
            "weights": weights,
            "log_likelihood": log_likelihood,
            "optimizer_success": bool(best_result.success),
        }

    def _min_weight_for_k(self, K):
        K = int(K)
        if K <= 1:
            return 0.0

        w2 = float(self._mmnl2_min_weight)
        w5 = float(self._mmnl5_min_weight)
        if w2 <= 0.0 or w5 <= 0.0:
            raise ValueError("Minimum weights must be strictly positive.")

        # Power-law interpolation anchored at K=2 and K=5.
        exponent = np.log(w2 / w5) / np.log(5.0 / 2.0)
        scale = w2 * (2.0 ** exponent)
        min_weight = scale / (float(K) ** exponent)

        min_weight = max(min_weight, float(self._mmnl_min_weight_floor))

        # Enforce simplex feasibility: K * min_weight < 1.
        feasible_upper = (1.0 - 1e-9) / float(K)
        if min_weight >= feasible_upper:
            min_weight = max(float(self._mmnl_min_weight_floor), 0.95 * feasible_upper)

        return float(min_weight)

    def estimate_mnl(self):
        fit_result = self._fit_latent_class_model(1, min_weight=0.0)
        beta_hat = float(fit_result["betas"][0])
        beta_hat = float(np.clip(beta_hat, self._beta_bounds[0], self._beta_bounds[1]))

        return {
            "beta": beta_hat,
            "lambda": self.lambda_val,
            **self._fit_stats(fit_result["log_likelihood"], self._n_obs, 1),
            "success": fit_result["optimizer_success"],
        }

    def estimate_mmnl(self, K=5):
        K = int(K)
        if K < 2:
            raise ValueError("ScipyEstimator.estimate_mmnl requires at least 2 support points.")

        min_weight = self._min_weight_for_k(K)

        fit_result = self._fit_latent_class_model(K, min_weight=min_weight)
        n_parameters = K + (K - 1)
        betas_raw = np.asarray(fit_result["betas"], dtype=float)
        betas_raw = np.clip(betas_raw, self._beta_bounds[0], self._beta_bounds[1])

        return {
            "betas": [float(beta) for beta in betas_raw],
            "n_segments": K,
            "mixing_weights": [float(weight) for weight in fit_result["weights"]],
            "lambda": self.lambda_val,
            **self._fit_stats(fit_result["log_likelihood"], self._n_obs, n_parameters),
            "success": fit_result["optimizer_success"],
        }


if __name__ == "__main__":
    from src.talluri_env import TalluriExample2

    test_episodes = int(os.getenv("SCIPY_TEST_EPISODES", "5"))
    environment = TalluriExample2(efficient_sets=None)

    try:
        test_observations = collect_transaction_data(environment, n_episodes=test_episodes)
    finally:
        environment.close()

    test_estimator = ScipyEstimator(test_observations)

    print("Running scipy estimation smoke test")
    print(f"Episodes: {test_episodes}")
    print("MNL:")
    print(test_estimator.estimate_mnl())
    print("MMNL 2PT:")
    print(test_estimator.estimate_mmnl(K=2))
    print("MMNL 5PT:")
    print(test_estimator.estimate_mmnl(K=5))