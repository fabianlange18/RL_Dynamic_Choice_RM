import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

import constants as C


BETA_BOUNDS = (-0.05, -1e-6)
MMNL_2PT_MIN_WEIGHT = 0.2
MMNL_5PT_MIN_WEIGHT = 0.055
EPS = 1e-300

# Read from environment variables if set, otherwise use defaults
MMNL_WEIGHT_SEP_LAMBDA = 0.0
MMNL_WEIGHT_SEP_ALPHA = 5.0


def _min_weight_for_k(k):
    """Interpolate true per-segment minimum weight between K=2 and K>=5 settings."""
    k_val = max(int(k), 2)
    ratio = min(max((k_val - 2) / 3.0, 0.0), 1.0)  # 2->0.0, 5+->1.0
    min_w = (1.0 - ratio) * MMNL_2PT_MIN_WEIGHT + ratio * MMNL_5PT_MIN_WEIGHT
    max_feasible = (1.0 - 1e-9) / k_val
    return float(min(min_w, max_feasible))


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


class ScipyEstimator:
    """SciPy MLE estimator with output format aligned with BiogemeEstimator."""

    @staticmethod
    def _estimate_lambda(observations):
        return float(np.mean([obs["arrival_flag"] for obs in observations]))

    @staticmethod
    def _build_dataframe(observations):
        n = len(C.r)
        rows = []
        for obs in observations:
            if not obs["arrival_flag"]:
                continue
            row = {"CHOICE": obs["purchase_index"] if obs["purchase_index"] is not None else n}
            action_binary = np.asarray(obs["action_binary"], dtype=int)
            for j in range(n):
                row[f"AV_{j}"] = int(action_binary[j])
            row[f"AV_{n}"] = 1
            rows.append(row)
        return pd.DataFrame(rows)

    @staticmethod
    def _fit_stats(log_likelihood, n_params, n_obs):
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(max(n_obs, 1)) - 2 * log_likelihood
        return {
            "final_log_likelihood": float(log_likelihood),
            "aic": float(aic),
            "bic": float(bic),
        }

    def __init__(self, observations):
        self.observations = observations
        self.n = len(C.r)
        self.r = np.asarray(C.r, dtype=float)
        self.lambda_val = self._estimate_lambda(observations)

        self._dataframe = self._build_dataframe(observations)
        self.n_obs = len(self._dataframe)
        self.choice = self._dataframe["CHOICE"].to_numpy(dtype=int)
        self.av = self._dataframe[[f"AV_{j}" for j in range(self.n + 1)]].to_numpy(dtype=bool)

    def _chosen_logprob(self, product_utility):
        full_utility = np.concatenate([np.asarray(product_utility, dtype=float), [0.0]])
        util = np.where(self.av, full_utility[None, :], -np.inf)
        lse = logsumexp(util, axis=1)
        chosen_util = util[np.arange(self.n_obs), self.choice]
        return chosen_util - lse

    def _chosen_prob(self, product_utility):
        return np.exp(self._chosen_logprob(product_utility))

    @staticmethod
    def _run_minimize(fun, x0, bounds):
        return minimize(
            fun,
            x0=np.asarray(x0, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 2000, "ftol": 1e-9},
        )

    @staticmethod
    def _sample_start_point(base_x0, bounds, rng, jitter_scale=0.25):
        """Sample a perturbed start point that respects finite parameter bounds."""
        candidate = np.asarray(base_x0, dtype=float).copy()
        for i, (low, high) in enumerate(bounds):
            if low is not None and high is not None:
                width = max(float(high) - float(low), 1e-12)
                step = rng.normal(0.0, jitter_scale * width)
                candidate[i] = np.clip(candidate[i] + step, low, high)
            else:
                scale = max(abs(candidate[i]), 1.0)
                candidate[i] = candidate[i] + rng.normal(0.0, jitter_scale * scale)
        return candidate

    def _run_multistart(self, fun, x0, bounds, n_starts=1, random_seed=0):
        """Run L-BFGS-B from multiple starts and keep the best objective value."""
        rng = np.random.default_rng(random_seed)

        starts = [np.asarray(x0, dtype=float)]
        for _ in range(max(int(n_starts) - 1, 0)):
            starts.append(self._sample_start_point(x0, bounds, rng))

        best_res = None
        best_fun = np.inf
        for start in starts:
            res = self._run_minimize(fun, x0=start, bounds=bounds)
            if np.isfinite(res.fun) and res.fun < best_fun:
                best_fun = float(res.fun)
                best_res = res

        # Fallback in the unlikely case every run is invalid
        if best_res is None:
            best_res = self._run_minimize(fun, x0=x0, bounds=bounds)

        return best_res

    def estimate_mnl(self):
        beta_low, beta_high = BETA_BOUNDS

        def nll(theta):
            beta = theta[0]
            ll = np.sum(self._chosen_logprob(beta * self.r))
            return -ll

        res = self._run_minimize(nll, x0=[-0.01], bounds=[(beta_low, beta_high)])
        beta_hat = float(res.x[0])
        ll = -float(res.fun)
        return {
            "beta": beta_hat,
            "lambda": self.lambda_val,
            **self._fit_stats(ll, n_params=1, n_obs=self.n_obs),
            "success": bool(res.success),
        }

    def estimate_mmnl(self, K=5, n_starts=5, random_seed=0, weight_sep_lambda=MMNL_WEIGHT_SEP_LAMBDA, weight_sep_alpha=MMNL_WEIGHT_SEP_ALPHA):
        beta_low, beta_high = BETA_BOUNDS
        min_weight = _min_weight_for_k(K)
        init_betas = np.linspace(beta_low, beta_high, K)

        x0 = []
        bounds = []

        # beta_1..beta_K
        for k in range(K):
            x0.append(float(init_betas[k]))
            bounds.append((beta_low, beta_high))

        # wlogit_1..wlogit_{K-1}
        for _ in range(K - 1):
            x0.append(0.0)
            bounds.append((None, None))

        def unpack(theta):
            idx = K
            betas = [float(theta[k]) for k in range(K)]
            logits = theta[idx : idx + (K - 1)]
            logits = np.asarray(logits, dtype=float)
            exp_logits = np.exp(np.clip(logits, -700, 700))
            denom = 1.0 + np.sum(exp_logits)
            simplex_weights = np.concatenate([exp_logits / denom, [1.0 / denom]])
            weights = min_weight + (1.0 - K * min_weight) * simplex_weights
            return np.asarray(betas, dtype=float), weights

        def _weight_separation_penalty(logits):
            if weight_sep_lambda <= 0.0:
                return 0.0
            exp_logits = np.exp(np.clip(logits, -700, 700))
            denom = 1.0 + np.sum(exp_logits)
            simplex_weights = np.concatenate([exp_logits / denom, [1.0 / denom]])
            weights = min_weight + (1.0 - K * min_weight) * simplex_weights
            penalty = 0.0
            for i in range(K):
                for j in range(i + 1, K):
                    diff = weights[i] - weights[j]
                    penalty += np.exp(-weight_sep_alpha * diff * diff)
            return float(weight_sep_lambda * penalty)

        def nll(theta):
            betas, weights = unpack(theta)
            class_probs = []
            for k in range(K):
                class_probs.append(self._chosen_prob(betas[k] * self.r))
            class_probs = np.vstack(class_probs)
            mix_prob = np.dot(weights, class_probs)
            logits = np.asarray(theta[-(K - 1):], dtype=float)
            return -np.sum(np.log(np.clip(mix_prob, EPS, None))) + _weight_separation_penalty(logits)

        res = self._run_multistart(
            nll,
            x0=x0,
            bounds=bounds,
            n_starts=n_starts,
            random_seed=random_seed,
        )
        betas_hat, weights_hat = unpack(res.x)
        ll = -float(res.fun)
        n_params = K + (K - 1)
        return {
            "betas": [float(b) for b in betas_hat],
            "n_segments": K,
            "mixing_weights": [float(w) for w in weights_hat],
            "lambda": self.lambda_val,
            **self._fit_stats(ll, n_params=n_params, n_obs=self.n_obs),
            "success": bool(res.success),
        }

    def estimate_mmnl_continuous(self, n_draws=500, draw_seed=0, n_starts=5, random_seed=0):
        rng = np.random.default_rng(draw_seed)
        z = rng.standard_normal(int(n_draws))

        def nll(theta):
            mu_b, sigma_b = theta
            sigma_b = max(sigma_b, 0.0)
            betas = -np.exp(mu_b + sigma_b * z)
            probs = np.empty((len(betas), self.n_obs), dtype=float)
            for d, beta_d in enumerate(betas):
                probs[d] = self._chosen_prob(beta_d * self.r)
            mix_prob = np.mean(probs, axis=0)
            return -np.sum(np.log(np.clip(mix_prob, EPS, None)))

        x0 = [np.log(0.01), 0.3]
        bounds = [(None, None), (0.0, None)]
        res = self._run_multistart(
            nll,
            x0=x0,
            bounds=bounds,
            n_starts=n_starts,
            random_seed=random_seed,
        )

        mu_hat = float(res.x[0])
        sigma_hat = float(max(res.x[1], 0.0))
        ll = -float(res.fun)
        return {
            "mu_b": mu_hat,
            "sigma_b": sigma_hat,
            "mean_beta": float(-np.exp(mu_hat + 0.5 * sigma_hat**2)),
            "lambda": self.lambda_val,
            **self._fit_stats(ll, n_params=2, n_obs=self.n_obs),
            "success": bool(res.success),
        }