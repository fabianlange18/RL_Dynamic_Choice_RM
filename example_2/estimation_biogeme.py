import logging
import math

import numpy as np
import pandas as pd

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, log, exp, Numeric, bioDraws, MonteCarlo
from biogeme.parameters import Parameters

import constants as C

logging.getLogger("biogeme").setLevel(logging.ERROR)

BETA_BOUNDS = (-0.05, -1e-6)
MMNL_2PT_MIN_WEIGHT = 0.2
MMNL_5PT_MIN_WEIGHT = 0.055

# Read from environment variables if set, otherwise use defaults
MMNL_WEIGHT_SEP_LAMBDA = 0.0
MMNL_WEIGHT_SEP_ALPHA = 5.0


def _logit(p):
    p = min(max(float(p), 1e-9), 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


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


class BiogemeEstimator:
    """Unified estimator for multiple MMNL models with shared database and precomputed values."""
    
    @staticmethod
    def _estimate_lambda(observations):
        """Calculate arrival rate lambda from observations."""
        return float(np.mean([obs["arrival_flag"] for obs in observations]))

    @staticmethod
    def _build_dataframe(observations):
        """Convert observations into Biogeme-compatible DataFrame."""
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
    def _compute_fit_statistics(results):
        """Extract fit statistics from Biogeme estimation results."""
        stats = results.get_general_statistics()
        return {
            "final_log_likelihood": float(stats["Final log likelihood"]),
            "aic": float(stats["Akaike Information Criterion"]),
            "bic": float(stats["Bayesian Information Criterion"])
        }

    @staticmethod
    def _make_biogeme(database, logprob, name):
        """Create a BIOGEME instance with standard parameters."""
        params = Parameters()
        params.set_value("optimization_algorithm", "scipy")
        params.set_value("save_iterations", False)
        m = bio.BIOGEME(
            database,
            logprob,
            parameters=params,
            generate_html=False,
            generate_yaml=False,
        )
        m.modelName = name
        return m

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

    def _estimate_multistart_biogeme(self, builder_fn, n_starts=1, random_seed=0, make_biogeme_fn=None):
        """Run Biogeme estimation from multiple starts and keep best LL result."""
        rng = np.random.default_rng(random_seed)
        best = None
        make_biogeme_fn = self._make_biogeme if make_biogeme_fn is None else make_biogeme_fn

        base_x0, bounds = builder_fn(start_values=None, return_x0_only=True)
        starts = [np.asarray(base_x0, dtype=float)]
        for _ in range(max(int(n_starts) - 1, 0)):
            starts.append(self._sample_start_point(base_x0, bounds, rng))

        for start in starts:
            _, _, logprob, model_name = builder_fn(start_values=start, return_x0_only=False)
            biogeme_model = make_biogeme_fn(self.database, logprob, model_name)
            res = biogeme_model.estimate()

            ll = float(self._compute_fit_statistics(res)["final_log_likelihood"])
            if (best is None) or (ll > best["ll"]):
                best = {"ll": ll, "result": res}

        return best["result"]
    
    def __init__(self, observations):
        """Initialize estimator with observations.
        
        Parameters
        ----------
        observations : list
            List of observation dictionaries from collect_transaction_data()
        """
        self.observations = observations
        self.n = len(C.r)
        
        # Precompute once
        self._dataframe = self._build_dataframe(observations)
        self.n_obs = len(self._dataframe)
        self.database = db.Database("shared_estimation", self._dataframe)
        self.lambda_val = self._estimate_lambda(observations)
    
    def estimate_mnl(self):
        """Estimate MNL model."""
        beta = Beta("beta", -0.01, -0.05, -1e-6, 0)
        V = {j: beta * Numeric(float(C.r[j])) for j in range(self.n)}
        AV = {j: Variable(f"AV_{j}") for j in range(self.n)}
        V[self.n] = Numeric(0.0)
        AV[self.n] = Numeric(1)

        logprob = models.loglogit(V, AV, Variable("CHOICE"))
        results = self._make_biogeme(self.database, logprob, "MNL").estimate()
        bv = results.get_beta_values()
        fit_stats = self._compute_fit_statistics(results)

        return {
            "beta": float(bv["beta"]),
            "lambda": self.lambda_val,
            **fit_stats,
            "success": True,
        }

    def estimate_mmnl(self, K=5, n_starts=5, random_seed=0, weight_sep_lambda=MMNL_WEIGHT_SEP_LAMBDA, weight_sep_alpha=MMNL_WEIGHT_SEP_ALPHA):
        """Estimate finite-mixture MMNL with K latent classes and free class betas."""
        beta_low, beta_high = BETA_BOUNDS
        min_weight = _min_weight_for_k(K)

        # Availability
        AV = {j: Variable(f"AV_{j}") for j in range(self.n)}
        AV[self.n] = Numeric(1)
        choice = Variable("CHOICE")

        init_betas = np.linspace(beta_low, beta_high, K)

        def build_model(start_values=None, return_x0_only=False):
            base_start_values = [
                *[float(init_betas[k]) for k in range(K)],
                *[0.0 for _ in range(K - 1)],
            ]
            free_bounds = [
                *[(beta_low, beta_high) for _ in range(K)],
                *[(None, None) for _ in range(K - 1)],
            ]
            if return_x0_only:
                return base_start_values, free_bounds

            start_values = np.asarray(base_start_values if start_values is None else start_values, dtype=float)
            idx = K
            beta_inits = [float(start_values[k]) for k in range(K)]
            logit_inits = [float(start_values[idx + j]) for j in range(K - 1)]

            # --- Segment-specific betas ---
            segment_betas = [
                Beta(
                    f"beta_{k+1}",
                    float(beta_inits[k]),
                    beta_low,
                    beta_high,
                    0,
                )
                for k in range(K)
            ]

            # --- Mixing weights with bounded logits to discourage collapse ---
            weight_logits = [
                Beta(f"wlogit_{k+1}", float(logit_inits[k]), None, None, 0)
                for k in range(K - 1)
            ]

            denom = Numeric(1.0)
            for lg in weight_logits:
                denom += exp(lg)

            simplex_weights = [exp(lg) / denom for lg in weight_logits] + [Numeric(1.0) / denom]
            slack = Numeric(float(1.0 - K * min_weight))
            floor = Numeric(float(min_weight))
            mixing_weights = [floor + slack * w for w in simplex_weights]

            mixture_prob = Numeric(0.0)
            for k in range(K):
                beta_k = segment_betas[k]
                V_k = {j: beta_k * Numeric(float(C.r[j])) for j in range(self.n)}
                V_k[self.n] = Numeric(0.0)
                class_prob = models.logit(V_k, AV, choice)
                mixture_prob += mixing_weights[k] * class_prob

            separation_penalty = Numeric(0.0)
            if weight_sep_lambda > 0.0:
                alpha_expr = Numeric(float(weight_sep_alpha))
                for i in range(K):
                    for j in range(i + 1, K):
                        diff = mixing_weights[i] - mixing_weights[j]
                        separation_penalty += exp(-alpha_expr * diff * diff)

            per_obs_penalty = Numeric(float(weight_sep_lambda / max(self.n_obs, 1))) * separation_penalty

            return None, None, log(mixture_prob) - per_obs_penalty, f"MMNL_{K}PT"

        results = self._estimate_multistart_biogeme(
            build_model,
            n_starts=n_starts,
            random_seed=random_seed,
        )
        bv = results.get_beta_values()
        fit_stats = self._compute_fit_statistics(results)

        # --- Recover estimated weights ---
        denom_w = 1.0 + sum(np.exp(bv[f"wlogit_{m+1}"]) for m in range(K - 1))
        simplex_weights = [
            *[
                float(np.exp(bv[f"wlogit_{k+1}"]) / denom_w)
                for k in range(K - 1)
            ],
            float(1.0 / denom_w),
        ]
        mixing_weights = [float(min_weight + (1.0 - K * min_weight) * w) for w in simplex_weights]

        betas = [float(bv[f"beta_{k+1}"]) for k in range(K)]

        return {
            "betas": betas,
            "n_segments": K,
            "mixing_weights": mixing_weights,
            "lambda": self.lambda_val,
            **fit_stats,
            "success": True,
        }

    def estimate_mmnl_continuous(self, n_starts=5, random_seed=0):
        """Continuous MMNL with lognormal price sensitivity.

        beta = -exp(mu_b + sigma_b * N(0,1)), so beta is always negative.
        Estimated via Biogeme MonteCarlo integration (only 2 free params,
        so the expression tree stays shallow and Biogeme handles it well).
        Returns mean_beta = -exp(mu_b + sigma_b^2/2) as summary statistic.
        """
        def build_model(start_values=None, return_x0_only=False):
            base_start_values = [np.log(0.01), 0.3]
            free_bounds = [(None, None), (0.0, None)]
            if return_x0_only:
                return base_start_values, free_bounds

            start_values = np.asarray(base_start_values if start_values is None else start_values, dtype=float)
            mu_b = Beta("mu_b", float(start_values[0]), None, None, 0)
            sigma_b = Beta("sigma_b", float(max(start_values[1], 0.0)), 0.0, None, 0)
            beta = -exp(mu_b + sigma_b * bioDraws("draw_normal", "NORMAL"))

            V = {j: beta * Numeric(float(C.r[j])) for j in range(self.n)}
            V[self.n] = Numeric(0.0)
            AV = {j: Variable(f"AV_{j}") for j in range(self.n)}
            AV[self.n] = Numeric(1)
            return None, None, log(MonteCarlo(models.logit(V, AV, Variable("CHOICE")))), "MMNL_cont"

        def _make_biogeme_with_draws(database, logprob, name):
            params = Parameters()
            params.set_value("optimization_algorithm", "scipy")
            params.set_value("save_iterations", False)
            params.set_value("number_of_draws", 500)
            model = bio.BIOGEME(
                database,
                logprob,
                parameters=params,
                generate_html=False,
                generate_yaml=False,
            )
            model.modelName = name
            return model

        results = self._estimate_multistart_biogeme(
            build_model,
            n_starts=n_starts,
            random_seed=random_seed,
            make_biogeme_fn=_make_biogeme_with_draws,
        )

        bv = results.get_beta_values()
        fit_stats = self._compute_fit_statistics(results)

        mu_hat = float(bv["mu_b"])
        sigma_hat = float(bv["sigma_b"])
        return {
            "mu_b": mu_hat,
            "sigma_b": sigma_hat,
            "mean_beta": float(-np.exp(mu_hat + 0.5 * sigma_hat ** 2)),
            "lambda": self.lambda_val,
            **fit_stats,
            "success": True,
        }

