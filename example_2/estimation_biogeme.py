import logging

import numpy as np
import pandas as pd

import biogeme.database as db
import biogeme.biogeme as bio
from biogeme import models
from biogeme.expressions import Beta, Variable, log, exp, Numeric, bioDraws, MonteCarlo
from biogeme.parameters import Parameters

import constants as C

logging.getLogger("biogeme").setLevel(logging.ERROR)

BETA_BOUNDS = (-0.05, -1e-7)

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
        self.database = db.Database("shared_estimation", self._dataframe)
        self.lambda_val = self._estimate_lambda(observations)
    
    def estimate_mnl(self):
        """Estimate MNL model."""
        beta_low, beta_high = BETA_BOUNDS
        beta = Beta("beta", -0.01, beta_low, beta_high, 0)
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

    def estimate_mmnl(self, K=5):
        """Estimate finite-mixture MMNL with K latent classes."""

        AV = {j: Variable(f"AV_{j}") for j in range(self.n)}
        AV[self.n] = Numeric(1)
        choice = Variable("CHOICE")

        beta_low, beta_high = BETA_BOUNDS
        init_betas = np.linspace(beta_low, beta_high, K)
        segment_betas = [
            Beta(f"beta_{k+1}", float(init_betas[k]), beta_low, beta_high, 0)
            for k in range(K)
        ]

        logit_bound = float(np.log(K))
        weight_logits = [
            Beta(f"wlogit_{k+1}", 0.0, -logit_bound, logit_bound, 0)
            for k in range(K - 1)
        ]

        denom = Numeric(1.0)
        for lg in weight_logits:
            denom += exp(lg)

        mixing_weights = (
            [exp(lg) / denom for lg in weight_logits]
            + [Numeric(1.0) / denom]
        )

        mixture_prob = Numeric(0.0)
        for k in range(K):
            V_k = {j: segment_betas[k] * Numeric(float(C.r[j])) for j in range(self.n)}
            V_k[self.n] = Numeric(0.0)
            mixture_prob += mixing_weights[k] * models.logit(V_k, AV, choice)

        # Log-likelihood
        logprob = log(mixture_prob)

        results = self._make_biogeme(self.database, logprob, "MMNL_5PT").estimate()
        bv = results.get_beta_values()
        fit_stats = self._compute_fit_statistics(results)

        # --- Recover estimated weights ---
        denom_w = 1.0 + sum(
            np.exp(bv[f"wlogit_{m+1}"]) for m in range(K - 1)
        )

        betas = [None] * K
        betas[0] = float(init_betas[0])
        for k in range(1, K):
            betas[k] = float(bv[f"beta_{k+1}"])

        return {
            "betas": betas,
            "n_segments": K,
            "mixing_weights": [
                *[
                    float(np.exp(bv[f"wlogit_{k+1}"]) / denom_w)
                    for k in range(K - 1)
                ],
                float(1.0 / denom_w),
            ],
            "lambda": self.lambda_val,
            **fit_stats,
            "success": True,
        }

    def estimate_mmnl_continuous(self):
        """Continuous MMNL with lognormal price sensitivity.

        beta = -exp(mu_b + sigma_b * N(0,1)), so beta is always negative.
        Estimated via Biogeme MonteCarlo integration (only 2 free params,
        so the expression tree stays shallow and Biogeme handles it well).
        Returns mean_beta = -exp(mu_b + sigma_b^2/2) as summary statistic.
        """
        mu_b = Beta("mu_b", np.log(0.01), None, None, 0)
        sigma_b = Beta("sigma_b", 0.2, 0.0, None, 0)
        beta = -exp(mu_b + sigma_b * bioDraws("draw_normal", "NORMAL"))

        V = {j: beta * Numeric(float(C.r[j])) for j in range(self.n)}
        V[self.n] = Numeric(0.0)
        AV = {j: Variable(f"AV_{j}") for j in range(self.n)}
        AV[self.n] = Numeric(1)

        logprob = log(MonteCarlo(models.logit(V, AV, Variable("CHOICE"))))

        params = Parameters()
        params.set_value("optimization_algorithm", "scipy")
        params.set_value("save_iterations", False)
        params.set_value("number_of_draws", 500)
        m = bio.BIOGEME(
            self.database,
            logprob,
            parameters=params,
            generate_html=False,
            generate_yaml=False,
        )
        m.modelName = "MMNL_cont"
        results = m.estimate()
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
