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

BETA_BOUNDS = (-0.05, -1e-6)

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


def _estimate_lambda(observations):
    return float(np.mean([obs["arrival_flag"] for obs in observations]))


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


def _compute_fit_statistics(results):
    stats = results.get_general_statistics()
    final_log_likelihood = stats["Final log likelihood"]
    aic = stats["Akaike Information Criterion"]
    bic = stats["Bayesian Information Criterion"]

    return {
        "final_log_likelihood": float(final_log_likelihood),
        "aic": float(aic),
        "bic": float(bic)
    }


def _make_biogeme(database, logprob, name):
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


def estimate_mnl_biogeme(observations):
    n = len(C.r)
    df = _build_dataframe(observations)
    database = db.Database("mnl", df)

    beta = Beta("beta", -0.01, -0.05, -1e-6, 0)
    V = {j: beta * Numeric(float(C.r[j])) for j in range(n)}
    AV = {j: Variable(f"AV_{j}") for j in range(n)}
    V[n] = Numeric(0.0)
    AV[n] = Numeric(1)

    logprob = models.loglogit(V, AV, Variable("CHOICE"))
    results = _make_biogeme(database, logprob, "MNL").estimate()
    bv = results.get_beta_values()
    fit_stats = _compute_fit_statistics(results)

    return {
        "beta": float(bv["beta"]),
        "lambda": _estimate_lambda(observations),
        **fit_stats,
        "success": True,
    }

def estimate_mmnl_biogeme(observations, K=5):
    """
    Finite-mixture MMNL with K latent classes.
    Fixes:
      (1) Class-specific ASCs
      (2) One beta fixed for scale normalization
    """
    n = len(C.r)
    beta_low, beta_high = BETA_BOUNDS

    df = _build_dataframe(observations)
    database = db.Database("mmnl", df)

    # Availability
    AV = {j: Variable(f"AV_{j}") for j in range(n)}
    AV[n] = Numeric(1)

    choice = Variable("CHOICE")

    # --- Segment-specific betas (beta_1 fixed) ---
    init_betas = np.linspace(beta_low, beta_high, K)
    segment_betas = [
        Beta(
            f"beta_{k+1}",
            float(init_betas[k]),
            beta_low,
            beta_high,
            int(k == 0),  # FIX beta_1
        )
        for k in range(K)
    ]

    # --- Class-specific ASCs (ASC_1,1 fixed) ---
    segment_ascs = [
        Beta(
            f"asc_{k+1}",
            0.0,
            None,
            None,
            int(k == 0),  # FIX first ASC
        )
        for k in range(K)
    ]

    # --- Mixing weights: softmax, last class normalized ---
    weight_logits = [
        Beta(f"wlogit_{k+1}", 0.0, None, None, 0)
        for k in range(K - 1)
    ]

    denom = Numeric(1.0)
    for lg in weight_logits:
        denom += exp(lg)

    mixing_weights = (
        [exp(lg) / denom for lg in weight_logits]
        + [Numeric(1.0) / denom]
    )

    # --- Mixture probability ---
    mixture_prob = Numeric(0.0)

    for k in range(K):
        beta_k = segment_betas[k]
        asc_k = segment_ascs[k]

        # Utility for class k
        V_k = {
            j: asc_k + beta_k * Numeric(float(C.r[j]))
            for j in range(n)
        }
        V_k[n] = Numeric(0.0)  # outside option

        class_prob = models.logit(V_k, AV, choice)
        mixture_prob += mixing_weights[k] * class_prob

    # Log-likelihood
    logprob = log(mixture_prob)

    results = _make_biogeme(database, logprob, "MMNL_5PT").estimate()
    bv = results.get_beta_values()
    fit_stats = _compute_fit_statistics(results)

    # --- Recover estimated weights ---
    denom_w = 1.0 + sum(
        np.exp(bv[f"wlogit_{m+1}"]) for m in range(K - 1)
    )

    
    betas = [None] * K

    # beta_1 ist fixiert → Initialwert
    betas[0] = float(init_betas[0])

    # restliche Betas kommen aus Biogeme
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
        "lambda": _estimate_lambda(observations),
        **fit_stats,
        "success": True,
    }



def estimate_mmnl_continuous_biogeme(observations):
    """Continuous MMNL with lognormal price sensitivity.

    beta = -exp(mu_b + sigma_b * N(0,1)), so beta is always negative.
    Estimated via Biogeme MonteCarlo integration (only 2 free params,
    so the expression tree stays shallow and Biogeme handles it well).
    Returns mean_beta = -exp(mu_b + sigma_b^2/2) as summary statistic.
    """
    n = len(C.r)
    df = _build_dataframe(observations)
    database = db.Database("mmnl_cont", df)

    mu_b = Beta("mu_b", np.log(0.01), None, None, 0)
    sigma_b = Beta("sigma_b", 0.3, 0.0, None, 0)
    beta = -exp(mu_b + sigma_b * bioDraws("draw_normal", "NORMAL"))

    V = {j: beta * Numeric(float(C.r[j])) for j in range(n)}
    V[n] = Numeric(0.0)
    AV = {j: Variable(f"AV_{j}") for j in range(n)}
    AV[n] = Numeric(1)

    logprob = log(MonteCarlo(models.logit(V, AV, Variable("CHOICE"))))

    params = Parameters()
    params.set_value("optimization_algorithm", "scipy")
    params.set_value("save_iterations", False)
    params.set_value("number_of_draws", 500)
    m = bio.BIOGEME(
        database,
        logprob,
        parameters=params,
        generate_html=False,
        generate_yaml=False,
    )
    m.modelName = "MMNL_cont"
    results = m.estimate()
    bv = results.get_beta_values()
    fit_stats = _compute_fit_statistics(results)

    mu_hat = float(bv["mu_b"])
    sigma_hat = float(bv["sigma_b"])
    return {
        "mu_b": mu_hat,
        "sigma_b": sigma_hat,
        "mean_beta": float(-np.exp(mu_hat + 0.5 * sigma_hat ** 2)),
        "lambda": _estimate_lambda(observations),
        **fit_stats,
        "success": True,
    }


def estimate_mmnl_twopoint_biogeme(observations):
    """Two-point mixture MMNL: two freely estimated segment betas with one mixing weight."""
    n = len(C.r)
    beta_low, beta_high = BETA_BOUNDS
    df = _build_dataframe(observations)
    database = db.Database("mmnl_2pt", df)

    AV = {j: Variable(f"AV_{j}") for j in range(n)}
    AV[n] = Numeric(1)
    choice = Variable("CHOICE")

    beta_1 = Beta("beta_1", -0.005, beta_low, beta_high, 0)
    beta_2 = Beta("beta_2", -0.02, beta_low, beta_high, 0)
    w_logit = Beta("w_logit", 0.0, None, None, 0)

    w1 = exp(w_logit) / (Numeric(1.0) + exp(w_logit))
    w2 = Numeric(1.0) / (Numeric(1.0) + exp(w_logit))

    V_1 = {j: beta_1 * Numeric(float(C.r[j])) for j in range(n)}
    V_1[n] = Numeric(0.0)
    V_2 = {j: beta_2 * Numeric(float(C.r[j])) for j in range(n)}
    V_2[n] = Numeric(0.0)

    mixture_prob = w1 * models.logit(V_1, AV, choice) + w2 * models.logit(V_2, AV, choice)
    logprob = log(mixture_prob)
    results = _make_biogeme(database, logprob, "MMNL_2PT").estimate()
    bv = results.get_beta_values()
    fit_stats = _compute_fit_statistics(results)

    w_val = float(np.exp(bv["w_logit"]) / (1.0 + np.exp(bv["w_logit"])))
    return {
        "betas": [float(bv["beta_1"]), float(bv["beta_2"])],
        "n_segments": 2,
        "mixing_weights": [w_val, 1.0 - w_val],
        "lambda": _estimate_lambda(observations),
        **fit_stats,
        "success": True,
    }