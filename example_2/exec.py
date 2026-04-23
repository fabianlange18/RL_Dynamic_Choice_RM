# %%
import os
import gc
import time
import pickle
import logging

import config as c
import constants as C

_pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "")
if "cxx=" not in _pytensor_flags:
    os.environ["PYTENSOR_FLAGS"] = f"{_pytensor_flags},cxx=".strip(",")
logging.getLogger("pytensor.configdefaults").setLevel(logging.ERROR)

from estimation_biogeme import (
    estimate_mnl_biogeme,
    estimate_mmnl_biogeme,
    estimate_mmnl_twopoint_biogeme,
    estimate_mmnl_continuous_biogeme,
    collect_transaction_data,
)
from env_example_2 import TalluriExample2
from efficient_sets import compute_efficient_sets
from choice_dp import solve_by_dp
from train_rl import train_rl
from evaluation import evaluate_saved_models, print_evaluation_table
from simulation import simulate

# %%
os.makedirs(C.OUTPUT_DIR, exist_ok=True)
log_path = os.path.join(C.OUTPUT_DIR, "00_exec.log")

def log_message(message):
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")

log_message(f"Run {c.GT_MODEL} - High Sensitivity: {c.HIGH_SENSITIVITY}")
log_message(f"Training on all available sets (if false only on efficient sets): {c.TRAIN_ON_ALL_SETS}")
log_message(f"RL Training Seeds: {C.N_EVAL_EPISODES}, RL Training Steps: {C.TOTAL_TIMESTEPS}, Estimation Episodes: {C.N_ESTIMATION_EPISODES}\n")

# %%
# -- Shared observations: collect once, estimate both models ----------
t0 = time.perf_counter()
_env = TalluriExample2(efficient_sets=None)
observations = collect_transaction_data(_env)
_env.close()
sampling_time = time.perf_counter() - t0
log_message(f"Observation sampling time: {sampling_time:.4f} seconds")

t0 = time.perf_counter()
estimation_mnl_result = estimate_mnl_biogeme(observations)
estimation_mnl_time = time.perf_counter() - t0

t0 = time.perf_counter()
estimation_mmnl_5pt_result = estimate_mmnl_biogeme(observations)
estimation_mmnl_5pt_time = time.perf_counter() - t0

t0 = time.perf_counter()
estimation_mmnl_2pt_result = estimate_mmnl_twopoint_biogeme(observations)
estimation_mmnl_2pt_time = time.perf_counter() - t0

t0 = time.perf_counter()
estimation_mmnl_cont_result = estimate_mmnl_continuous_biogeme(observations)
estimation_mmnl_cont_time = time.perf_counter() - t0

beta_mnl   = estimation_mnl_result["beta"]
lambda_mnl = estimation_mnl_result["lambda"]
ll_mnl = estimation_mnl_result["final_log_likelihood"]
aic_mnl = estimation_mnl_result["aic"]
bic_mnl = estimation_mnl_result["bic"]

betas_mmnl_5pt = estimation_mmnl_5pt_result["betas"]
weights_mmnl_5pt = estimation_mmnl_5pt_result["mixing_weights"]
lambda_mmnl_5pt = estimation_mmnl_5pt_result["lambda"]
ll_mmnl_5pt = estimation_mmnl_5pt_result["final_log_likelihood"]
aic_mmnl_5pt = estimation_mmnl_5pt_result["aic"]
bic_mmnl_5pt = estimation_mmnl_5pt_result["bic"]

betas_mmnl_2pt = estimation_mmnl_2pt_result["betas"]
weights_mmnl_2pt = estimation_mmnl_2pt_result["mixing_weights"]
lambda_mmnl_2pt = estimation_mmnl_2pt_result["lambda"]
ll_mmnl_2pt = estimation_mmnl_2pt_result["final_log_likelihood"]
aic_mmnl_2pt = estimation_mmnl_2pt_result["aic"]
bic_mmnl_2pt = estimation_mmnl_2pt_result["bic"]

mu_mmnl_cont = estimation_mmnl_cont_result["mu_b"]
sigma_mmnl_cont = estimation_mmnl_cont_result["sigma_b"]
lambda_mmnl_cont = estimation_mmnl_cont_result["lambda"]
ll_mmnl_cont = estimation_mmnl_cont_result["final_log_likelihood"]
aic_mmnl_cont = estimation_mmnl_cont_result["aic"]
bic_mmnl_cont = estimation_mmnl_cont_result["bic"]

log_message(f"Estimation_MNL  time: {estimation_mnl_time:.4f}s | beta: {beta_mnl:.6f}, lambda: {lambda_mnl:.6f}")
log_message(f"  LL: {ll_mnl:.6f}, AIC: {aic_mnl:.3f}, BIC: {bic_mnl:.3f}\n")

log_message(f"Estimation_MMNL_5PT time: {estimation_mmnl_5pt_time:.4f}s | lambda: {lambda_mmnl_5pt:.6f}")
log_message(f"  LL: {ll_mmnl_5pt:.6f}, AIC: {aic_mmnl_5pt:.3f}, BIC: {bic_mmnl_5pt:.3f}")
log_message(f"Betas: {[f'{b:.6f}' for b in betas_mmnl_5pt]}")
log_message(f"Weights: {[f'{w:.6f}' for w in weights_mmnl_5pt]}\n")

log_message(f"Estimation_MMNL_2PT time: {estimation_mmnl_2pt_time:.4f}s | lambda: {lambda_mmnl_2pt:.6f}")
log_message(f"  LL: {ll_mmnl_2pt:.6f}, AIC: {aic_mmnl_2pt:.3f}, BIC: {bic_mmnl_2pt:.3f}")
log_message(f"Betas: {[f'{b:.6f}' for b in betas_mmnl_2pt]}")
log_message(f"Weights: {[f'{w:.6f}' for w in weights_mmnl_2pt]}\n")

log_message(f"Estimation_MMNL_CONT time: {estimation_mmnl_cont_time:.4f}s | mu: {mu_mmnl_cont:.6f}, sigma: {sigma_mmnl_cont:.6f}, lambda: {lambda_mmnl_cont:.6f}")
log_message(f"  LL: {ll_mmnl_cont:.6f}, AIC: {aic_mmnl_cont:.3f}, BIC: {bic_mmnl_cont:.3f}\n")


del observations
gc.collect()

# %%
# -- Efficient sets ---------------------------------------------------
if c.TRAIN_ON_ALL_SETS:
    efficient_sets_mnl  = None
    efficient_sets_mmnl_5pt = None
    efficient_sets_mmnl_2pt = None
    efficient_sets_mmnl_cont = None
    efficient_sets_rl = None
    efficient_sets_time_mnl  = 0.0
    efficient_sets_time_mmnl_5pt = 0.0
    efficient_sets_time_mmnl_2pt = 0.0
    efficient_sets_time_mmnl_cont = 0.0
else:
    t0 = time.perf_counter()
    efficient_sets_mnl = compute_efficient_sets(model="MNL", beta=beta_mnl)
    efficient_sets_time_mnl = time.perf_counter() - t0
    log_message(f"MNL  efficient sets time: {efficient_sets_time_mnl:.4f}s | sets: {efficient_sets_mnl}")

    t0 = time.perf_counter()
    efficient_sets_mmnl_5pt = compute_efficient_sets(
        model="MMNL_5PT",
        segment_betas=betas_mmnl_5pt,
        segment_weights=weights_mmnl_5pt,
    )
    efficient_sets_time_mmnl_5pt = time.perf_counter() - t0
    log_message(f"MMNL 5PT efficient sets time: {efficient_sets_time_mmnl_5pt:.4f}s | sets: {efficient_sets_mmnl_5pt}")

    t0 = time.perf_counter()
    efficient_sets_mmnl_2pt = compute_efficient_sets(
        model="MMNL_2PT",
        segment_betas=betas_mmnl_2pt,
        segment_weights=weights_mmnl_2pt,
    )
    efficient_sets_time_mmnl_2pt = time.perf_counter() - t0
    log_message(f"MMNL 2PT efficient sets time: {efficient_sets_time_mmnl_2pt:.4f}s | sets: {efficient_sets_mmnl_2pt}")

    t0 = time.perf_counter()
    efficient_sets_mmnl_cont = compute_efficient_sets(
        model="MMNLcont",
        mu_b=mu_mmnl_cont,
        sigma_b=sigma_mmnl_cont,
    )
    efficient_sets_time_mmnl_cont = time.perf_counter() - t0
    log_message(f"MMNL Cont efficient sets time: {efficient_sets_time_mmnl_cont:.4f}s | sets: {efficient_sets_mmnl_cont}\n")

efficient_sets_rl_candidates = [
    efficient_sets_mnl,
    efficient_sets_mmnl_5pt,
    efficient_sets_mmnl_2pt,
    efficient_sets_mmnl_cont,
]
efficient_sets_rl = None
for _sets in efficient_sets_rl_candidates:
    if _sets:
        efficient_sets_rl = set(_sets) if efficient_sets_rl is None else (efficient_sets_rl | set(_sets))

# %%
# -- DP solutions -----------------------------------------------------
t0 = time.perf_counter()
v_mnl, pi_mnl = solve_by_dp(
    efficient_sets=efficient_sets_mnl,
    estimated_beta=beta_mnl,
    estimated_lambda=lambda_mnl,
    model="MNL",
)
dp_mnl_time = time.perf_counter() - t0
log_message(f"DP_MNL  time: {dp_mnl_time:.4f}s | V(0,C): {v_mnl[0, C.C]:.2f}")

t0 = time.perf_counter()
v_mmnl_5pt, pi_mmnl_5pt = solve_by_dp(
    efficient_sets=efficient_sets_mmnl_5pt,
    estimated_beta=None,
    estimated_lambda=lambda_mmnl_5pt,
    model="MMNL_5PT",
    segment_betas=betas_mmnl_5pt,
    segment_weights=weights_mmnl_5pt,
)
dp_mmnl_5pt_time = time.perf_counter() - t0
log_message(f"DP_MMNL_5PT time: {dp_mmnl_5pt_time:.4f}s | V(0,C): {v_mmnl_5pt[0, C.C]:.2f}")

t0 = time.perf_counter()
v_mmnl_2pt, pi_mmnl_2pt = solve_by_dp(
    efficient_sets=efficient_sets_mmnl_2pt,
    estimated_beta=None,
    estimated_lambda=lambda_mmnl_2pt,
    model="MMNL_2PT",
    segment_betas=betas_mmnl_2pt,
    segment_weights=weights_mmnl_2pt,
)
dp_mmnl_2pt_time = time.perf_counter() - t0
log_message(f"DP_MMNL_2PT time: {dp_mmnl_2pt_time:.4f}s | V(0,C): {v_mmnl_2pt[0, C.C]:.2f}")

t0 = time.perf_counter()
v_mmnl_cont, pi_mmnl_cont = solve_by_dp(
    efficient_sets=efficient_sets_mmnl_cont,
    estimated_beta=None,
    estimated_lambda=lambda_mmnl_cont,
    model="MMNLcont",
    mu_b=mu_mmnl_cont,
    sigma_b=sigma_mmnl_cont,
)
dp_mmnl_cont_time = time.perf_counter() - t0
log_message(f"DP_MMNL_CONT time: {dp_mmnl_cont_time:.4f}s | V(0,C): {v_mmnl_cont[0, C.C]:.2f}\n")

time_results = {
    "Sampling":                sampling_time,
    "Estimation_MNL":          estimation_mnl_time,
    "Estimation_MMNL_5PT":     estimation_mmnl_5pt_time,
    "Estimation_MMNL_2PT":     estimation_mmnl_2pt_time,
    "Estimation_MMNL_CONT":    estimation_mmnl_cont_time,
    "EfficientSets_MNL":       efficient_sets_time_mnl,
    "EfficientSets_MMNL_5PT":  efficient_sets_time_mmnl_5pt,
    "EfficientSets_MMNL_2PT":  efficient_sets_time_mmnl_2pt,
    "EfficientSets_MMNL_CONT": efficient_sets_time_mmnl_cont,
    "DP_MNL":                  dp_mnl_time,
    "DP_MMNL_5PT":             dp_mmnl_5pt_time,
    "DP_MMNL_2PT":             dp_mmnl_2pt_time,
    "DP_MMNL_CONT":            dp_mmnl_cont_time,
}

# %%
# RL always trains against MNL policy using MNL efficient sets
training_times_by_step = train_rl(dp_pi=pi_mnl, efficient_sets=efficient_sets_rl)

# %%
for step in C.TOTAL_TIMESTEPS:
    training_times_by_step[step].append(
        {
            "DP_MNL": dp_mnl_time,
            "DP_MMNL_5PT": dp_mmnl_5pt_time,
            "DP_MMNL_2PT": dp_mmnl_2pt_time,
            "DP_MMNL_CONT": dp_mmnl_cont_time,
        }
    )

# %%
dp_policy_configs = {
    "DP_MNL": {
        "pi": pi_mnl,
        "efficient_sets": efficient_sets_mnl,
    },
    "DP_MMNL_5PT": {
        "pi": pi_mmnl_5pt,
        "efficient_sets": efficient_sets_mmnl_5pt,
    },
    "DP_MMNL_2PT": {
        "pi": pi_mmnl_2pt,
        "efficient_sets": efficient_sets_mmnl_2pt,
    },
    "DP_MMNL_CONT": {
        "pi": pi_mmnl_cont,
        "efficient_sets": efficient_sets_mmnl_cont,
    },
}

evaluation_results_by_step = evaluate_saved_models(
    dp_policy_configs=dp_policy_configs,
    rl_efficient_sets=efficient_sets_rl,
)

# %%
for step in C.TOTAL_TIMESTEPS:
    table_str = print_evaluation_table(training_times_by_step[step], evaluation_results_by_step[step])
    log_message(f"\n=== {step:,} training timesteps ===\n{table_str}")

# %%
data = {
    "beta_mnl": beta_mnl,
    "lambda_mnl": lambda_mnl,
    "betas_mmnl_5pt": betas_mmnl_5pt,
    "weights_mmnl_5pt": weights_mmnl_5pt,
    "betas_mmnl_2pt": betas_mmnl_2pt,
    "weights_mmnl_2pt": weights_mmnl_2pt,
    "mu_mmnl_cont": mu_mmnl_cont,
    "sigma_mmnl_cont": sigma_mmnl_cont,
    "lambda_mmnl_5pt": lambda_mmnl_5pt,
    "lambda_mmnl_2pt": lambda_mmnl_2pt,
    "lambda_mmnl_cont": lambda_mmnl_cont,
    "efficient_sets_mnl": efficient_sets_mnl,
    "efficient_sets_mmnl_5pt": efficient_sets_mmnl_5pt,
    "efficient_sets_mmnl_2pt": efficient_sets_mmnl_2pt,
    "efficient_sets_mmnl_cont": efficient_sets_mmnl_cont,
    "efficient_sets_rl": efficient_sets_rl,
    "v_mnl": v_mnl,
    "pi_mnl": pi_mnl,
    "v_mmnl_5pt": v_mmnl_5pt,
    "pi_mmnl_5pt": pi_mmnl_5pt,
    "v_mmnl_2pt": v_mmnl_2pt,
    "pi_mmnl_2pt": pi_mmnl_2pt,
    "v_mmnl_cont": v_mmnl_cont,
    "pi_mmnl_cont": pi_mmnl_cont,
    "time_results": time_results,
    # "training_times_by_step": training_times_by_step,
    # "evaluation_results_by_step": evaluation_results_by_step,
}

with open(f"{C.OUTPUT_DIR}/results.pkl", "wb") as f:
    pickle.dump(data, f)

# %%
import pickle
from simulation import simulate
import constants as C

with open(f"{C.OUTPUT_DIR}/results.pkl", "rb") as f:
    _cached = pickle.load(f)

efficient_sets_mnl = _cached["efficient_sets_mnl"]
pi_mnl = _cached["pi_mnl"]

rewards = []

for i in range(1000):
    reward, _ = simulate(efficient_sets_mnl, pi_mnl, seed=i)
    rewards.append(reward)

log_message(f"Average reward over {len(rewards)} episodes: {sum(rewards) / len(rewards):.2f}")
# %%
