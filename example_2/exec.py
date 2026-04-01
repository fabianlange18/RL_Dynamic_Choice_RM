# %%
import os
import time
import pickle

import numpy as np

import config as c
import constants as C
from em_estimation import (
    estimate_mnl_em,
    estimate_mmnl_em,
    collect_incomplete_transaction_data,
)
from env_example_2 import TalluriExample2
from efficient_sets import compute_efficient_sets
from choice_dp import solve_by_dp
from train_rl import train_rl
from evaluation import evaluate_saved_models, print_evaluation_table

# %%
os.makedirs(C.OUTPUT_DIR, exist_ok=True)
log_path = os.path.join(C.OUTPUT_DIR, "00_exec.log")

def log_message(message):
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")

log_message(f"Run {c.GT_MODEL} - High Sensitivity: {c.HIGH_SENSITIVITY}")
log_message(f"Train Runs: {c.N_TRAIN_RUNS}, Training on all available sets (if false only on efficient sets): {c.TRAIN_ON_ALL_SETS}")
log_message(f"RL Training Steps: {C.TOTAL_TIMESTEPS}, Evaluation Episodes: {C.N_EVAL_EPISODES}, Estimation Episodes: {C.N_ESTIMATION_EPISODES}\n")

# %%
# -- Shared observations: collect once, estimate both models ----------
t0 = time.perf_counter()
_env = TalluriExample2(efficient_sets=None)
observations = collect_incomplete_transaction_data(_env)
_env.close()
sampling_time = time.perf_counter() - t0
log_message(f"Observation sampling time: {sampling_time:.4f} seconds")

t0 = time.perf_counter()
em_mnl_result = estimate_mnl_em(observations)
em_mnl_time = time.perf_counter() - t0

t0 = time.perf_counter()
em_mmnl_result = estimate_mmnl_em(observations)
em_mmnl_time = time.perf_counter() - t0

beta_mnl   = em_mnl_result["beta"]
lambda_mnl = em_mnl_result["lambda"]
betas_mmnl  = em_mmnl_result["betas"]
lambda_mmnl = em_mmnl_result["lambda"]

log_message(f"EM_MNL  time: {em_mnl_time:.4f}s | beta: {beta_mnl:.6f}, lambda: {lambda_mnl:.6f}")
log_message(f"EM_MMNL time: {em_mmnl_time:.4f}s | betas: {[f'{b:.6f}' for b in betas_mmnl]}, lambda: {lambda_mmnl:.6f}\n")

# %%
# -- Efficient sets ---------------------------------------------------
if c.TRAIN_ON_ALL_SETS:
    efficient_sets_mnl  = None
    efficient_sets_mmnl = None
    efficient_sets_time_mnl  = 0.0
    efficient_sets_time_mmnl = 0.0
else:
    t0 = time.perf_counter()
    efficient_sets_mnl = compute_efficient_sets(model="MNL", beta=beta_mnl)
    efficient_sets_time_mnl = time.perf_counter() - t0
    log_message(f"MNL  efficient sets time: {efficient_sets_time_mnl:.4f}s | sets: {efficient_sets_mnl}")

    t0 = time.perf_counter()
    efficient_sets_mmnl = compute_efficient_sets(model="MMNL", segment_betas=betas_mmnl)
    efficient_sets_time_mmnl = time.perf_counter() - t0
    log_message(f"MMNL efficient sets time: {efficient_sets_time_mmnl:.4f}s | sets: {efficient_sets_mmnl}\n")

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
v_mmnl, pi_mmnl = solve_by_dp(
    efficient_sets=efficient_sets_mmnl,
    estimated_beta=None,
    estimated_lambda=lambda_mmnl,
    model="MMNL",
    segment_betas=betas_mmnl,
)
dp_mmnl_time = time.perf_counter() - t0
log_message(f"DP_MMNL time: {dp_mmnl_time:.4f}s | V(0,C): {v_mmnl[0, C.C]:.2f}\n")

time_results = {
    "Sampling":           sampling_time,
    "EM_MNL":             em_mnl_time,
    "EM_MMNL":            em_mmnl_time,
    "EfficientSets_MNL":  efficient_sets_time_mnl,
    "EfficientSets_MMNL": efficient_sets_time_mmnl,
    "DP_MNL":             dp_mnl_time,
    "DP_MMNL":            dp_mmnl_time,
}

# Cumulative wall-clock time to produce each DP policy
dp_mnl_total  = sampling_time + em_mnl_time  + efficient_sets_time_mnl  + dp_mnl_time
dp_mmnl_total = sampling_time + em_mmnl_time + efficient_sets_time_mmnl + dp_mmnl_time
log_message(
    f"Cumulative time — DP_MNL: {dp_mnl_total:.4f}s | DP_MMNL: {dp_mmnl_total:.4f}s\n"
)

# %%
# RL always trains against MNL policy using MNL efficient sets
training_times_by_step = train_rl(dp_pi=pi_mnl, efficient_sets=efficient_sets_mnl)

# %%
for step in C.TOTAL_TIMESTEPS:
    training_times_by_step[step].append({"DP_MNL": dp_mnl_time, "DP_MMNL": dp_mmnl_time})

# %%
evaluation_results_by_step = evaluate_saved_models(
    dp_policies={"DP_MNL": pi_mnl},
    efficient_sets=efficient_sets_mnl,
    additional_dp_policies={"DP_MMNL": pi_mmnl},
    additional_efficient_sets=efficient_sets_mmnl,
)

# %%
for step in C.TOTAL_TIMESTEPS:
    table_str = print_evaluation_table(training_times_by_step[step], evaluation_results_by_step[step])
    log_message(f"\n=== {step:,} training timesteps ===\n{table_str}")

# %%
data = {
    "beta_mnl": beta_mnl,
    "lambda_mnl": lambda_mnl,
    "betas_mmnl": betas_mmnl,
    "lambda_mmnl": lambda_mmnl,
    "efficient_sets_mnl": efficient_sets_mnl,
    "efficient_sets_mmnl": efficient_sets_mmnl,
    "v_mnl": v_mnl,
    "pi_mnl": pi_mnl,
    "v_mmnl": v_mmnl,
    "pi_mmnl": pi_mmnl,
    "time_results": time_results,
    "training_times_by_step": training_times_by_step,
    "evaluation_results_by_step": evaluation_results_by_step,
}

with open(f"{C.OUTPUT_DIR}/results.pkl", "wb") as f:
    pickle.dump(data, f)