# %%
import os
import time
import pickle

import config as c
import constants as C
from em_estimation import run_em
from efficient_sets import compute_efficient_sets
from choice_dp import solve_by_dp
from train_rl import train_rl
from evaluation import evaluate_saved_models, print_evaluation_table


os.makedirs(C.OUTPUT_DIR, exist_ok=True)
log_path = os.path.join(C.OUTPUT_DIR, "00_exec.log")

def log_message(message):
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")

log_message(f"Run {c.GT_MODEL} - {c.OPT_MODEL} - High Sensitivity: {c.HIGH_SENSITIVITY}")
log_message(f"Train Runs: {c.N_TRAIN_RUNS}, Training on all available sets (if false only on efficient sets): {c.TRAIN_ON_ALL_SETS}")
log_message(f"RL Training Steps: {C.TOTAL_TIMESTEPS}, Evaluation Episodes: {C.N_EVAL_EPISODES}, Estimation Episodes: {C.N_ESTIMATION_EPISODES}\n")

# %%
t0 = time.perf_counter()
em_result = run_em()
em_time = time.perf_counter() - t0

beta_est, lambda_est = em_result["em_result"]["beta"], em_result["em_result"]["lambda"]

log_message(f"EM calculation time: {em_time:.4f} seconds")
log_message(f"Estimated beta: {beta_est}, Estimated lambda: {lambda_est}\n")

# %%
if c.TRAIN_ON_ALL_SETS:
    efficient_sets = None
else:
    t0 = time.perf_counter()
    efficient_sets = compute_efficient_sets(beta=beta_est)
    efficient_sets_time = time.perf_counter() - t0
    log_message(f"Efficient sets calculation time: {efficient_sets_time:.4f} seconds")
    log_message(f"Efficient sets: {efficient_sets}\n")

# %%
t0 = time.perf_counter()
v, pi = solve_by_dp(
    efficient_sets=efficient_sets,
    estimated_beta=beta_est,
    estimated_lambda=lambda_est,
)
dp_time = time.perf_counter() - t0
log_message(f"DP calculation time: {dp_time:.4f} seconds")
log_message(f"DP Value at (0, C): {v[0, C]}\n")

time_results = {
    "EM": em_time,
    "EfficientSets": efficient_sets_time,
    "DP": dp_time,
}

# %%
training_times = train_rl(dp_pi=pi, efficient_sets=efficient_sets)

# %%
training_times.append({'DP': dp_time})
# %%
evaluation_results = evaluate_saved_models(pi=pi, efficient_sets=efficient_sets)

# %%
log_message(print_evaluation_table(training_times, evaluation_results))

# %%
data = {
    "beta_est": beta_est,
    "lambda_est": lambda_est,
    "efficient_sets": efficient_sets,
    "v": v,
    "pi": pi,
    "time_results": time_results,
    "training_times": training_times,
    "evaluation_results": evaluation_results,
}

with open(f"{C.OUTPUT_DIR}/results.pkl", "wb") as f:
    pickle.dump(data, f)