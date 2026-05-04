import os
import pickle

import config as c
import constants as C
from evaluation import evaluate_saved_models, print_evaluation_table
from train_rl import train_rl


def log_message(message):
    """Log a message to the RL-only execution log."""
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")


def main():
    """Run only RL training for the large-product experiment setup."""
    if not c.LARGE_PRODUCT_SET:
        raise ValueError("exec_RL.py is intended for LARGE_PRODUCT_SET=True.")

    task_id = os.environ.get("TASK_ID", "N/A")
    log_message(f"\n{'='*60}")
    log_message(f"TASK_ID: {task_id}")
    log_message(f"LARGE_PRODUCT_SET: {c.LARGE_PRODUCT_SET}")
    log_message(f"TRAIN_ON_ALL_SETS: {c.TRAIN_ON_ALL_SETS}")
    log_message(f"HIGH_SENSITIVITY: {c.HIGH_SENSITIVITY}")
    log_message(f"GT_MODEL: {c.GT_MODEL}")
    log_message(f"{'='*60}")
    log_message(
        f"RL Training Seeds: {C.N_EVAL_EPISODES}, RL Training Steps: {C.TOTAL_TIMESTEPS}"
    )
    log_message("RL-only mode: skipping estimation, efficient sets, DP, and evaluation.\n")

    efficient_sets_rl = None
    training_times_by_step = train_rl(efficient_sets=efficient_sets_rl)
    evaluation_results_by_step = evaluate_saved_models(
        dp_policy_configs={},
        rl_efficient_sets=efficient_sets_rl,
    )

    for step in C.TOTAL_TIMESTEPS:
        log_message(f"RL-only training finished for {step:,} timesteps: {training_times_by_step[step]}")
        table_str = print_evaluation_table(training_times_by_step[step], evaluation_results_by_step[step])
        log_message(f"\n=== {step:,} training timesteps ===\n{table_str}")

    data = {
        "efficient_sets_rl": efficient_sets_rl,
        "training_times_by_step": training_times_by_step,
        "evaluation_results_by_step": evaluation_results_by_step,
    }

    with open(f"{C.OUTPUT_DIR}/results.pkl", "wb") as file_handle:
        pickle.dump(data, file_handle)


if __name__ == "__main__":
    os.makedirs(C.OUTPUT_DIR, exist_ok=True)
    log_path = os.path.join(C.OUTPUT_DIR, "00_exec_RL.log")
    main()