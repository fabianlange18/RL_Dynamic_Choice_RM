import os
import re
import numpy as np

from simulation import simulate
import constants as C
import config as c

def evaluate_saved_models(pi, efficient_sets=None):
    evaluation_results = {}
    for algorithm_name in C.RL_ALGORITHMS.keys():
        for run in range(1, c.N_TRAIN_RUNS + 1):
            model_path = os.path.join(C.OUTPUT_DIR, f"{algorithm_name}_model_run{run}.zip")
            if os.path.exists(model_path):
                model = C.RL_ALGORITHMS[algorithm_name].load(model_path)
                rewards = []
                load_factors = []
                for seed in range(C.N_EVAL_EPISODES):
                    reward, load_factor = simulate(efficient_sets=efficient_sets, model=model, seed=seed)
                    rewards.append(reward)
                    load_factors.append(load_factor)
                evaluation_results[f"{algorithm_name}_run{run}"] = {
                    "mean_reward": np.mean(rewards),
                    "std_reward": np.std(rewards),
                    "mean_load_factor": np.mean(load_factors),
                    "std_load_factor": np.std(load_factors),

                }
            else:
                Warning(f"Model file not found for {algorithm_name}: {model_path}")

    dp_rewards = []
    dp_load_factors = []
    for seed in range(C.N_EVAL_EPISODES):
        reward, load_factor = simulate(efficient_sets=efficient_sets, pi=pi, seed=seed)
        dp_rewards.append(reward)
        dp_load_factors.append(load_factor)
    evaluation_results["DP"] = {
        "mean_reward": np.mean(dp_rewards),
        "std_reward": np.std(dp_rewards),
        "mean_load_factor": np.mean(dp_load_factors),
        "std_load_factor": np.std(dp_load_factors),
    }

    return evaluation_results


def print_evaluation_table(training_times, evaluation_results):
    
    methods = list(training_times[0].keys())

    # Collect run-level metrics per method from keys like "DQN_run1"
    run_reward_means = {m: [] for m in methods}
    run_load_means = {m: [] for m in methods}

    run_key_pattern = re.compile(r"^(?P<algo>.+)_run\d+$")
    for key, vals in evaluation_results.items():
        match = run_key_pattern.match(key)
        if not match:
            continue
        algo = match.group("algo")
        if algo not in run_reward_means:
            run_reward_means[algo] = []
            run_load_means[algo] = []
            methods.append(algo)
        run_reward_means[algo].append(float(vals["mean_reward"]))
        run_load_means[algo].append(float(vals["mean_load_factor"]))

    # Get DP reward for percentage calculation
    dp_reward = float(evaluation_results["DP"]["mean_reward"]) if "DP" in evaluation_results else None

    # Append DP as a method row (no train time)
    if "DP" in evaluation_results:
        methods.append("DP")

    # Build rows
    rows = []
    for method in methods:
        # Training times across runs (not available for DP)
        t_vals = [float(run_dict[method]) for run_dict in training_times if method in run_dict]
        t_mean = float(np.mean(t_vals)) if t_vals else np.nan
        t_std = float(np.std(t_vals)) if t_vals else np.nan

        # Rewards/load factors from run-level means
        if method == "DP":
            # Single summary entry for DP
            r_vals = [float(evaluation_results["DP"]["mean_reward"])]
            l_vals = [float(evaluation_results["DP"]["mean_load_factor"])]
            pct_dp_vals = [100.0]  # DP is 100% of itself
        else:
            r_vals = run_reward_means.get(method, [])
            l_vals = run_load_means.get(method, [])
            # Calculate percentage of DP reached for each run
            if dp_reward is not None and dp_reward != 0:
                pct_dp_vals = [(r / dp_reward) * 100 for r in r_vals]
            else:
                pct_dp_vals = []

        r_mean = float(np.mean(r_vals)) if r_vals else np.nan
        r_std = float(np.std(r_vals)) if r_vals else np.nan
        l_mean = float(np.mean(l_vals)) if l_vals else np.nan
        l_std = float(np.std(l_vals)) if l_vals else np.nan
        pct_dp_mean = float(np.mean(pct_dp_vals)) if pct_dp_vals else np.nan
        pct_dp_std = float(np.std(pct_dp_vals)) if pct_dp_vals else np.nan

        rows.append(
            {
                "Method": method,
                "TrainTimeMean(s)": t_mean,
                "TrainTimeStd(s)": t_std,
                "RewardMean": r_mean,
                "RewardStdAcrossRuns": r_std,
                "PctOfDP(%)": pct_dp_mean,
                "PctOfDPStd(%)": pct_dp_std,
                "LoadFactorMean(%)": l_mean,
                "LoadFactorStdAcrossRuns(%)": l_std,
            }
        )

    # Print table (row per method)
    headers = [
        "Method",
        "TrainTimeMean(s)",
        "TrainTimeStd(s)",
        "RewardMean",
        "RewardStdAcrossRuns",
        "PctOfDP(%)",
        "PctOfDPStd(%)",
        "LoadFactorMean(%)",
        "LoadFactorStdAcrossRuns(%)",
    ]

    def _fmt(v):
        if isinstance(v, str):
            return v
        if np.isnan(v):
            return "-"
        return f"{v:.2f}"

    # Compute column widths
    col_widths = {}
    for h in headers:
        max_cell = max([len(_fmt(row[h])) for row in rows], default=0)
        col_widths[h] = max(len(h), max_cell)

    # Header
    header_line = " | ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "-+-".join("-" * col_widths[h] for h in headers)

    lines = [header_line, sep_line]

    # Rows
    for row in rows:
        line = " | ".join(_fmt(row[h]).ljust(col_widths[h]) for h in headers)
        lines.append(line)

    return "\n".join(lines)