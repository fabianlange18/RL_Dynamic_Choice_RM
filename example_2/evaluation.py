import os
import re
import numpy as np

from simulation import simulate
import constants as C
import config as c

def evaluate_saved_models(dp_policies, efficient_sets=None, additional_dp_policies=None, additional_efficient_sets=None):
    # Evaluate each DP policy once — independent of training timesteps
    dp_results = {}
    for dp_name, dp_pi in dp_policies.items():
        rewards = []
        load_factors = []
        for seed in range(C.N_EVAL_EPISODES):
            reward, load_factor = simulate(efficient_sets=efficient_sets, pi=dp_pi, seed=seed)
            rewards.append(reward)
            load_factors.append(load_factor)
        dp_results[dp_name] = {
            "mean_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "mean_load_factor": np.mean(load_factors),
            "std_load_factor": np.std(load_factors),
        }

    # Evaluate additional DP policies with their own efficient_sets
    if additional_dp_policies is not None:
        for dp_name, dp_pi in additional_dp_policies.items():
            rewards = []
            load_factors = []
            for seed in range(C.N_EVAL_EPISODES):
                reward, load_factor = simulate(efficient_sets=additional_efficient_sets, pi=dp_pi, seed=seed)
                rewards.append(reward)
                load_factors.append(load_factor)
            dp_results[dp_name] = {
                "mean_reward": np.mean(rewards),
                "std_reward": np.std(rewards),
                "mean_load_factor": np.mean(load_factors),
                "std_load_factor": np.std(load_factors),
            }

    results_by_step = {}
    for step in C.TOTAL_TIMESTEPS:
        step_results = {}
        for algorithm_name in C.RL_ALGORITHMS.keys():
            for run in range(1, c.N_TRAIN_RUNS + 1):
                model_path = os.path.join(C.OUTPUT_DIR, f"{algorithm_name}_model_run{run}_step{step}.zip")
                if os.path.exists(model_path):
                    model = C.RL_ALGORITHMS[algorithm_name].load(model_path)
                    rewards = []
                    load_factors = []
                    for seed in range(C.N_EVAL_EPISODES):
                        reward, load_factor = simulate(efficient_sets=efficient_sets, model=model, seed=seed)
                        rewards.append(reward)
                        load_factors.append(load_factor)
                    step_results[f"{algorithm_name}_run{run}"] = {
                        "mean_reward": np.mean(rewards),
                        "std_reward": np.std(rewards),
                        "mean_load_factor": np.mean(load_factors),
                        "std_load_factor": np.std(load_factors),
                    }
                else:
                    Warning(f"Model file not found for {algorithm_name} at step {step}: {model_path}")
        step_results.update(dp_results)
        results_by_step[step] = step_results

    return results_by_step


def print_evaluation_table(training_times, evaluation_results):
    run_key_pattern = re.compile(r"^(?P<algo>.+)_run\d+$")

    # Keys that are DP policies (not run-level RL entries)
    non_run_keys = [k for k in evaluation_results if not run_key_pattern.match(k)]

    methods = list(training_times[0].keys()) if training_times else []

    # Collect run-level metrics per method from keys like "DQN_run1"
    run_reward_means = {m: [] for m in methods}
    run_load_means = {m: [] for m in methods}

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

    # DP_MNL is the percentage reference; fall back to first non-run key if absent
    dp_ref_key = "DP_MNL" if "DP_MNL" in evaluation_results else (non_run_keys[0] if non_run_keys else None)
    dp_reward_ref = float(evaluation_results[dp_ref_key]["mean_reward"]) if dp_ref_key else None

    # Append all DP-like rows after RL methods
    for k in non_run_keys:
        methods.append(k)

    # Build rows
    rows = []
    for method in methods:
        # Training times across runs (not available for DP rows)
        t_vals = [float(run_dict[method]) for run_dict in training_times if method in run_dict]
        t_mean = float(np.mean(t_vals)) if t_vals else np.nan
        t_std = float(np.std(t_vals)) if t_vals else np.nan

        # Rewards/load factors
        if method in non_run_keys:
            r_vals = [float(evaluation_results[method]["mean_reward"])]
            l_vals = [float(evaluation_results[method]["mean_load_factor"])]
            if dp_reward_ref is not None and dp_reward_ref != 0:
                pct_dp_vals = [(r / dp_reward_ref) * 100 for r in r_vals]
            else:
                pct_dp_vals = []
        else:
            r_vals = run_reward_means.get(method, [])
            l_vals = run_load_means.get(method, [])
            if dp_reward_ref is not None and dp_reward_ref != 0:
                pct_dp_vals = [(r / dp_reward_ref) * 100 for r in r_vals]
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