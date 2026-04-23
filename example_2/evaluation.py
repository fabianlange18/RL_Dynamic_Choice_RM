import os
import numpy as np

from simulation import simulate
import constants as C


def _evaluate_dp_policy(policy, efficient_sets):
    rewards = []
    load_factors = []
    for seed in range(C.N_EVAL_EPISODES):
        reward, load_factor = simulate(efficient_sets=efficient_sets, pi=policy, seed=seed)
        rewards.append(reward)
        load_factors.append(load_factor)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_load_factor": np.mean(load_factors),
        "std_load_factor": np.std(load_factors),
        "seed_rewards": rewards,
    }


def evaluate_saved_models(dp_policy_configs, rl_efficient_sets=None):
    """Evaluate all saved RL runs and provided DP policies.

    Parameters
    ----------
    dp_policy_configs : dict
        Mapping policy name -> {"pi": policy_array, "efficient_sets": set_or_none}
        Each DP policy is evaluated with its own efficient-set mapping.
    rl_efficient_sets : iterable or None
        Efficient sets used for evaluating RL policies.
    """
    dp_results = {}
    for policy_name, policy_config in dp_policy_configs.items():
        dp_results[policy_name] = _evaluate_dp_policy(
            policy=policy_config["pi"],
            efficient_sets=policy_config["efficient_sets"],
        )

    results_by_step = {}
    for step in C.TOTAL_TIMESTEPS:
        step_results = {}
        for algorithm_name in C.RL_ALGORITHMS.keys():
            algo_rewards = []
            algo_load_factors = []
            for seed in range(C.N_EVAL_EPISODES):
                model_path = os.path.join(C.OUTPUT_DIR, f"{algorithm_name}_model_seed{seed}_step{step}.zip")
                if os.path.exists(model_path):
                    model = C.RL_ALGORITHMS[algorithm_name].load(model_path)
                    reward, load_factor = simulate(efficient_sets=rl_efficient_sets, model=model, seed=seed)
                    algo_rewards.append(reward)
                    algo_load_factors.append(load_factor)
            if algo_rewards:
                step_results[algorithm_name] = {
                    "mean_reward": np.mean(algo_rewards),
                    "std_reward": np.std(algo_rewards),
                    "mean_load_factor": np.mean(algo_load_factors),
                    "std_load_factor": np.std(algo_load_factors),
                    "seed_rewards": algo_rewards,
                }
        step_results.update(dp_results)
        results_by_step[step] = step_results

    return results_by_step


def print_evaluation_table(training_times, evaluation_results):
    # Keys that are RL algorithms (not DP policies)
    rl_algo_keys = [k for k in evaluation_results if k in C.RL_ALGORITHMS.keys()]
    dp_policy_keys = [k for k in evaluation_results if k not in rl_algo_keys]

    methods = rl_algo_keys + dp_policy_keys

    # DP_MNL is the percentage reference; fall back to first DP key if absent
    dp_ref_key = "DP_MNL" if "DP_MNL" in evaluation_results else (dp_policy_keys[0] if dp_policy_keys else None)
    dp_seed_rewards_ref = evaluation_results[dp_ref_key].get("seed_rewards", []) if dp_ref_key else []

    # Build rows
    rows = []
    for method in methods:
        t_vals = [float(run_dict[method]) for run_dict in training_times if method in run_dict]
        if "seed_rewards" in evaluation_results[method]:
            method_seed_rewards = evaluation_results[method]["seed_rewards"]
            pct_vals = [
                (method_seed_rewards[idx] / dp_seed_rewards_ref[idx]) * 100
                for idx in range(C.N_EVAL_EPISODES)
            ]
            pct_dp_mean = float(np.mean(pct_vals))
            pct_dp_std = float(np.std(pct_vals))
        else:
            pct_dp_mean = np.nan
            pct_dp_std = np.nan
        
        rows.append(
            {
                "Method": method,
                "TrainTimeMean(s)": float(np.mean(t_vals)) if t_vals else np.nan,
                "TrainTimeStd(s)": float(np.std(t_vals)) if t_vals else np.nan,
                "RewardMean": float(evaluation_results[method]["mean_reward"]),
                "RewardStdAcrossRuns": float(evaluation_results[method]["std_reward"]),
                "PctOfDP(%)": pct_dp_mean,
                "PctOfDPStd(%)": pct_dp_std,
                "LoadFactorMean(%)": float(evaluation_results[method]["mean_load_factor"]),
                "LoadFactorStdAcrossRuns(%)": float(evaluation_results[method]["std_load_factor"]),
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