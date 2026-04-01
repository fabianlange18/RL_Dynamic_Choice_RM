import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

import config as c
import constants as C
from env_example_2 import TalluriExample2
from evaluation_callback import PercentOptimalCallback

def _use_multibinary_action_space(algorithm_name):
    return algorithm_name in C.MULTIBINARY_ALGORITHMS

def _create_train_env(algorithm_name, efficient_sets):
    return make_vec_env(
        TalluriExample2,
        n_envs=4,
        vec_env_cls=DummyVecEnv,
        env_kwargs={
            "efficient_sets": efficient_sets,
            "use_multibinary_action_space": _use_multibinary_action_space(algorithm_name),
        },
    )

def train_rl(dp_pi, efficient_sets=None):
    checkpoints = list(C.TOTAL_TIMESTEPS)
    training_times_by_step = {step: [] for step in checkpoints}

    for run_id in range(c.N_TRAIN_RUNS):
        run_times_by_step = {step: {} for step in checkpoints}

        for algorithm_name, algorithm_cls in C.RL_ALGORITHMS.items():
            if C.LEARNING_CURVE_ENABLED:
                eval_callback = PercentOptimalCallback(dp_pi, efficient_sets)

            print(f"Training {algorithm_name} (run {run_id + 1}/{c.N_TRAIN_RUNS})")
            train_env = _create_train_env(algorithm_name, efficient_sets)
            model = algorithm_cls("MlpPolicy", train_env, verbose=0)

            prev_steps = 0
            cumulative_time = 0.0
            for i, total_steps in enumerate(checkpoints):
                delta = total_steps - prev_steps
                start_time = time.perf_counter()
                model.learn(
                    total_timesteps=delta,
                    callback=eval_callback if C.LEARNING_CURVE_ENABLED else None,
                    progress_bar=C.PROGRESS_BAR_ENABLED,
                    reset_num_timesteps=(i == 0),
                )
                cumulative_time += time.perf_counter() - start_time
                model.save(f"{C.OUTPUT_DIR}/{algorithm_name}_model_run{run_id + 1}_step{total_steps}")
                run_times_by_step[total_steps][algorithm_name] = cumulative_time
                prev_steps = total_steps

            if C.LEARNING_CURVE_ENABLED:
                plt.figure(figsize=(8, 5))
                x = np.asarray(eval_callback.timesteps, dtype=float)
                y = np.asarray(eval_callback.pct_optimal_mean, dtype=float)
                y_std = np.asarray(eval_callback.pct_optimal_std, dtype=float)
                plt.plot(x, y, marker="o", label="Mean % of optimal")
                plt.fill_between(x, y - y_std, y + y_std, alpha=0.2, label="±1 SD")
                plt.xlabel("Training timesteps")
                plt.ylabel("% of DP optimal reward")
                plt.title(f"Learning performance over training ({algorithm_name})")
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"{C.OUTPUT_DIR}/{algorithm_name}_learning_curve.png", dpi=150)
                plt.close()

        if not C.LEARNING_CURVE_ENABLED:
            for step in checkpoints:
                training_times_by_step[step].append(run_times_by_step[step])

        C.LEARNING_CURVE_ENABLED = False

    return training_times_by_step
