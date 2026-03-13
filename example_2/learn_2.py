from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import ARS, QRDQN, TRPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from example_2 import TalluriExample2
from em_estimation import (
    ALL_METHODS,
    SENSITIVITY_BETA_TARGETS,
    estimate_beta_and_arrival_em,
    estimate_betas_for_both_sensitivities_all_methods,
)
from config import OPT_MODEL as CONFIG_OPT_MODEL, TOTAL_TIMESTEPS as CONFIG_TOTAL_TIMESTEPS
import numpy as np
import os
import time

N_EVAL_EPISODES = 100
N_TRAIN_RUNS = 5
N_ESTIMATION_EPISODES = 50
LEARNING_CURVE_ENABLED = False
LEARNING_CURVE_EVAL_FREQ = 100
LEARNING_CURVE_EVAL_EPISODES = 30
LEARNING_CURVE_OUTPUT_DIR = "example_2"


ALGORITHMS = {
    "DQN": DQN,
    "QRDQN": QRDQN,
    "ARS": ARS,
    "A2C": A2C,
    "TRPO": TRPO,
    "PPO": PPO,
}


class PercentOptimalCallback(BaseCallback):
    def __init__(self, eval_env, v, pi, eval_freq, n_eval_episodes, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.v = v
        self.pi = pi
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.timesteps = []
        self.pct_optimal_mean = []
        self.pct_optimal_std = []

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            pct_values = []
            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                dp_reward, _ = self.eval_env.optimal(self.v, self.pi)

                total_reward = 0.0
                for _ in range(self.eval_env.T):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = self.eval_env.step(action)
                    total_reward += float(reward)
                    if done or truncated:
                        break

                if dp_reward != 0:
                    pct_values.append(100.0 * total_reward / float(dp_reward))

            if pct_values:
                self.timesteps.append(int(self.num_timesteps))
                self.pct_optimal_mean.append(float(np.mean(pct_values)))
                self.pct_optimal_std.append(float(np.std(pct_values)))

        return True


def create_train_env():
    return make_vec_env(
        TalluriExample2,
        n_envs=4,
        vec_env_cls=DummyVecEnv,
    )


def _reward_to_purchase_index(env, reward):
    if reward <= 0:
        return None

    matching = np.where(np.isclose(env.r, reward))[0]
    if len(matching) == 0:
        return None
    return int(matching[0])


def collect_incomplete_transaction_data(env, n_episodes):
    observations = []

    for _ in range(n_episodes):
        obs, _ = env.reset()

        for _ in range(env.T):
            action_idx = int(np.random.randint(0, env.action_space.n))
            action_binary = env._action_to_binary(env._resolve_action(action_idx))

            obs, reward, done, truncated, _ = env.step(action_idx)

            observations.append(
                {
                    "action_binary": action_binary,
                    "purchase_index": _reward_to_purchase_index(env, reward),
                }
            )

            if done or truncated:
                break

    return observations


def estimate_beta_via_em(model):
    estimation_env = TalluriExample2()
    observations = collect_incomplete_transaction_data(estimation_env, N_ESTIMATION_EPISODES)
    em_result = estimate_beta_and_arrival_em(
        observations=observations,
        prices=estimation_env.r,
        model=model,
        beta_init=-0.002,
        lambda_init=0.5,
        beta_bounds=(-0.05, -1e-6),
        max_iter=200,
        tol=1e-7,
    )
    estimation_env.close()
    return em_result


def _collect_observations_for_sensitivity(target_beta, n_episodes):
    original_beta = TalluriExample2.ENV_BETA
    try:
        TalluriExample2.ENV_BETA = float(target_beta)
        estimation_env = TalluriExample2()
        observations = collect_incomplete_transaction_data(estimation_env, n_episodes)
        estimation_env.close()
        return observations
    finally:
        TalluriExample2.ENV_BETA = original_beta


def print_all_beta_estimates_table(n_episodes_per_sensitivity=N_ESTIMATION_EPISODES):

    observations_by_sensitivity = {
        "low": _collect_observations_for_sensitivity(
            SENSITIVITY_BETA_TARGETS["low"],
            n_episodes_per_sensitivity,
        ),
        "high": _collect_observations_for_sensitivity(
            SENSITIVITY_BETA_TARGETS["high"],
            n_episodes_per_sensitivity,
        ),
    }

    estimation_results = estimate_betas_for_both_sensitivities_all_methods(
        observations_by_sensitivity=observations_by_sensitivity,
        prices=TalluriExample2.r,
        methods=ALL_METHODS,
        beta_bounds=(-0.05, -1e-6),
        max_iter=200,
        tol=1e-7,
    )

    print("\n=== EM Beta Estimates (All Methods, Both Sensitivities) ===")
    print(
        f"{'Sensitivity':<12} {'Method':<14} {'Beta(est)':>12} "
        f"{'Lambda':>10} {'Iter':>6}"
    )
    for sensitivity in ("low", "high"):
        for method in ALL_METHODS:
            result = estimation_results[sensitivity][method]
            beta_est = float(result["beta"])
            lambda_est = float(result["lambda"])
            iterations = int(result["iterations"])
            print(
                f"{sensitivity:<12} {method:<14} {beta_est:>12.6f} "
                f"{lambda_est:>10.4f} {iterations:>6d}"
            )


def train_model(algorithm_name, callback=None):
    env = create_train_env()
    start_time = time.perf_counter()
    model = ALGORITHMS[algorithm_name]("MlpPolicy", env)
    model.learn(total_timesteps=CONFIG_TOTAL_TIMESTEPS, progress_bar=True, callback=callback)
    elapsed = time.perf_counter() - start_time
    env.close()
    return model, elapsed


def monitor_learning_curve(algorithm_name, output_path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plotting. Install it with: pip install matplotlib")
        return

    em_result = estimate_beta_via_em(CONFIG_OPT_MODEL)
    estimated_beta = float(em_result["beta"])

    eval_env = TalluriExample2()
    v, pi = eval_env.solve_by_dp(estimated_beta)

    callback = PercentOptimalCallback(
        eval_env=eval_env,
        v=v,
        pi=pi,
        eval_freq=LEARNING_CURVE_EVAL_FREQ,
        n_eval_episodes=LEARNING_CURVE_EVAL_EPISODES,
    )

    print(
        f"\nLearning curve run: {algorithm_name} | eval every {LEARNING_CURVE_EVAL_FREQ} timesteps "
        f"| {LEARNING_CURVE_EVAL_EPISODES} episodes per eval"
    )
    train_model(algorithm_name, callback=callback)

    if not callback.timesteps:
        print("No evaluation points were collected. Increase total timesteps or lower LEARNING_CURVE_EVAL_FREQ.")
        eval_env.close()
        return

    plt.figure(figsize=(8, 5))
    x = np.asarray(callback.timesteps, dtype=float)
    y = np.asarray(callback.pct_optimal_mean, dtype=float)
    y_std = np.asarray(callback.pct_optimal_std, dtype=float)
    plt.plot(x, y, marker="o", label="Mean % of optimal")
    plt.fill_between(x, y - y_std, y + y_std, alpha=0.2, label="±1 SD")
    plt.xlabel("Training timesteps")
    plt.ylabel("% of DP optimal reward")
    plt.title(f"Learning performance over training ({algorithm_name})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    print(f"Saved learning curve plot to: {output_path}")
    eval_env.close()


def monitor_learning_curves_all_algorithms():
    os.makedirs(LEARNING_CURVE_OUTPUT_DIR, exist_ok=True)

    for algorithm_name in ALGORITHMS:
        output_path = os.path.join(
            LEARNING_CURVE_OUTPUT_DIR,
            f"pct_optimal_learning_curve_{algorithm_name}.png",
        )
        monitor_learning_curve(algorithm_name=algorithm_name, output_path=output_path)


def evaluate_model(model, eval_env, v, pi):
    rewards = []
    load_factors = []
    optimal_rewards_xi = []
    optimal_load_factors_xi = []

    for _ in range(N_EVAL_EPISODES):
        obs, _ = eval_env.reset()

        optimal_reward_xi, optimal_load_factor_xi = eval_env.optimal(v, pi)
        optimal_rewards_xi.append(optimal_reward_xi)
        optimal_load_factors_xi.append(optimal_load_factor_xi)

        total_reward = 0
        load_factor = 0
        for _ in range(eval_env.T):
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward
            load_factor = 100.0 * (1.0 - (obs[1] / eval_env.C))

            if done or truncated:
                break
        rewards.append(total_reward)
        load_factors.append(float(load_factor))

    dp_mean, dp_std = float(np.mean(optimal_rewards_xi)), float(np.std(optimal_rewards_xi))
    dp_load_factor_mean = float(np.mean(optimal_load_factors_xi))
    dp_load_factor_std = float(np.std(optimal_load_factors_xi))
    model_mean, model_std = float(np.mean(rewards)), float(np.std(rewards))
    load_factor_mean, load_factor_std = float(np.mean(load_factors)), float(np.std(load_factors))
    pct_rewards = [
        100.0 * model_reward / dp_reward if dp_reward != 0 else np.nan
        for model_reward, dp_reward in zip(rewards, optimal_rewards_xi)
    ]

    return {
        "dp_reward_mean": dp_mean,
        "dp_reward_std": dp_std,
        "dp_load_factor_mean": dp_load_factor_mean,
        "dp_load_factor_std": dp_load_factor_std,
        "reward_mean": model_mean,
        "reward_std": model_std,
        "load_factor_mean": load_factor_mean,
        "load_factor_std": load_factor_std,
        "pct_reward": float(np.mean(pct_rewards)),
        "pct_reward_std": float(np.nanstd(pct_rewards)),
    }


def benchmark_algorithms():

    em_start_time = time.perf_counter()
    em_result = estimate_beta_via_em(CONFIG_OPT_MODEL)
    em_estimation_time_sec = time.perf_counter() - em_start_time
    estimated_beta = float(em_result["beta"])
    estimated_lambda = float(em_result["lambda"])

    print(f"EM estimation time: {em_estimation_time_sec:.2f} sec | EM iterations: {em_result['iterations']} | Estimation periods: {em_result['n_periods']}")
    print(f"Estimated beta: {estimated_beta:.6f} | True env beta: {TalluriExample2.ENV_BETA:.6f}")
    print(f"Estimated arrival probability: {estimated_lambda:.4f} | True: {TalluriExample2.ARRIVAL_PROB:.4f}")

    eval_env = TalluriExample2()
    dp_start_time = time.perf_counter()
    v, pi = eval_env.solve_by_dp(estimated_beta)
    dp_solve_time_sec = time.perf_counter() - dp_start_time
    print(f"DP solution time: {dp_solve_time_sec:.2f} sec")
    print(f"Optimal value at (0, C): {v[0, eval_env.C]:.2f}")

    dp_rewards = []
    dp_load_factors = []
    for _ in range(N_EVAL_EPISODES):
        eval_env.reset()
        dp_reward, dp_load_factor = eval_env.optimal(v, pi)
        dp_rewards.append(dp_reward)
        dp_load_factors.append(dp_load_factor)

    results = [{
                "algorithm": "DP",
                "training_time_sec": dp_solve_time_sec,
                "dp_reward_mean": float(np.mean(dp_rewards)),
                "dp_reward_std": float(np.std(dp_rewards)),
                "dp_load_factor_mean": float(np.mean(dp_load_factors)),
                "dp_load_factor_std": float(np.std(dp_load_factors)),
                "reward_mean": float(np.mean(dp_rewards)),
                "reward_std": float(np.std(dp_rewards)),
                "load_factor_mean": float(np.mean(dp_load_factors)),
                "load_factor_std": float(np.std(dp_load_factors)),
                "pct_reward": 100.0,
                "pct_reward_std": 0.0,
            }]

    for algorithm_name in ALGORITHMS:
        print(f"\n--- {algorithm_name}: training ({N_TRAIN_RUNS} runs) ---")
        run_metrics = []

        for run_idx in range(N_TRAIN_RUNS):
            print(f"{algorithm_name} run {run_idx + 1}/{N_TRAIN_RUNS}")
            try:
                model, learning_time_sec = train_model(algorithm_name)
                metrics = evaluate_model(model, eval_env, v, pi)
                metrics["training_time_sec"] = learning_time_sec
                run_metrics.append(metrics)
                print(
                    f"  reward: {metrics['reward_mean']:.2f} ± {metrics['reward_std']:.2f} "
                    f"| load factor: {metrics['load_factor_mean']:.2f}% ± {metrics['load_factor_std']:.2f}% "
                    f"| time: {learning_time_sec:.2f} sec "
                    f"| % optimal: {metrics['pct_reward']:.2f}% ± {metrics['pct_reward_std']:.2f}%"
                )
            except Exception as error:
                print(f"  run failed: {error}")

        if not run_metrics:
            print(f"{algorithm_name} failed in all runs.")
            continue

        averaged_metrics = {
            "algorithm": algorithm_name,
            "training_time_sec": float(np.mean([metric["training_time_sec"] for metric in run_metrics])),
            "reward_mean": float(np.mean([metric["reward_mean"] for metric in run_metrics])),
            "reward_std": float(np.mean([metric["reward_std"] for metric in run_metrics])),
            "dp_reward_mean": float(np.mean([metric["dp_reward_mean"] for metric in run_metrics])),
            "dp_reward_std": float(np.mean([metric["dp_reward_std"] for metric in run_metrics])),
            "load_factor_mean": float(np.mean([metric["load_factor_mean"] for metric in run_metrics])),
            "load_factor_std": float(np.mean([metric["load_factor_std"] for metric in run_metrics])),
            "dp_load_factor_mean": float(np.mean([metric["dp_load_factor_mean"] for metric in run_metrics])),
            "dp_load_factor_std": float(np.mean([metric["dp_load_factor_std"] for metric in run_metrics])),
            "pct_reward": float(np.mean([metric["pct_reward"] for metric in run_metrics])),
            "pct_reward_std": float(np.std([metric["pct_reward"] for metric in run_metrics])),
        }

        results.append(averaged_metrics)

        print(f"{algorithm_name} average reward: {averaged_metrics['reward_mean']:.2f} ± {averaged_metrics['reward_std']:.2f}")
        print(
            f"{algorithm_name} average load factor: "
            f"{averaged_metrics['load_factor_mean']:.2f}% ± {averaged_metrics['load_factor_std']:.2f}%"
        )
        print(f"DP reward: {averaged_metrics['dp_reward_mean']:.2f} ± {averaged_metrics['dp_reward_std']:.2f}")
        print(
            f"DP load factor: "
            f"{averaged_metrics['dp_load_factor_mean']:.2f}% ± {averaged_metrics['dp_load_factor_std']:.2f}%"
        )
        print(
            f"{algorithm_name} average % of DP reward: "
            f"{averaged_metrics['pct_reward']:.2f}% ± {averaged_metrics['pct_reward_std']:.2f}%"
        )

    eval_env.close()

    return results


def print_summary(results):
    if not results:
        print("\nNo algorithm finished successfully.")
        return

    print("\n=== Benchmark Summary (Example 2) ===")
    print(f"{'Algorithm':<10} {'Time(s)':>10} {'Reward':>12} {'Load%':>10} {'%DP R':>8} {'%DP SD':>8}")
    for item in results:
        print(
            f"{item['algorithm']:<10} "
            f"{item['training_time_sec']:>10.2f} "
            f"{item['reward_mean']:>12.2f} "
            f"{item['load_factor_mean']:>10.2f} "
            f"{item['pct_reward']:>8.2f} "
            f"{item['pct_reward_std']:>8.2f}"
        )

def main():
    if LEARNING_CURVE_ENABLED:
        monitor_learning_curves_all_algorithms()

    results = benchmark_algorithms()
    print_summary(results)


if __name__ == "__main__":
    main()