from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import ARS, QRDQN, TRPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
from example_1 import TalluriExample1
import numpy as np
import time

TOTAL_TIMESTEPS = 1000000
N_EVAL_EPISODES = 100
N_TRAIN_RUNS = 5

ALGORITHMS = {
    "DQN": DQN,
    "QRDQN": QRDQN,
    "ARS": ARS,
    "A2C": A2C,
    "TRPO": TRPO,
    "PPO": PPO,
}


def create_train_env():
    return make_vec_env(
        TalluriExample1,
        n_envs=4,
        vec_env_cls=DummyVecEnv,
    )


def train_model(algorithm_name):
    env = create_train_env()
    start_time = time.perf_counter()
    model = ALGORITHMS[algorithm_name]("MlpPolicy", env)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, progress_bar=True)
    elapsed = time.perf_counter() - start_time
    env.close()
    return model, elapsed


def evaluate_model(model, eval_env, v, pi):
    rewards = []
    optimal_rewards_xi = []

    for _ in range(N_EVAL_EPISODES):
        obs, _ = eval_env.reset()

        optimal_reward_xi = eval_env.optimal(v, pi)
        optimal_rewards_xi.append(optimal_reward_xi)

        total_reward = 0
        for _ in range(eval_env.T):
            action = model.predict(obs, deterministic=True)[0]
            obs, reward, done, truncated, _ = eval_env.step(action)
            total_reward += reward

            if done or truncated:
                break
        rewards.append(total_reward)

    dp_mean, dp_std = float(np.mean(optimal_rewards_xi)), float(np.std(optimal_rewards_xi))
    model_mean, model_std = float(np.mean(rewards)), float(np.std(rewards))
    pct_rewards = [
        100.0 * model_reward / dp_reward if dp_reward != 0 else np.nan
        for model_reward, dp_reward in zip(rewards, optimal_rewards_xi)
    ]

    return {
        "dp_reward_mean": dp_mean,
        "dp_reward_std": dp_std,
        "reward_mean": model_mean,
        "reward_std": model_std,
        "pct_reward": float(np.mean(pct_rewards)),
        "pct_reward_std": float(np.nanstd(pct_rewards)),
    }


def benchmark_algorithms():
    eval_env = TalluriExample1()
    dp_start_time = time.perf_counter()
    v, pi = eval_env.solve_by_dp()
    dp_solve_time_sec = time.perf_counter() - dp_start_time
    print(f"Optimal value at (0, C): {v[0, eval_env.C]:.2f}")
    print(f"DP solution time: {dp_solve_time_sec:.2f} sec")

    results = [{
                "algorithm": "DP",
                "training_time_sec": dp_solve_time_sec,
                "dp_reward_mean": v[0, eval_env.C],
                "dp_reward_std": 0.0,
                "reward_mean": v[0, eval_env.C],
                "reward_std": 0.0,
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
            "pct_reward": float(np.mean([metric["pct_reward"] for metric in run_metrics])),
            "pct_reward_std": float(np.std([metric["pct_reward"] for metric in run_metrics])),
        }

        results.append(averaged_metrics)

        print(f"{algorithm_name} average reward: {averaged_metrics['reward_mean']:.2f} ± {averaged_metrics['reward_std']:.2f}")
        print(f"DP reward: {averaged_metrics['dp_reward_mean']:.2f} ± {averaged_metrics['dp_reward_std']:.2f}")
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

    print("\n=== Benchmark Summary (Example 1) ===")
    print(f"{'Algorithm':<10} {'Time(s)':>10} {'Reward':>12} {'%DP R':>8} {'%DP SD':>8}")
    for item in results:
        print(
            f"{item['algorithm']:<10} "
            f"{item['training_time_sec']:>10.2f} "
            f"{item['reward_mean']:>12.2f} "
            f"{item['pct_reward']:>8.2f} "
            f"{item['pct_reward_std']:>8.2f}"
        )

def main():
    results = benchmark_algorithms()
    print_summary(results)


if __name__ == "__main__":
    main()