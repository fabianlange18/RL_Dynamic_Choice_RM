import constants as C

from env_example_2 import TalluriExample2


def simulate(efficient_sets = None, pi = None, model = None, seed = 42):

    env = TalluriExample2(efficient_sets=efficient_sets)

    obs = env.reset(seed=seed)[0]
    total_reward = 0

    for _ in range(C.T):
        action = int(pi[obs[0], obs[1]]) if pi is not None else model.predict(obs, deterministic=True)[0]
        action = env._action_to_binary(action)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += reward
        load_factor = 100 * (1 - obs[1] / C.C)

        if done or truncated:
            break

    return total_reward, load_factor