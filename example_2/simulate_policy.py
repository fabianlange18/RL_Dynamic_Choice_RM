import numpy as np

import constants as C


def simulate_dp(env, v, pi):
    """Simulate the DP optimal policy on the given environment.

    Resets the env internally and follows policy pi for up to C.T steps.

    Args:
        env: TalluriExample2 environment instance
        v: DP value function array (T+1, C+1)
        pi: DP policy array (T, C+1) with action indices into env.possible_sets

    Returns:
        total_reward: Total revenue accumulated over the episode
    """
    obs, _ = env.reset()
    total_reward = 0.0

    for _ in range(C.T):
        action_idx = int(pi[obs[0], obs[1]])
        action_int = env.possible_sets[action_idx]
        action = np.array([(int(action_int) >> i) & 1 for i in range(C.n)], dtype=int)
        obs, reward, done, truncated, _ = env.step(action)
        total_reward += float(reward)

        if done or truncated:
            break

    return total_reward


