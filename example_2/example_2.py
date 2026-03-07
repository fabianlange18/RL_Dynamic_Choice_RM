import numpy as np
import gymnasium as gym
import time
from config import (
            HIGH_SENSITIVITY as CONFIG_HIGH_SENSITIVITY,
            GT_MODEL as CONFIG_GT_MODEL,
            OPT_MODEL as CONFIG_OPT_MODEL,
        )
from buying_probabilities import get_buying_probabilities_by_model


class TalluriExample2(gym.Env):

    T = 410
    C = 185
    n = 10
    ARRIVAL_PROB = 0.5
    r = np.asarray([600, 550, 475, 400, 300, 280, 240, 200, 185, 175], dtype=float)
    ENV_BETA = -0.005 if CONFIG_HIGH_SENSITIVITY else -0.0015

    _efficient_sets_cache = None

    def __init__(self):

        self.efficient_sets = self.precompute_efficient_sets()
        self.action_space = gym.spaces.Discrete(len(self.efficient_sets))
        self.observation_space = gym.spaces.MultiDiscrete([self.T + 1, self.C + 1])

    @classmethod
    def precompute_efficient_sets(cls):
        if cls._efficient_sets_cache is None:
            start_time = time.perf_counter()
            efficient_sets = cls.identify_efficient_sets(beta=cls.ENV_BETA)
            elapsed_time = time.perf_counter() - start_time
            cls._efficient_sets_cache = tuple(efficient_sets)
            print(f"Time to identify efficient sets: {elapsed_time:.4f} seconds")
            print(f"Identified {len(efficient_sets)} efficient sets:", efficient_sets)

        return cls._efficient_sets_cache

    def reset(self, seed=None, options=None):
        self.s = (0, self.C)
        self.arrival_xi = np.random.choice(
            [0, 1],
            self.T,
            p=[1.0 - self.ARRIVAL_PROB, self.ARRIVAL_PROB],
        )
        self.buying_xi = np.random.uniform(0, 1, self.T)
        return self.s, {}

    def step(self, action):
        resolved_action = self._resolve_action(action)
        action = self._action_to_binary(resolved_action)

        t, inventory = self.s

        if self.arrival_xi[t] == 0:
            t += 1
            self.s = (t, inventory)
            return self.s, 0, t == self.T, False, {}

        buying_probabilities = self.get_buying_probabilities(
            action,
            model=CONFIG_GT_MODEL,
        )
        cumulative_probs = np.cumsum(buying_probabilities)
        choice = np.searchsorted(cumulative_probs, self.buying_xi[t])

        reward = self.r[choice] if choice < self.n else 0

        if choice < self.n:
            inventory -= 1

        t += 1
        done = t == self.T or inventory == 0
        self.s = (t, inventory)
        return self.s, reward, done, False, {}

    def _to_action_int(self, action):
        if isinstance(action, np.ndarray):
            if action.shape == ():
                return int(action.item())
            if action.shape == (1,):
                return int(action[0])
            raise ValueError(f"Expected scalar action, got shape {action.shape}")

        return int(action)

    def _resolve_action(self, action):
        action_int = self._to_action_int(action)
        if 0 <= action_int < len(self.efficient_sets):
            return self.efficient_sets[action_int]
        return action_int

    def _action_to_binary(self, action):
        if isinstance(action, np.ndarray) and action.shape == (self.n,):
            return action.astype(int)

        action_int = self._to_action_int(action)
        return np.array([(action_int >> i) & 1 for i in range(self.n)], dtype=int)

    @classmethod
    def get_buying_probabilities(cls, action, beta=None, model=None):
        if beta is None:
            beta = cls.ENV_BETA

        probabilities = get_buying_probabilities_by_model(
            action_binary=action,
            prices=cls.r,
            beta=beta,
            model=model,
        )
        return probabilities

    @classmethod
    def identify_efficient_sets(cls, beta=None, model="MNL"):
        offer_set_metrics = []

        for action_int in range(2 ** cls.n):
            action = np.array([(action_int >> i) & 1 for i in range(cls.n)], dtype=int)
            buying_probabilities = np.asarray(
                cls.get_buying_probabilities(action, beta=beta, model=model),
                dtype=float,
            )
            q_value = float(np.sum(buying_probabilities))
            r_value = float(np.dot(cls.r, buying_probabilities))
            offer_set_metrics.append({"action": action_int, "Q": q_value, "R": r_value})

        tol = 1e-12
        efficient_sequence = [0]
        current = offer_set_metrics[0]

        while True:
            best_candidate = None
            best_ratio = -np.inf

            for candidate in offer_set_metrics:
                if candidate["action"] in efficient_sequence:
                    continue

                if candidate["Q"] + tol < current["Q"] or candidate["R"] + tol < current["R"]:
                    continue

                delta_q = candidate["Q"] - current["Q"]
                delta_r = candidate["R"] - current["R"]

                if delta_q <= tol:
                    continue

                marginal_ratio = delta_r / delta_q
                if (
                    marginal_ratio > best_ratio + tol
                    or (
                        abs(marginal_ratio - best_ratio) <= tol
                        and best_candidate is not None
                        and (
                            candidate["R"] > best_candidate["R"] + tol
                            or (
                                abs(candidate["R"] - best_candidate["R"]) <= tol
                                and candidate["Q"] > best_candidate["Q"] + tol
                            )
                        )
                    )
                    or best_candidate is None
                ):
                    best_ratio = marginal_ratio
                    best_candidate = candidate

            if best_candidate is None:
                break

            efficient_sequence.append(best_candidate["action"])
            current = best_candidate

        return efficient_sequence

    def solve_by_dp(self, estimated_beta=None):
        v = np.zeros((self.T + 1, self.C + 1))
        pi = np.zeros((self.T, self.C + 1), dtype=int)

        for t in range(self.T - 1, -1, -1):
            for x in range(self.C + 1):
                if x > 0:
                    best_value = 0
                    best_action = 0

                    for action_idx in range(self.action_space.n):
                        action = self._action_to_binary(self._resolve_action(action_idx))
                        expected_value = 0

                        no_arrival_prob = 1.0 - self.ARRIVAL_PROB
                        expected_value += no_arrival_prob * v[t + 1, x]

                        arrival_prob = 1 - no_arrival_prob
                        if arrival_prob > 0:
                            buying_probabilities = self.get_buying_probabilities(
                                action,
                                beta=estimated_beta,
                                model=CONFIG_OPT_MODEL,
                            )

                            immediate_reward = 0
                            for j in range(self.n):
                                immediate_reward += buying_probabilities[j] * self.r[j]

                            expected_future_value = 0
                            purchase_prob = sum(buying_probabilities)
                            if x > 0 and purchase_prob > 0:
                                expected_future_value += purchase_prob * v[t + 1, x - 1]

                            no_purchase_prob = 1 - purchase_prob
                            expected_future_value += no_purchase_prob * v[t + 1, x]

                            expected_value += arrival_prob * (immediate_reward + expected_future_value)

                        if expected_value > best_value:
                            best_value = expected_value
                            best_action = action_idx

                    v[t, x] = best_value
                    pi[t, x] = best_action

        return v, pi

    def optimal(self, v=None, pi=None):
        if pi is None:
            v, pi = self.solve_by_dp()

        self.s = (0, self.C)
        obs = self.s
        total_reward = 0

        for _ in range(self.T):
            action = pi[obs[0], obs[1]]
            obs, reward, done, truncated, _ = self.step(action)
            total_reward += reward

            if done or truncated:
                self.s = (0, self.C)
                break

        load_factor = 100 * (1 - obs[1] / self.C)
        return total_reward, load_factor