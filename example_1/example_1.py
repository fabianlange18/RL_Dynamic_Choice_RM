import numpy as np
import gymnasium as gym
import time

class TalluriExample1(gym.Env):

    T = 50 # T * ARRIVAL_PROB is the number of expected arrivals (30, 40 or 50)
    C = 20
    n = 3
    ARRIVAL_PROB = 0.5
    r = np.asarray([800, 500, 450], dtype=float)
    BUYING_PROBABILITIES = np.asarray(
        [
            [0, 0, 0, 1],
            [0.3, 0, 0, 0.7],
            [0, 0.4, 0, 0.6],
            [0, 0, 0.5, 0.5],
            [0.1, 0.6, 0, 0.3],
            [0.3, 0, 0.5, 0.2],
            [0, 0.4, 0.5, 0.1],
            [0.1, 0.4, 0.5, 0],
        ],
        dtype=float,
    )

    _efficient_sets_cache = None

    def __init__(self):

        self.efficient_sets = self.precompute_efficient_sets()
        self.action_space = gym.spaces.Discrete(len(self.efficient_sets))
        self.observation_space = gym.spaces.MultiDiscrete([self.T + 1, self.C + 1])

    @classmethod
    def precompute_efficient_sets(cls):
        if cls._efficient_sets_cache is None:
            start_time = time.time()
            efficient_sets = cls.identify_efficient_sets()
            elapsed_time = time.time() - start_time
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

        t, inventory = self.s

        if self.arrival_xi[t] == 0:
            t += 1
            done = t == self.T or inventory == 0
            self.s = (t, inventory)
            return self.s, 0, done, False, {}

        buying_probabilities = self.get_buying_probabilities(self._resolve_action(action))

        cumulative_probs = np.cumsum(buying_probabilities)
        choice = np.searchsorted(cumulative_probs, self.buying_xi[t])
        
        reward = self.r[choice] if choice < self.n else 0

        if choice < self.n:
            inventory -= 1
        
        t += 1
        
        done = t == self.T or inventory == 0
        
        self.s = (t, inventory)

        return self.s, reward, done, False, {}

    def _resolve_action(self, action):
        return self.efficient_sets[action]
    
    @classmethod
    def get_buying_probabilities(cls, action):
        return cls.BUYING_PROBABILITIES[action]

    @classmethod
    def identify_efficient_sets(cls):
        offer_set_metrics = []

        for action in range(2 ** cls.n):
            buying_probabilities = cls.get_buying_probabilities(action)
            q_value = float(np.sum(buying_probabilities[:cls.n]))
            r_value = float(np.dot(cls.r, buying_probabilities[:cls.n]))
            offer_set_metrics.append({"action": action, "Q": q_value, "R": r_value})

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
    
    def solve_by_dp(self):
        v = np.zeros((self.T + 1, self.C + 1))
        pi = np.zeros((self.T, self.C + 1), dtype=int)
        
        for t in range(self.T - 1, -1, -1):
            for x in range(1, self.C + 1):

                best_value = 0
                best_action = 0

                for action in range(self.action_space.n):
                    buying_probabilities = self.get_buying_probabilities(self._resolve_action(action))

                    no_purchase_probability = float(buying_probabilities[self.n])

                    total_expected_value = (
                        self.ARRIVAL_PROB
                        * float(
                            np.sum(
                                buying_probabilities[:self.n]
                                * (self.r + v[t + 1, x - 1])
                            )
                        )
                        + (self.ARRIVAL_PROB * no_purchase_probability + (1.0 - self.ARRIVAL_PROB))
                        * v[t + 1, x]
                    )

                    if total_expected_value > best_value:
                        best_value = total_expected_value
                        best_action = action

                v[t, x] = best_value
                pi[t, x] = best_action

        return v, pi


    def optimal(self, v = None, pi = None):
        
        if pi is None:
            v, pi = self.solve_by_dp()
        
        self.s = (0, self.C)
        obs = self.s
        total_reward = 0
        
        for _ in range(self.T):
            action = pi[obs[0], obs[1]]
            obs, reward, done, truncated, info = self.step(action)
            total_reward += reward
            
            if done or truncated:
                self.s = (0, self.C)
                break
        
        return total_reward