import numpy as np
import gymnasium as gym
from collections import deque

import config as c
import constants as C
from buying_probabilities import get_buying_probabilities_by_model


class TalluriExample2(gym.Env):

    def __init__(self, efficient_sets = None, use_multibinary_action_space: bool = False):

        if efficient_sets is not None:
            self.possible_sets = tuple(efficient_sets)
        elif use_multibinary_action_space or c.LARGE_PRODUCT_SET:
            # In MultiBinary mode, actions are direct product-wise 0/1 decisions.
            # Do not materialize all 2^n sets.
            self.possible_sets = None
        else:
            self.possible_sets = tuple(range(2 ** C.n))
        
        self.use_multibinary_action_space = use_multibinary_action_space or self.possible_sets is None

        if self.use_multibinary_action_space:
            self.action_space = gym.spaces.MultiBinary(C.n)
        else:
            self.action_space = gym.spaces.Discrete(len(self.possible_sets))
            
        self.observation_space = gym.spaces.MultiDiscrete([C.T + 1, C.C + 1])
        
        self._seed = None
        self._recent_timestep_offer_means = deque(maxlen=5)

    def reset(self, seed=None, options=None):
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        self._recent_timestep_offer_means.clear()
        self.s = (0, C.C)
        self.arrival_xi = self.rng.choice([0, 1], C.T, p=[1.0 - C.ARRIVAL_PROB, C.ARRIVAL_PROB])
        self.buying_xi = self.rng.uniform(0, 1, C.T)
        return self.s, {}

    def step(self, action):
        
        action = self._action_to_binary(action)
        reference_price = float(np.mean(self._recent_timestep_offer_means)) if self._recent_timestep_offer_means else float(np.mean(C.r))

        # Update reference price buffer
        offered_indices = np.where(action == 1)[0]
        current_offer_mean = float(np.mean(C.r[offered_indices])) if offered_indices.size else float(np.mean(C.r))
        self._recent_timestep_offer_means.append(current_offer_mean)

        t, inventory = self.s

        if self.arrival_xi[t] == 0:
            t += 1
            self.s = (t, inventory)
            return self.s, 0, t == C.T, False, {}

        buying_probabilities = get_buying_probabilities_by_model(
            action_binary=action,
            beta=C.SENSITIVITY_BETA_GT["high"] if c.HIGH_SENSITIVITY else C.SENSITIVITY_BETA_GT["low"],
            model=c.GT_MODEL,
            reference_price=reference_price,
            seed=self._seed + t if self._seed is not None else None,
        )

        cumulative_probs = np.cumsum(buying_probabilities)
        choice = np.searchsorted(cumulative_probs, self.buying_xi[t])

        reward = C.r[choice] if choice < C.n else 0

        if choice < C.n:
            inventory -= 1

        t += 1
        done = t == C.T or inventory == 0
        self.s = (t, inventory)

        return self.s, reward, done, False, {}

    
    def _action_to_binary(self, action):
        """Convert action input to binary offer-set vector of length C.n.

        Accepted inputs:
        - MultiBinary action arrays of shape (C.n,)
        - Discrete RL action indices (mapped through self.possible_sets)
        - Integer bitmasks (used directly)
        """
        if isinstance(action, np.ndarray):
            if action.shape == (C.n,):
                return action.astype(int)
            if action.shape == ():
                action_int = int(action.item())
            elif action.shape == (1,):
                action_int = int(action[0])
            else:
                raise ValueError(f"Unsupported action shape {action.shape}")
        else:
            action_int = int(action)

        if self.possible_sets is None:
            # In MultiBinary mode, scalar actions are interpreted as bitmasks.
            if action_int < 0:
                raise ValueError("Action bitmask must be non-negative")
        else:
            action_int = int(self.possible_sets[action_int])

        return np.array([(action_int >> i) & 1 for i in range(C.n)], dtype=int)