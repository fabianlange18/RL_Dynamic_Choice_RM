import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import constants as C
from simulation import simulate


class PercentOptimalCallback(BaseCallback):
    def __init__(self, pi, possible_sets = None, verbose=0):
        super().__init__(verbose)
        self.possible_sets = possible_sets
        self.pi = pi
        self.eval_freq = C.TOTAL_TIMESTEPS // 20
        self._next_eval_timestep = C.TOTAL_TIMESTEPS // 20
        self.timesteps = []
        self.pct_optimal_mean = []
        self.pct_optimal_std = []

    def _on_step(self):
        if self.eval_freq > 0 and self.num_timesteps >= self._next_eval_timestep:
            pct_values = []
            for seed in range(C.N_EVAL_EPISODES):
                dp_reward, _ = simulate(self.possible_sets, pi=self.pi, seed=seed)
                rl_reward, _ = simulate(self.possible_sets, model=self.model, seed=seed)

                if dp_reward != 0:
                    pct_values.append(100.0 * rl_reward / float(dp_reward))

            if pct_values:
                self.timesteps.append(int(self.num_timesteps))
                self.pct_optimal_mean.append(float(np.mean(pct_values)))
                self.pct_optimal_std.append(float(np.std(pct_values)))

            while self._next_eval_timestep <= self.num_timesteps:
                self._next_eval_timestep += self.eval_freq

        return True
