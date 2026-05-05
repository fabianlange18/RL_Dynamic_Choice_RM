import os
import numpy as np
import config as c

from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import ARS, QRDQN, TRPO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOTAL_TIMESTEPS = [20_000, 100_000, 500_000, 1_000_000] if c.TRAIN_ON_ALL_SETS else [20_000, 100_000, 500_000]
N_EVAL_EPISODES = 15
N_ESTIMATION_EPISODES = 50

T = 410
C = 185
n = 100 if c.LARGE_PRODUCT_SET else 10
ARRIVAL_PROB = 0.5
r = np.linspace(600, 175, n) if c.LARGE_PRODUCT_SET else np.asarray([600, 550, 475, 400, 300, 280, 240, 200, 185, 175], dtype=float)

LEARNING_CURVE_ENABLED = False
PROGRESS_BAR_ENABLED = True

DEMAND_MODELS = (
    "MNL",
    "MMNL_2PT",
    "MMNL_5PT",
	"MMNLcont",
	"Probit",
	"MNLrefPrice",
	"MNLConsidSet",
    "TMNL",
	"NLogit"
)

MULTIBINARY_ALGORITHMS = {
    "A2C",
    "TRPO",
    "PPO"
}

if c.TRAIN_ON_ALL_SETS:
    RL_ALGORITHMS = {
        "A2C": A2C,
        "TRPO": TRPO,
        "PPO": PPO,
    } if c.LARGE_PRODUCT_SET else {
        "DQN": DQN,
        "ARS": ARS,
        "A2C": A2C,
        "TRPO": TRPO,
        "PPO": PPO,
    }
else:
    RL_ALGORITHMS = {
        "DQN": DQN,
        "QRDQN": QRDQN,
        "ARS": ARS,
        "A2C": A2C,
        "TRPO": TRPO,
        "PPO": PPO,
    }


SENSITIVITY_BETA_GT = {
	"low": -0.0015,
	"high": -0.005
}

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    f"{'large' if c.LARGE_PRODUCT_SET else 'small'}_{'classical' if c.TRAIN_ON_ALL_SETS else 'model_informed'}_{'high' if c.HIGH_SENSITIVITY else 'low'}_{c.GT_MODEL}",
)
