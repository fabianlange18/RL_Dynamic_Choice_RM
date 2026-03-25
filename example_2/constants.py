import os
import numpy as np
import config as c

from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import ARS, QRDQN, TRPO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOTAL_TIMESTEPS = 1_000_000 if c.TRAIN_ON_ALL_SETS else 100_000
N_EVAL_EPISODES = 15
N_ESTIMATION_EPISODES = 50

T = 410
C = 185
n = 10
ARRIVAL_PROB = 0.5
r = np.asarray([600, 550, 475, 400, 300, 280, 240, 200, 185, 175], dtype=float)

LEARNING_CURVE_ENABLED = True

DEMAND_MODELS = (
    "MNL",
	"MMNL",
	"Probit",
	"MNLrefPrice",
	"MNLConsidSet",
	"NLogit",
)

RL_ALGORITHMS = {
    "DQN": DQN,
    "QRDQN": QRDQN,
    "ARS": ARS,
    "A2C": A2C,
    "TRPO": TRPO,
    "PPO": PPO,
}

MULTIBINARY_ALGORITHMS = {
    "A2C",
    "TRPO",
    "PPO"
}

SENSITIVITY_BETA_GT = {
	"low": -0.0015,
	"high": -0.005,
}

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "results",
    f"{c.GT_MODEL}_{c.OPT_MODEL}_{'high' if c.HIGH_SENSITIVITY else 'low'}_{'all' if c.TRAIN_ON_ALL_SETS else 'effsets'}",
)
