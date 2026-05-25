import os
import numpy as np
import src.config as c

from stable_baselines3 import PPO, DQN, A2C
from sb3_contrib import ARS, QRDQN, TRPO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TOTAL_TIMESTEPS = [100_000, 500_000, 1_000_000]
if c.TRAIN_ON_ALL_SETS and c.LARGE_PRODUCT_SET:
    TOTAL_TIMESTEPS = [10 * steps for steps in TOTAL_TIMESTEPS]

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
	"Probit",
	"MNLrefPrice",
	"MNLConsidSet",
    "TMNL",
	"NLogit"
)

# Demand-model parameters
MMNL_2PT_BETA_MULTIPLIERS = np.asarray([0.5, 1.5], dtype=float)
MMNL_5PT_BETA_MULTIPLIERS = np.asarray([0.6, 0.8, 1.0, 1.2, 1.4], dtype=float)

MNL_REFERENCE_PRICE_BETA_GAIN = 0.0025
MNL_REFERENCE_PRICE_BETA_LOSS = 0.0035

MNL_CONSIDERATION_LOGIT_SLOPE = 0.0125
MNL_CONSIDERATION_EXACT_MAX_PRODUCTS = 20
MNL_CONSIDERATION_QUADRATURE_POINTS = 64
MNL_CONSIDERATION_MONTE_CARLO_DRAWS = 5000
MNL_CONSIDERATION_NORMALIZATION_TOLERANCE = 1e-10
MNL_CONSIDERATION_SUBSET_CACHE_SIZE = 32
MNL_CONSIDERATION_QUADRATURE_CACHE_SIZE = 8

NESTED_LOGIT_MU_A = 0.7
NESTED_LOGIT_MU_B = 0.8

TMNL_DEFAULT_DELTA = 0.5
TMNL_OUTSIDE_UTILITY = -1.0

# Gurobi solver constants
MAX_STATE_SOLVE_SECONDS = 1.0
MAX_EFFICIENT_SET_SOLVE_SECONDS = 60.0
GUROBI_SUPPORTED_MODELS = ("MNL", "MMNL_5PT", "MMNL_2PT")

# Efficient-frontier search constants
EFFICIENT_SET_FRONTIER_Q_EPS = 1e-8
EFFICIENT_SET_DINKELBACH_TOL = 1e-8
EFFICIENT_SET_DINKELBACH_MAX_ITER = 50
EFFICIENT_SET_NUMERIC_TOL = 1e-10

# Estimation constants
ESTIMATION_BETA_BOUNDS = (-0.05, -1e-7)

# Experiment-grid constants
GRID_SENSITIVITIES = (False, True)
GRID_TRAIN_ON_ALL_SETS_OPTIONS = (False, True)
GRID_LARGE_PRODUCT_SET_OPTIONS = (False, True)

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
