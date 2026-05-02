import os

# Read from environment variables if set, otherwise use defaults
HIGH_SENSITIVITY = os.environ.get("HIGH_SENSITIVITY", "True").lower() in ("true", "1")
GT_MODEL = os.environ.get("GT_MODEL", "MNL")  # must be one of MNL, MMNL_5PT, MMNL_2PT, MMNLcont, Probit, MNLrefPrice, MNLConsidSet, TMNL, NLogit
TRAIN_ON_ALL_SETS = os.environ.get("TRAIN_ON_ALL_SETS", "True").lower() in ("true", "1")