import argparse
import itertools
import runpy
import os
import sys
from pathlib import Path

GT_MODELS = ("MNL", "MMNL_5PT", "MMNL_2PT", "MMNLcont", "Probit", "MNLrefPrice", "MNLConsidSet", "TMNL", "NLogit") 
SENSITIVITIES = (False, True)
TRAIN_ON_ALL_SETS_OPTIONS = (False, True)
LARGE_PRODUCT_SET = (False, True)


def _mask(value):
    if not value:
        return "<unset>"
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def _print_gurobi_license_context():
    """Log non-sensitive Gurobi license context for SLURM debugging."""
    grb_license_file = os.environ.get("GRB_LICENSE_FILE")
    wls_access_id = os.environ.get("WLSACCESSID")
    license_id = os.environ.get("LICENSEID")

    print(
        "Gurobi preflight: GRB_LICENSE_FILE={}, exists={}, WLSACCESSID={}, LICENSEID={}".format(
            grb_license_file if grb_license_file else "<unset>",
            os.path.exists(grb_license_file) if grb_license_file else False,
            _mask(wls_access_id),
            license_id if license_id else "<unset>",
        )
    )

def build_grid():
    return list(itertools.product(SENSITIVITIES, GT_MODELS, TRAIN_ON_ALL_SETS_OPTIONS, LARGE_PRODUCT_SET))


def run_single(task_id):
    _print_gurobi_license_context()

    grid = build_grid()
    if task_id < 0 or task_id >= len(grid):
        raise ValueError(f"task_id {task_id} out of range [0, {len(grid) - 1}]")

    repo_root = Path(__file__).resolve().parents[2]
    example2_dir = repo_root / "example_2"

    sys.path.insert(0, str(example2_dir))

    import config as c
    c.HIGH_SENSITIVITY, c.GT_MODEL, c.TRAIN_ON_ALL_SETS, c.LARGE_PRODUCT_SET = grid[task_id]

    print(
        "Running task_id={} with HIGH_SENSITIVITY={}, GT_MODEL={}, TRAIN_ON_ALL_SETS={}, LARGE_PRODUCT_SET={}".format(
            task_id,
            c.HIGH_SENSITIVITY,
            c.GT_MODEL,
            c.TRAIN_ON_ALL_SETS,
            c.LARGE_PRODUCT_SET,
        )
    )

    # Set TASK_ID in environment for logging
    os.environ["TASK_ID"] = str(task_id)
    
    # Run exec.py
    exec_path = example2_dir / "exec_RL.py"
    runpy.run_path(str(exec_path), run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one experiment from the 72-case grid.")
    parser.add_argument("--task-id", type=int, required=True, help="Index in [0, 71] for the experiment grid")
    args = parser.parse_args()

    run_single(args.task_id)
