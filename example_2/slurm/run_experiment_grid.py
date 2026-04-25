import argparse
import itertools
import subprocess
import sys
import os
from pathlib import Path


GT_MODELS = ("MNL", "MMNL_5PT", "MMNL_2PT", "MMNLcont", "Probit", "MNLrefPrice", "MNLConsidSet", "TMNL", "NLogit") 
SENSITIVITIES = (False, True)
TRAIN_ON_ALL_SETS_OPTIONS = (False, True)


def build_grid():
    return list(itertools.product(SENSITIVITIES, GT_MODELS, TRAIN_ON_ALL_SETS_OPTIONS))


def run_single(task_id):
    grid = build_grid()
    if task_id < 0 or task_id >= len(grid):
        raise ValueError(f"task_id {task_id} out of range [0, {len(grid) - 1}]")

    high_sensitivity, gt_model, train_on_all_sets = grid[task_id]

    repo_root = Path(__file__).resolve().parents[2]
    example2_dir = repo_root / "example_2"

    print(
        "Running task_id={} with HIGH_SENSITIVITY={}, GT_MODEL={}, TRAIN_ON_ALL_SETS={}".format(
            task_id,
            high_sensitivity,
            gt_model,
            train_on_all_sets,
        )
    )

    # Use subprocess.run to properly execute exec.py with multiprocessing guard
    exec_path = example2_dir / "exec.py"
    env = os.environ.copy()
    env.update({
        "HIGH_SENSITIVITY": str(high_sensitivity),
        "GT_MODEL": str(gt_model),
        "TRAIN_ON_ALL_SETS": str(train_on_all_sets),
    })
    
    result = subprocess.run(
        [sys.executable, str(exec_path)],
        cwd=str(example2_dir),
        env=env,
        capture_output=False,
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"exec.py failed with return code {result.returncode}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one experiment from the 36-case grid.")
    parser.add_argument("--task-id", type=int, required=True, help="Index in [0, 35] for the experiment grid")
    args = parser.parse_args()

    run_single(args.task_id)
