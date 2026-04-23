import argparse
import itertools
import runpy
import sys
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

    sys.path.insert(0, str(example2_dir))

    import config as c

    c.HIGH_SENSITIVITY = bool(high_sensitivity)
    c.GT_MODEL = str(gt_model)
    c.TRAIN_ON_ALL_SETS = bool(train_on_all_sets)

    print(
        "Running task_id={} with HIGH_SENSITIVITY={}, GT_MODEL={}, TRAIN_ON_ALL_SETS={}".format(
            task_id,
            c.HIGH_SENSITIVITY,
            c.GT_MODEL,
            c.TRAIN_ON_ALL_SETS,
        )
    )

    exec_path = example2_dir / "exec.py"
    runpy.run_path(str(exec_path), run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one experiment from the 24-case grid.")
    parser.add_argument("--task-id", type=int, required=True, help="Index in [0, 23] for the experiment grid")
    args = parser.parse_args()

    run_single(args.task_id)
