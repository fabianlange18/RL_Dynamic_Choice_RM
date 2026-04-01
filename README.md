# RL Choice Study

Reinforcement learning for pricing and assortment decisions in product choice environments.

## Quick Start

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run a single experiment:**
```bash
python example_2/exec.py
```

**Submit 24 parallel experiments to Slurm:**
```bash
sbatch example_2/slurm/submit_experiment_grid.sbatch
```

## Structure

- **example_1/** — Introductory example
- **example_2/** — Main codebase: RL training, EM estimation, DP solvers, evaluation
- **example_2/slurm/** — Batch job submission scripts for cluster runs
- **example_2/results/** — Model checkpoints, learning curves, evaluation results