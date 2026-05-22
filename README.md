# RL Choice Study

Reinforcement learning for pricing and assortment decisions in product choice environments.

## Quick Start

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Run a single experiment:**
```bash
python src/exec.py
```

**Submit 24 parallel experiments to Slurm:**
```bash
sbatch src/slurm/submit_experiment_grid.sbatch
```

## Structure

- **src/** — Codebase: RL training, EM estimation, DP solvers, evaluation
- **src/slurm/** — Batch job submission scripts for cluster runs
- **src/results/** — Model checkpoints, learning curves, evaluation results