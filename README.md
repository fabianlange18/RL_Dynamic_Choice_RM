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

**Submit all experiments to Slurm:**
```bash
sh src/slurm/submit_two_phase.sh
```

**Retry experiments**
```bash
sbatch --array=67,71 --export=ALL,EXEC_PHASE=1_PREP --job-name=rl_choice_p1_prep_retry src/slurm/submit_phase1_dpadp.sbatch
```

## Structure

- **src/** — Codebase: RL training, EM estimation, DP solvers, evaluation
- **src/slurm/** — Batch job submission scripts for cluster runs
- **src/results/** — Model checkpoints, learning curves, evaluation results
