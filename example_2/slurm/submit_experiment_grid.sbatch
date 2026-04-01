#!/bin/bash
#SBATCH --job-name=rl_choice_grid
#SBATCH --array=0-23
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --output=example_2/slurm/logs/%x_%A_%a.out
#SBATCH --error=example_2/slurm/logs/%x_%A_%a.err

set -euo pipefail

# Run from repository root
cd "${SLURM_SUBMIT_DIR}"

mkdir -p example_2/slurm/logs

# Activate your environment (adjust as needed for your cluster)
# Example with venv:
# source .rl_venv/bin/activate
# Example with conda:
# conda activate <your_env>

python example_2/slurm/run_experiment_grid.py --task-id "${SLURM_ARRAY_TASK_ID}"
