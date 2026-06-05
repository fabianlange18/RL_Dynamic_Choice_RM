#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=rl_choice_phase2
#SBATCH --array=3,7,11,15,19,23,27,31,35,39,43,47,51,55,59,63,67,71
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --output=src/slurm/logs/%x_%A_%a.out
#SBATCH --error=src/slurm/logs/%x_%A_%a.err

set -euo pipefail

# Run from repository root
cd "${SLURM_SUBMIT_DIR}"

export XDG_CACHE_HOME=$TMPDIR/.cache
mkdir -p $XDG_CACHE_HOME/arviz

export PYTHONWARNINGS="ignore::FutureWarning"

mkdir -p src/slurm/logs

source .rl_venv/bin/activate

export EXEC_PHASE=2

python src/slurm/run_experiment_grid.py --task-id "${SLURM_ARRAY_TASK_ID}"