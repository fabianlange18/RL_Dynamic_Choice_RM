#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=rl_choice_grid
#SBATCH --array=0-23
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=2G
#SBATCH --output=example_2/slurm/logs/%x_%A_%a.out
#SBATCH --error=example_2/slurm/logs/%x_%A_%a.err
#SBATCH --mail-user=fabian.lange@uni-mannheim.de
#SBATCH --mail-type=END

set -euo pipefail

# Run from repository root
cd "${SLURM_SUBMIT_DIR}"

mkdir -p example_2/slurm/logs

# Activate your environment (adjust as needed for your cluster)
# Example with venv:
source .rl_venv/bin/activate
# Example with conda:
#module load devel/miniforge/25.3.1-python-3.12
#source "$(conda info --base)/etc/profile.d/conda.sh"
#conda activate rl_venv

python example_2/slurm/run_experiment_grid.py --task-id "${SLURM_ARRAY_TASK_ID}"
