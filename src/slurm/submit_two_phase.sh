#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$REPO_ROOT"

REGULAR_JOB_ID=$(sbatch --job-name=rl_choice_grid_all src/slurm/submit_experiment_grid.sbatch | awk '{print $4}')

PHASE1_PREP_JOB_ID=$(sbatch --export=ALL,EXEC_PHASE=1_PREP --job-name=rl_choice_p1_prep src/slurm/submit_phase1_dpadp.sbatch | awk '{print $4}')
PHASE1_DP_MNL_JOB_ID=$(sbatch --dependency=afterok:$PHASE1_PREP_JOB_ID --export=ALL,EXEC_PHASE=1_DP_MNL --job-name=rl_choice_p1_dpmnl src/slurm/submit_phase1_dpadp.sbatch | awk '{print $4}')
PHASE1_DP_MMNL5_JOB_ID=$(sbatch --dependency=afterok:$PHASE1_DP_MNL_JOB_ID --export=ALL,EXEC_PHASE=1_DP_MMNL_5PT --job-name=rl_choice_p1_dpmmnl5 src/slurm/submit_phase1_dpadp.sbatch | awk '{print $4}')
PHASE1_DP_MMNL2_JOB_ID=$(sbatch --dependency=afterok:$PHASE1_DP_MMNL5_JOB_ID --export=ALL,EXEC_PHASE=1_DP_MMNL_2PT --job-name=rl_choice_p1_dpmmnl2 src/slurm/submit_phase1_dpadp.sbatch | awk '{print $4}')
PHASE2_JOB_ID=$(sbatch --dependency=afterok:$PHASE1_DP_MMNL2_JOB_ID src/slurm/submit_phase2_rl_eval.sbatch | awk '{print $4}')