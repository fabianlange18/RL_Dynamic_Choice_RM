#!/bin/bash
# Submit all experiments:
# - regular grid jobs (single-pass execution for non-split task IDs)
# - split two-phase pipeline for large-classical task IDs
# Usage: ./src/slurm/submit_two_phase.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$REPO_ROOT"

echo "=========================================="
echo "RL Choice Study: All-Experiments Submission"
echo "=========================================="
echo ""

# Submit regular grid jobs (non-split IDs).
echo "Submitting regular grid jobs (all non-split task IDs)..."
REGULAR_OUTPUT=$(sbatch --job-name=rl_choice_grid_all src/slurm/submit_experiment_grid.sbatch)
REGULAR_JOB_ID=$(echo "$REGULAR_OUTPUT" | awk '{print $4}')

echo "Regular Grid Job ID: $REGULAR_JOB_ID"
echo "$REGULAR_OUTPUT"
echo ""

# Submit split pipeline for large-classical IDs.
echo "Submitting split pipeline for large-classical task IDs..."
echo "Submitting Phase 1 PREP (sampling, estimation, efficient sets)..."
PHASE1_PREP_OUTPUT=$(sbatch --export=ALL,EXEC_PHASE=1_PREP --job-name=rl_choice_p1_prep src/slurm/submit_phase1_dpadp.sbatch)
PHASE1_PREP_JOB_ID=$(echo "$PHASE1_PREP_OUTPUT" | awk '{print $4}')

echo "Phase 1 PREP Job ID: $PHASE1_PREP_JOB_ID"
echo "$PHASE1_PREP_OUTPUT"
echo ""

# Submit Phase 1 DP_MNL
echo "Submitting Phase 1 DP_MNL with dependency on PREP..."
PHASE1_DP_MNL_OUTPUT=$(sbatch --dependency=afterok:$PHASE1_PREP_JOB_ID --export=ALL,EXEC_PHASE=1_DP_MNL --job-name=rl_choice_p1_dpmnl src/slurm/submit_phase1_dpadp.sbatch)
PHASE1_DP_MNL_JOB_ID=$(echo "$PHASE1_DP_MNL_OUTPUT" | awk '{print $4}')

echo "Phase 1 DP_MNL Job ID: $PHASE1_DP_MNL_JOB_ID (depends on $PHASE1_PREP_JOB_ID)"
echo "$PHASE1_DP_MNL_OUTPUT"
echo ""

# Submit Phase 1 DP_MMNL_5PT
echo "Submitting Phase 1 DP_MMNL_5PT with dependency on DP_MNL..."
PHASE1_DP_MMNL5_OUTPUT=$(sbatch --dependency=afterok:$PHASE1_DP_MNL_JOB_ID --export=ALL,EXEC_PHASE=1_DP_MMNL_5PT --job-name=rl_choice_p1_dpmmnl5 src/slurm/submit_phase1_dpadp.sbatch)
PHASE1_DP_MMNL5_JOB_ID=$(echo "$PHASE1_DP_MMNL5_OUTPUT" | awk '{print $4}')

echo "Phase 1 DP_MMNL_5PT Job ID: $PHASE1_DP_MMNL5_JOB_ID (depends on $PHASE1_DP_MNL_JOB_ID)"
echo "$PHASE1_DP_MMNL5_OUTPUT"
echo ""

# Submit Phase 1 DP_MMNL_2PT
echo "Submitting Phase 1 DP_MMNL_2PT with dependency on DP_MMNL_5PT..."
PHASE1_DP_MMNL2_OUTPUT=$(sbatch --dependency=afterok:$PHASE1_DP_MMNL5_JOB_ID --export=ALL,EXEC_PHASE=1_DP_MMNL_2PT --job-name=rl_choice_p1_dpmmnl2 src/slurm/submit_phase1_dpadp.sbatch)
PHASE1_DP_MMNL2_JOB_ID=$(echo "$PHASE1_DP_MMNL2_OUTPUT" | awk '{print $4}')

echo "Phase 1 DP_MMNL_2PT Job ID: $PHASE1_DP_MMNL2_JOB_ID (depends on $PHASE1_DP_MMNL5_JOB_ID)"
echo "$PHASE1_DP_MMNL2_OUTPUT"
echo ""

# Submit Phase 2 with dependency on all phase 1 DP sub-steps
echo "Submitting Phase 2 (RL training + evaluation) with dependency on DP_MMNL_2PT..."
PHASE2_OUTPUT=$(sbatch --dependency=afterok:$PHASE1_DP_MMNL2_JOB_ID src/slurm/submit_phase2_rl_eval.sbatch)
PHASE2_JOB_ID=$(echo "$PHASE2_OUTPUT" | awk '{print $4}')

echo "Phase 2 Job ID: $PHASE2_JOB_ID (depends on $PHASE1_DP_MMNL2_JOB_ID)"
echo "$PHASE2_OUTPUT"
echo ""

echo "=========================================="
echo "Job Submission Summary"
echo "=========================================="
echo "Regular Grid (non-split IDs): Job ID $REGULAR_JOB_ID (72 hours)"
echo ""
echo "Split Pipeline (large-classical IDs only):"
echo "Phase 1 PREP:        Job ID $PHASE1_PREP_JOB_ID (72 hours)"
echo "Phase 1 DP_MNL:      Job ID $PHASE1_DP_MNL_JOB_ID (72 hours, starts after PREP)"
echo "Phase 1 DP_MMNL_5PT: Job ID $PHASE1_DP_MMNL5_JOB_ID (72 hours, starts after DP_MNL)"
echo "Phase 1 DP_MMNL_2PT: Job ID $PHASE1_DP_MMNL2_JOB_ID (72 hours, starts after DP_MMNL_5PT)"
echo "Phase 2 (RL+Eval):   Job ID $PHASE2_JOB_ID (72 hours, starts after DP_MMNL_2PT)"
echo ""
echo "Monitor jobs with: squeue -j $REGULAR_JOB_ID,$PHASE1_PREP_JOB_ID,$PHASE1_DP_MNL_JOB_ID,$PHASE1_DP_MMNL5_JOB_ID,$PHASE1_DP_MMNL2_JOB_ID,$PHASE2_JOB_ID"
echo "View logs: tail -f src/slurm/logs/rl_choice_*.out"
echo "=========================================="
