#!/bin/bash
# Submit two-phase job pipeline with dependency management
# Usage: ./src/slurm/submit_two_phase.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

cd "$REPO_ROOT"

echo "=========================================="
echo "RL Choice Study: Two-Phase Job Submission"
echo "=========================================="
echo ""

# Submit Phase 1
echo "Submitting Phase 1 (DP/ADP computation)..."
PHASE1_OUTPUT=$(sbatch src/slurm/submit_phase1_dpadp.sbatch)
PHASE1_JOB_ID=$(echo "$PHASE1_OUTPUT" | awk '{print $4}')

echo "Phase 1 Job ID: $PHASE1_JOB_ID"
echo "$PHASE1_OUTPUT"
echo ""

# Submit Phase 2 with dependency on Phase 1
echo "Submitting Phase 2 (RL training + evaluation) with dependency on Phase 1..."
PHASE2_OUTPUT=$(sbatch --dependency=afterok:$PHASE1_JOB_ID src/slurm/submit_phase2_rl_eval.sbatch)
PHASE2_JOB_ID=$(echo "$PHASE2_OUTPUT" | awk '{print $4}')

echo "Phase 2 Job ID: $PHASE2_JOB_ID (depends on $PHASE1_JOB_ID)"
echo "$PHASE2_OUTPUT"
echo ""

echo "=========================================="
echo "Job Submission Summary"
echo "=========================================="
echo "Phase 1 (DP/ADP):    Job ID $PHASE1_JOB_ID (72 hours)"
echo "Phase 2 (RL+Eval):   Job ID $PHASE2_JOB_ID (72 hours, starts after Phase 1)"
echo ""
echo "Monitor jobs with: squeue -j $PHASE1_JOB_ID,$PHASE2_JOB_ID"
echo "View logs: tail -f src/slurm/logs/rl_choice_phase*.log"
echo "=========================================="
