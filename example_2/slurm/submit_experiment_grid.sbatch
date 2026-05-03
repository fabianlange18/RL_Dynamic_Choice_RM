#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=rl_choice_grid
#SBATCH --array=0-71
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --output=example_2/slurm/logs/%x_%A_%a.out
#SBATCH --error=example_2/slurm/logs/%x_%A_%a.err
#SBATCH --mail-user=fabian.lange@uni-mannheim.de
#SBATCH --mail-type=END

set -euo pipefail

# Run from repository root
cd "${SLURM_SUBMIT_DIR}"

export XDG_CACHE_HOME=$TMPDIR/.cache
mkdir -p $XDG_CACHE_HOME/arviz

export PYTHONWARNINGS="ignore::FutureWarning"
export WLSACCESSID=9f1b1284-0ea6-423e-9658-9af7206ddedb
export WLSSECRET=7e8a0849-bda0-4b5d-8cdf-3c8b5d5d2143
export LICENSEID=2817305

# Force Gurobi to use this WLS license in batch jobs.
LICENSE_DIR="${TMPDIR:-/tmp}/gurobi_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "${LICENSE_DIR}"
chmod 700 "${LICENSE_DIR}"
export GRB_LICENSE_FILE="${LICENSE_DIR}/gurobi.lic"
cat > "${GRB_LICENSE_FILE}" <<EOF
WLSACCESSID=${WLSACCESSID}
WLSSECRET=${WLSSECRET}
LICENSEID=${LICENSEID}
EOF
chmod 600 "${GRB_LICENSE_FILE}"
echo "Using Gurobi license file: ${GRB_LICENSE_FILE} (LICENSEID=${LICENSEID})"

mkdir -p example_2/slurm/logs

source .rl_venv/bin/activate

python example_2/slurm/run_experiment_grid.py --task-id "${SLURM_ARRAY_TASK_ID}"
