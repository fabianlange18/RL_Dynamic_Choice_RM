#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --job-name=rl_choice_grid
#SBATCH --array=0-63
#SBATCH --time=72:00:00
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

# Four WLS licenses. Keep defaults for license 1, and override via env vars when needed.
WLSACCESSID_1="${WLSACCESSID_1:-9f1b1284-0ea6-423e-9658-9af7206ddedb}"
WLSSECRET_1="${WLSSECRET_1:-7e8a0849-bda0-4b5d-8cdf-3c8b5d5d2143}"
LICENSEID_1="${LICENSEID_1:-2817305}"

# Replace these defaults with your other licenses (or set via exported env vars).
WLSACCESSID_2="${WLSACCESSID_2:-9f1b1284-0ea6-423e-9658-9af7206ddedb}"
WLSSECRET_2="${WLSSECRET_2:-7e8a0849-bda0-4b5d-8cdf-3c8b5d5d2143}"
LICENSEID_2="${LICENSEID_2:-2817305}"

WLSACCESSID_3="${WLSACCESSID_3:-9f1b1284-0ea6-423e-9658-9af7206ddedb}"
WLSSECRET_3="${WLSSECRET_3:-7e8a0849-bda0-4b5d-8cdf-3c8b5d5d2143}"
LICENSEID_3="${LICENSEID_3:-2817305}"

WLSACCESSID_4="${WLSACCESSID_4:-9f1b1284-0ea6-423e-9658-9af7206ddedb}"
WLSSECRET_4="${WLSSECRET_4:-7e8a0849-bda0-4b5d-8cdf-3c8b5d5d2143}"
LICENSEID_4="${LICENSEID_4:-2817305}"

WLSACCESSIDS=("${WLSACCESSID_1}" "${WLSACCESSID_2}" "${WLSACCESSID_3}" "${WLSACCESSID_4}")
WLSSECRETS=("${WLSSECRET_1}" "${WLSSECRET_2}" "${WLSSECRET_3}" "${WLSSECRET_4}")
LICENSEIDS=("${LICENSEID_1}" "${LICENSEID_2}" "${LICENSEID_3}" "${LICENSEID_4}")

license_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))
export WLSACCESSID="${WLSACCESSIDS[$license_idx]}"
export WLSSECRET="${WLSSECRETS[$license_idx]}"
export LICENSEID="${LICENSEIDS[$license_idx]}"

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
echo "Using Gurobi license file: ${GRB_LICENSE_FILE} (license index=${license_idx}, LICENSEID=${LICENSEID})"

mkdir -p src/slurm/logs

source .rl_venv/bin/activate

export GUROBI_MAX_SLOTS=2
export GUROBI_SLOT_LOCK_DIR="${SLURM_SUBMIT_DIR}/src/slurm/.gurobi_slots_license_${LICENSEID}"
mkdir -p "${GUROBI_SLOT_LOCK_DIR}"

python src/slurm/run_experiment_grid.py --task-id "${SLURM_ARRAY_TASK_ID}"
