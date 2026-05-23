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

# License 1
WLSACCESSID_1="${WLSACCESSID_1:-9f1b1284-0ea6-423e-9658-9af7206ddedb}"
WLSSECRET_1="${WLSSECRET_1:-7e8a0849-bda0-4b5d-8cdf-3c8b5d5d2143}"
LICENSEID_1="${LICENSEID_1:-2817305}"

# License 2
WLSACCESSID_2="${WLSACCESSID_2:-eb0106ef-198e-43c9-bb56-233cc08191da}"
WLSSECRET_2="${WLSSECRET_2:-4418da38-fdaf-489e-a24f-27dfcaceb67a}"
LICENSEID_2="${LICENSEID_2:-2826830}"

WLSACCESSIDS=("${WLSACCESSID_1}" "${WLSACCESSID_2}")
WLSSECRETS=("${WLSSECRET_1}" "${WLSSECRET_2}")
LICENSEIDS=("${LICENSEID_1}" "${LICENSEID_2}")

license_idx=$(( SLURM_ARRAY_TASK_ID % 2 ))
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
