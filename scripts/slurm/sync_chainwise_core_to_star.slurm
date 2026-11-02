#!/usr/bin/env bash
# CPU-only HPC -> Star transfer of core cached training data. NO ESM weights.
# PHASE=base copies available inputs. PHASE=final must depend on base + ESM job.
# Never launches training or uses a GPU; no delete or overwriting old Star data.
#SBATCH --job-name=star_core_sync
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=2-00:00:00
set -euo pipefail
BASE=/hpc2hdd/home/shuang886/PLC/experiments
SOURCE="${SOURCE:-${BASE}/chainwise_esm_fullcomplex_20261101}"
WORK="${WORK:-${BASE}/chainwise_core_star_export_20261102}"
DESTINATION="${DESTINATION:-/data1/PLC/experiments/chainwise_esm_core_20261102}"
CODE_ROOT="${CODE_ROOT:?Set immutable transfer code root}"
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
mkdir -p "${WORK}"
exec 9>"${WORK}/transfer.lock"
flock -n 9 || exit 2
"${PYTHON}" -u "${CODE_ROOT}/scripts/sync/sync_chainwise_core_to_star.py" "${PHASE:?base or final}" \
  --source "${SOURCE}" --work "${WORK}" --destination "${DESTINATION}"
