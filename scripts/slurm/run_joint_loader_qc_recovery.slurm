#!/usr/bin/env bash
# Reuse completed SI, normalization and materialized features. Rerun ALL loader
# QC with the shadowing fix; do not skip checks or modify the training recipe.
# The patched preparation file is included in the resulting asset hashes.
#SBATCH --job-name=joint_loader_qc
#SBATCH --partition=i64m512ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
set -euo pipefail
: "${EXPERIMENT:?}" "${POOL:?}" "${BASE:?}" "${RECOVERY_CODE:?}"
CODE="${EXPERIMENT}/code_v1"
export PYTHONPATH="${CODE}/scripts/diagnostics:${CODE}:${EXPERIMENT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${CODE}"
exec 9>"${EXPERIMENT}/preparation.lock"
flock -n 9 || exit 2
/hpc2ssd/softwares/anaconda3/envs/af3/bin/python -u \
    "${RECOVERY_CODE}/scripts/diagnostics/prepare_internal_gfgf.py" \
    --experiment "${EXPERIMENT}" --pool-dir "${POOL}" --base "${BASE}" \
    --cohort pdbbind_biolip_joint
