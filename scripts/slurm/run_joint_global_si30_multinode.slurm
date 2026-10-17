#!/usr/bin/env bash
# Four Intel ue nodes, 64 processes/node. Both all-pairs SI and independent
# cross-set audit are sharded. Reuse completed, hash-verified legacy chunks.
# Run only after the previous single-node writer exits; downstream preparation
# must depend on this job's successful completion, not on the cancelled job.
#SBATCH --job-name=joint_si_4nodes
#SBATCH --partition=i64m512ue
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00

set -euo pipefail
: "${EXPERIMENT:?}" "${POOL:?}" "${MULTINODE_CODE:?}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${EXPERIMENT}/code_v1:${EXPERIMENT}/deps:${PYTHONPATH:-}"
cd "${EXPERIMENT}/code_v1"
exec 9>"${EXPERIMENT}/preparation.lock"
flock -n 9 || { echo "Previous SI writer is still active" >&2; exit 2; }
/hpc2ssd/softwares/anaconda3/envs/af3/bin/python -u \
    "${MULTINODE_CODE}/scripts/diagnostics/run_internal_pairwise_multinode.py" controller \
    --experiment "${EXPERIMENT}" --pool-dir "${POOL}" --nodes 4 --workers 64
