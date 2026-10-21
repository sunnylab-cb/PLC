#!/usr/bin/env bash
# Reference-only TEST, not a development signal. One frozen VAL-best epoch25
# checkpoint and its existing VAL gate. Real/shuffle/zero run sequentially.
# Report all/PDBbind/BioLiP, F1/MCC at0.5 and fixed VAL gate, AUPRC/AUROC.
# No TEST tuning, checkpoint comparison, model update, or silent row dropping.
# BF16 and batch8 are fixed before TEST; no performance-driven reruns.
#SBATCH --job-name=joint_reference_test
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=12:00:00

set -euo pipefail
ROOT="${REFERENCE_ROOT:?Set immutable reference evaluation root}"
ORIGINAL="${TRAINING_ROOT:?Set original training experiment root}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${ROOT}/code_v1:${ORIGINAL}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${ROOT}/triton_cache_${SLURM_JOB_ID}"
cd "${ROOT}/code_v1"
exec 9>"${ROOT}/evaluation.lock"
flock -n 9 || { echo "Reference evaluation already running" >&2; exit 2; }
"${PYTHON}" -u scripts/diagnostics/eval_joint_reference_test.py \
    --output "${ROOT}/results" --protocol-sha256 "${PROTOCOL_SHA256:?Set locked protocol hash}"
