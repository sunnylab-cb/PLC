#!/usr/bin/env bash
# CPU-only fixed Top-K metrics from existing joint epoch25 TEST predictions.
# All K values are reported; no TEST tuning, model inference, or retraining.
# Top20 is the primary diagnostic. Inputs and the old reference remain intact.
#SBATCH --job-name=joint_topk_ref
#SBATCH --partition=i64m512ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
ROOT="${TOPK_ROOT:?Set isolated Top-K run root}"
INPUT="${REFERENCE_INPUT:?Set completed reference results directory}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec 9>"${ROOT}/analysis.lock"
flock -n 9 || { echo "Top-K analysis already running" >&2; exit 2; }
"${PYTHON}" -u "${ROOT}/code_v1/scripts/diagnostics/recompute_joint_topk_reference.py" \
    --input "${INPUT}" --output "${ROOT}/results"
