#!/usr/bin/env bash
# Frozen joint GFGF seed44: ALL-chain HOLO4K context diagnostic, no training.
# Slice cached protein features by chain, infer each chain with the same ligand,
# stitch onto original <=1800 residue axis; global Top20/frozen VAL gate.
# GT-binding-chain metrics are separately labelled oracle, not blind inference.
# CPU prepare -> one UE GPU inference -> CPU summarize; do not tune on HOLO.
# Invoke with EVAL_ROOT and PHASE; submit GPU phase with --gres=gpu:1 only.
#SBATCH --job-name=holo_chain_context
#SBATCH --partition=i64m512ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00

set -euo pipefail
: "${EVAL_ROOT:?}" "${PHASE:?}"
BASE=/hpc2hdd/home/shuang886/PLC/experiments
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export PYTHONPATH="${EVAL_ROOT}/code_v1:${BASE}/joint_global_si30_gfgf_20261017/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${EVAL_ROOT}/triton_${SLURM_JOB_ID}"
cd "${EVAL_ROOT}/code_v1"
exec 9>"${EVAL_ROOT}/${PHASE}.lock"
flock -n 9 || exit 2
if [[ "${PHASE}" != infer ]]; then export CUDA_VISIBLE_DEVICES=''; fi
"${PYTHON}" -u scripts/benchmark/eval_holo_chain_context.py "${PHASE}" \
  --work "${EVAL_ROOT}/results" \
  --parent "${BASE}/joint_holo_reference_20261023" \
  --cohorts "${BASE}/joint_holo_nonshared_20261023/results" --workers 4
