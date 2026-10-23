#!/usr/bin/env bash
# Latest joint VAL-best epoch25 -> HOLO4K, fixed VAL gate and fixed Top20.
# Neither HOLO4K nor COACH420 was excluded from this training cohort.
# Reference only: no homology-clean claim, benchmark tuning or model changes.
# Re-normalize RAW spatial/ligand features using joint TRAIN-only statistics.
# Prepare on i64m512ue, then one GPU on i64m1tga800ue after successful QC.
# Reuse frozen LiBRe predictions; score identical 7238 pairs plus PLC full QC.
# No ligand atom cap. BF16/batch2 fixed before evaluation; fail on bad logits.
#SBATCH --job-name=joint_holo_reference
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00

set -euo pipefail
: "${EVAL_ROOT:?}" "${PHASE:?}"
BASE=/hpc2hdd/home/shuang886/PLC
TRAIN="${BASE}/experiments/joint_global_si30_gfgf_20261017"
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export PYTHONPATH="${EVAL_ROOT}/code_v1:${TRAIN}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${EVAL_ROOT}/triton_${SLURM_JOB_ID}"
cd "${EVAL_ROOT}/code_v1"
exec 9>"${EVAL_ROOT}/${PHASE}.lock"
flock -n 9 || exit 2
if [[ "${PHASE}" == prepare ]]; then export CUDA_VISIBLE_DEVICES=''; fi
"${PYTHON}" -u scripts/benchmark/eval_joint_holo_reference.py "${PHASE}" \
  --work "${EVAL_ROOT}" --training "${TRAIN}" \
  --reference "${BASE}/experiments/joint_reference_test_20261021/results" \
  --benchmark /hpc2hdd/home/shuang886/jhspoolers/PLC/benchmark \
  --libre "${BASE}/external_baselines/libre/results/benchmark_12960610/holo4k" \
  --workers 4
