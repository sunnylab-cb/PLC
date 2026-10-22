#!/usr/bin/env bash
# Released LiBRe vs latest frozen PLC joint TEST. No retraining or TEST tuning.
# Submit phase=prepare on i64m512ue, then phase=infer with afterok on A800-ue.
# Overlap is accepted, never filtered. Input failures remain explicit and both
# models are compared on identical successful rows; no silent all-row claim.
#SBATCH --job-name=libre_joint_test
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

set -euo pipefail
WORK="${LIBRE_JOINT_ROOT:?Set isolated run directory}"
PHASE="${PHASE:?Set prepare or infer}"
PYTHON="/hpc2ssd/softwares/anaconda3/envs/af3/bin/python"
BASE="/hpc2hdd/home/shuang886/PLC"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "${WORK}/code_v1"
exec 9>"${WORK}/${PHASE}.lock"
flock -n 9 || { echo "Duplicate phase already running" >&2; exit 2; }
if [[ "${PHASE}" == prepare ]]; then
  export CUDA_VISIBLE_DEVICES=''
elif [[ "${PHASE}" != infer ]]; then
  echo "Invalid phase" >&2
  exit 2
fi
"${PYTHON}" -u scripts/benchmark/run_libre_joint_test.py "${PHASE}" \
  --work "${WORK}" --libre "${BASE}/external_baselines/libre" \
  --reference "${BASE}/experiments/joint_reference_test_20261021/results" \
  --pool "${BASE}/experiments/biolip_feature_expansion_20261015/qc_pool" \
  --sdf-root "${WORK}/pdbbind_sdf" --workers "${SLURM_CPUS_PER_TASK:-8}"
