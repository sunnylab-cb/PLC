#!/usr/bin/env bash
# Frozen strict base44 site-assignment trials, NOT backbone resume/training.
# STAGE=smoke: small token export + all controls, must pass before full export.
# STAGE=export: full strict TRAIN/VAL token cache shared by every head branch.
# STAGE=head: array 0=original quality controls, 1=gain, 2=tokens, 3=boundary.
# Each branch runs its matched controls sequentially on ONE GPU; use array %3.
# No TEST/COACH/HOLO access, no GT candidates, no shared ligand-protein frame.
# Outputs/cache live in HDD experiments, never jhspoolers. No torch.compile.
# ACCEPTANCE_POLICY selects the explicit historical or practical audit policy.
#SBATCH --job-name=site_parallel
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
EXPERIMENT="${EXPERIMENT:-${ARCHIVE}/experiments/strict_site_parallel_20260926}"
CODE="${CODE:-${EXPERIMENT}/code_pilots_v1}"
PRIOR="${ARCHIVE}/experiments/strict_assignment_20260919"
DATA="${ARCHIVE}/datasets/pdbbind_biolip_chain_si30_final"
CKPT="${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_scratch_seed44_12222454_1_v11m13_20260826_055836/best_f1_c0_model.pt"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
OUT="${EXPERIMENT}/run_${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-none}"
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${CODE}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
EXTRA=()
PRECISION_ARGS=()
if [[ -n "${INFERENCE_PROTOCOL:-}" ]]; then
  PRECISION_ARGS=(--inference-protocol "${INFERENCE_PROTOCOL}")
  if [[ "${INFERENCE_PROTOCOL}" != legacy_bf16 && "${STAGE:-smoke}" != head ]]; then
    : "${PRECISION_AUDIT:?Set a successful full VAL precision audit summary}"
    PRECISION_ARGS+=(--precision-audit "${PRECISION_AUDIT}")
  fi
fi
if [[ -n "${ACCEPTANCE_POLICY:-}" ]]; then
  PRECISION_ARGS+=(--acceptance-policy "${ACCEPTANCE_POLICY}")
fi
EPOCHS=20
if [[ "${STAGE:-smoke}" == smoke ]]; then EXTRA=(--smoke); EPOCHS=2; fi
if [[ "${STAGE:-smoke}" != head ]]; then
  CACHE="${OUT}/cache"
  "${PYTHON}" -u scripts/diagnostics/export_site_quality_cache.py \
    --checkpoint "${CKPT}" --root "${DATA}" --tokens \
    --label-alignment "${PRIOR}/label_alignment_verified.json" \
    --reference-val "${PRIOR}/val_12804287/base44" --output "${CACHE}" "${EXTRA[@]}" "${PRECISION_ARGS[@]}"
fi
if [[ "${STAGE:-smoke}" == export ]]; then
  echo "SHARED_CACHE_COMPLETE=${CACHE}"
  exit 0
fi
if [[ "${STAGE:-smoke}" == head ]]; then
  : "${CACHE:?Set exact successful export cache path}"
  IDS=("${SLURM_ARRAY_TASK_ID:?Array task 0-3 required}")
else
  IDS=(0 1 2 3)
fi
for ID in "${IDS[@]}"; do
  if [[ "${ID}" == 0 ]]; then
    "${PYTHON}" -u scripts/train/train_site_quality_probe.py --cache "${CACHE}" \
      --splits "${DATA}/splits.txt" --output "${OUT}/baseline" --epochs "${EPOCHS}" \
      --batch-size 32 --seed 44 "${EXTRA[@]}"
  else
    BRANCHES=(unused gain tokens boundary)
    BRANCH="${BRANCHES[$ID]}"
    "${PYTHON}" -u scripts/train/train_site_parallel.py --cache "${CACHE}" \
      --splits "${DATA}/splits.txt" --output "${OUT}/${BRANCH}" --branch "${BRANCH}" \
      --epochs "${EPOCHS}" --batch-size 8 --lr 0.0002 --margin 0.02 --seed 44 "${EXTRA[@]}"
  fi
done
echo "SITE_PARALLEL_JOB_COMPLETE=${OUT}"
