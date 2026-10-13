#!/usr/bin/env bash
# Strict TRAIN/VAL candidate-quality pilot, NOT a new backbone training run.
# Frozen base44 -> audited pre-cross features + same hybrid proposal pool.
# Train three new heads: stats-only / protein-site / ligand-site, sequentially.
# Actual output selects a mask only above a fixed 0.02 predicted-quality margin;
# original prediction is always retained. No test/COACH/HOLO or native cross-distance.
# SMOKE=1: small cache incl all ten label overlays, three head epochs; must pass
# before full job via Slurm afterok. Main caches remain in HDD, not jhspoolers.
#SBATCH --job-name=strict_sitequality
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00

set -euo pipefail
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
EXPERIMENT="${EXPERIMENT:-${ARCHIVE}/experiments/strict_site_quality_20260926}"
CODE="${EXPERIMENT}/code"
PRIOR="${ARCHIVE}/experiments/strict_assignment_20260919"
DATA="${ARCHIVE}/datasets/pdbbind_biolip_chain_si30_final"
CKPT="${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_scratch_seed44_12222454_1_v11m13_20260826_055836/best_f1_c0_model.pt"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
OUT="${EXPERIMENT}/run_${SLURM_JOB_ID:-manual}"
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${CODE}"
ARGS=()
EPOCHS=20
if [[ "${SMOKE:-0}" == 1 ]]; then ARGS=(--smoke); EPOCHS=3; fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
"${PYTHON}" -u scripts/diagnostics/export_site_quality_cache.py \
  --checkpoint "${CKPT}" --root "${DATA}" \
  --label-alignment "${PRIOR}/label_alignment_verified.json" \
  --reference-val "${PRIOR}/val_12804287/base44" --output "${OUT}/cache" "${ARGS[@]}"
"${PYTHON}" -u scripts/train/train_site_quality_probe.py \
  --cache "${OUT}/cache" --splits "${DATA}/splits.txt" --output "${OUT}/heads" \
  --epochs "${EPOCHS}" --batch-size 32 --lr 0.0002 --margin 0.02 --seed 44 "${ARGS[@]}"
echo "STRICT_SITE_QUALITY_JOB_COMPLETE=${OUT}"
