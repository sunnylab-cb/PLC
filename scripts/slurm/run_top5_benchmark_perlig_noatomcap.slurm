#!/bin/bash
# Run top V11-family checkpoints on COACH420/HOLO4K as single protein-ligand
# pairs, without ligand atom truncation.
#
# Motivation:
#   Aggregate-by-PDB benchmark merges all ligands for one protein into one input.
#   For HOLO4K this can create unnatural multi-ligand samples and can confound
#   interpretation.  This script uses per-ligand tags so each sample is exactly
#   one protein + one ligand, and uses --max-atoms -1 to disable test-time atom
#   truncation.
#
# Usage:
#   BENCH_TASK_ID=0 sbatch scripts/slurm/run_top5_benchmark_perlig_noatomcap.slurm
#
# Task mapping:
#   0 seed44_u_mem4b2_lr2e4
#   1 seed44_u_mem1b2_lr2e4
#   2 seed44_all3_lr5e4
#   3 seed44_group_rescompat_lr2e4
#   4 seed44_chem_adapter_lr2e4

#SBATCH --job-name=top5_plig
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/top5_perlig_noatom_%j.out
#SBATCH --error=logs/top5_perlig_noatom_%j.err

set -euo pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BENCH_ROOT="${PROJECT_DIR}/benchmark"
FEATURE_STATS="${PROJECT_DIR}/data/stats_output.pt"

RUN_NAMES=(
  "seed44_u_mem4b2_lr2e4"
  "seed44_u_mem1b2_lr2e4"
  "seed44_all3_lr5e4"
  "seed44_group_rescompat_lr2e4"
  "seed44_chem_adapter_lr2e4"
)

CHECKPOINTS=(
  "outputs/pocket_gvp_fusion_seed44_u_mem4b2_lr2e4_v11m12_20260611_111611/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed44_u_mem1b2_lr2e4_v11m12_20260611_052606/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed44_all3_lr5e4_v11m13_20260611_205603/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed44_chem_adapter_lr2e4_v11m13_20260611_205604/best_f1_c0_model.pt"
)

TASK_ID="${BENCH_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
RUN_NAME="${RUN_NAMES[$TASK_ID]}"
CKPT="${PROJECT_DIR}/${CHECKPOINTS[$TASK_ID]}"
OUT_DIR="${PROJECT_DIR}/results/top5_perlig_noatom_${RUN_NAME}_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Task ID: ${TASK_ID}"
echo "Run name: ${RUN_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "Checkpoint: ${CKPT}"
echo "Benchmark root: ${BENCH_ROOT}"
echo "Feature stats: ${FEATURE_STATS}"
echo "Output: ${OUT_DIR}"

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: checkpoint not found: ${CKPT}" >&2
  exit 1
fi
if [[ ! -f "${FEATURE_STATS}" ]]; then
  echo "ERROR: feature stats not found: ${FEATURE_STATS}" >&2
  exit 1
fi
ls -lh "${BENCH_ROOT}/residue_ligand_tags/atom_h/"*_per_ligand_dmin.pt

conda run -n af3 python -u scripts/infer/infer_v11_benchmark.py \
  --checkpoint "${CKPT}" \
  --benchmark-root "${BENCH_ROOT}" \
  --feature-stats "${FEATURE_STATS}" \
  --datasets coach420,holo4k \
  --tag-mode per_ligand \
  --ligand-mode real \
  --output "${OUT_DIR}" \
  --batch-size 2 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --max-seq-len 1800 \
  --max-atoms -1 \
  --amp

echo "End: $(date)"
echo "Summaries:"
cat "${OUT_DIR}/coach420_summary.json"
cat "${OUT_DIR}/holo4k_summary.json"
