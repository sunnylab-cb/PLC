#!/bin/bash
# Benchmark the five max_atoms=512 retrained top candidates on COACH420/HOLO4K.
#
# This is a diagnostic benchmark for the "relaxed ligand atom cap during
# training" reruns.  It uses the same external benchmark protocol as the recent
# per-ligand/no-atom-cap evaluation:
#   - one protein + one ligand sample per row
#   - no test-time atom truncation (--max-atoms -1)
#   - updated benchmark summary with sample_mean, protein_mean_of_ligand_mean,
#     protein_mean_of_ligand_max, PDR, LAA, BP, key alignment, and contact-count
#     distributions.
#
# Usage:
#   BENCH_TASK_ID=0 sbatch scripts/slurm/run_top5_maxatoms_benchmark_perlig_noatomcap.slurm
#
# Task mapping:
#   0 seed44_u_mem4b2_lr2e4_maxatoms512
#   1 seed44_u_mem1b2_lr2e4_maxatoms512
#   2 seed44_group_rescompat_lr2e4_maxatoms512
#   3 seed44_all3_lr5e4_maxatoms512
#   4 seed44_chem_adapter_lr2e4_maxatoms512

#SBATCH --job-name=mx512_plig
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --output=logs/top5_maxatoms_perlig_noatom_%j.out
#SBATCH --error=logs/top5_maxatoms_perlig_noatom_%j.err

set -euo pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BENCH_ROOT="${PROJECT_DIR}/benchmark"
FEATURE_STATS="${PROJECT_DIR}/data/stats_output.pt"

RUN_NAMES=(
  "seed44_u_mem4b2_lr2e4_maxatoms512"
  "seed44_u_mem1b2_lr2e4_maxatoms512"
  "seed44_group_rescompat_lr2e4_maxatoms512"
  "seed44_all3_lr5e4_maxatoms512"
  "seed44_chem_adapter_lr2e4_maxatoms512"
)

CHECKPOINTS=(
  "outputs/pocket_gvp_fusion_seed44_u_mem4b2_lr2e4_maxatoms512_seed44_u_mem4b2_lr2e4_maxatoms_v11m12_20260617_215656/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed44_u_mem1b2_lr2e4_maxatoms512_seed44_u_mem1b2_lr2e4_maxatoms_v11m12_20260617_225223/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_maxatoms512_seed44_group_rescompat_lr2e4_maxatoms_v11m13_20260617_225259/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed44_all3_lr5e4_maxatoms512_seed44_all3_lr5e4_maxatoms_v11m13_20260617_225257/best_f1_c0_model.pt"
  "outputs/pocket_gvp_fusion_seed44_chem_adapter_lr2e4_maxatoms512_seed44_chem_adapter_lr2e4_maxatoms_v11m13_20260617_225259/best_f1_c0_model.pt"
)

TASK_ID="${BENCH_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
RUN_NAME="${RUN_NAMES[$TASK_ID]}"
CKPT="${PROJECT_DIR}/${CHECKPOINTS[$TASK_ID]}"
OUT_DIR="${PROJECT_DIR}/results/top5_maxatoms_perlig_noatom_${RUN_NAME}_${SLURM_JOB_ID}"

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
