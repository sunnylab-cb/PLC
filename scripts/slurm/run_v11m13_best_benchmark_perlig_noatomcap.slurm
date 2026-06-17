#!/bin/bash
# Run current strongest V11-family checkpoint on COACH420/HOLO4K as
# single protein-ligand pairs, without ligand atom truncation.
#
# Why this exists:
#   The aggregate-by-PDB benchmark merges all ligands for one protein before
#   inference.  This can create unnatural multi-ligand inputs, especially for
#   HOLO4K, and can interact badly with atom caps.  This script evaluates the
#   cleaner diagnostic setting: one protein + one ligand + its own distance
#   tags per sample.
#
# Main variables versus aggregate benchmark:
#   --tag-mode per_ligand
#   --max-atoms -1  (no atom truncation)

#SBATCH --job-name=v11m13_plig
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v11m13_perlig_noatom_%j.out
#SBATCH --error=logs/v11m13_perlig_noatom_%j.err

set -euo pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
CKPT="${PROJECT_DIR}/outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt"
BENCH_ROOT="${PROJECT_DIR}/benchmark"
FEATURE_STATS="${PROJECT_DIR}/data/stats_output.pt"
OUT_DIR="${PROJECT_DIR}/results/v11m13_best_benchmark_perlig_noatomcap_${SLURM_JOB_ID}"

mkdir -p "${PROJECT_DIR}/logs" "${OUT_DIR}"
cd "${PROJECT_DIR}"

echo "Job ID: ${SLURM_JOB_ID}"
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

echo "Preflight counts:"
printf "protein coach/holo: "
find "${BENCH_ROOT}/features_fixed_conservative/protein_v4/coach420" -type f -name "*_protein.pt" | wc -l
find "${BENCH_ROOT}/features_fixed_conservative/protein_v4/holo4k" -type f -name "*_protein.pt" | wc -l
printf "esm coach/holo: "
find "${BENCH_ROOT}/esm2/coach420" -maxdepth 1 -type f -name "*.pt" | wc -l
find "${BENCH_ROOT}/esm2/holo4k" -maxdepth 1 -type f -name "*.pt" | wc -l
printf "ligand coach/holo: "
find "${BENCH_ROOT}/ligand_features_v5/coach420" -maxdepth 1 -type f -name "*.pt" | wc -l
find "${BENCH_ROOT}/ligand_features_v5/holo4k" -maxdepth 1 -type f -name "*.pt" | wc -l
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
