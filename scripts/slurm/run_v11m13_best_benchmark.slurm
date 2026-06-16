#!/bin/bash
# Run the current strongest V11-family checkpoint on COACH420 and HOLO4K.
#
# Model:
#   seed44_group_rescompat_lr2e4_v11m13 best_f1_c0_model.pt
#
# Benchmark data:
#   protein V4: benchmark/features_fixed_conservative/protein_v4
#   ESM2:       benchmark/esm2
#   ligand V5:  benchmark/ligand_features_v5
#   tags:       benchmark/residue_ligand_tags/atom_h/*_aggregate_pdb_dmin.pt
#
# Main reported mode:
#   aggregate-by-PDB tags, real ligand, Contact<5A / Near 5-8A.

#SBATCH --job-name=v11m13_bench
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/v11m13_best_benchmark_%j.out
#SBATCH --error=logs/v11m13_best_benchmark_%j.err

set -euo pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
CKPT="${PROJECT_DIR}/outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt"
BENCH_ROOT="${PROJECT_DIR}/benchmark"
FEATURE_STATS="${PROJECT_DIR}/data/stats_output.pt"
OUT_DIR="${PROJECT_DIR}/results/v11m13_best_benchmark_norm_${SLURM_JOB_ID}"

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
ls -lh "${BENCH_ROOT}/residue_ligand_tags/atom_h/"*_aggregate_pdb_dmin.pt

conda run -n af3 python -u scripts/infer/infer_v11_benchmark.py \
  --checkpoint "${CKPT}" \
  --benchmark-root "${BENCH_ROOT}" \
  --feature-stats "${FEATURE_STATS}" \
  --datasets coach420,holo4k \
  --tag-mode aggregate \
  --ligand-mode real \
  --output "${OUT_DIR}" \
  --batch-size 4 \
  --num-workers "${SLURM_CPUS_PER_TASK}" \
  --max-seq-len 1800 \
  --max-atoms 128 \
  --atom-truncation-policy alternating_ends \
  --amp

echo "End: $(date)"
echo "Summaries:"
cat "${OUT_DIR}/coach420_summary.json"
cat "${OUT_DIR}/holo4k_summary.json"
