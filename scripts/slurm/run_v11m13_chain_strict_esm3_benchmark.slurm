#!/usr/bin/env bash
# Probe strict-split PDBbind+BioLiP ESM3 global-prior checkpoints on external
# COACH420/HOLO4K per-ligand benchmarks. Each task evaluates one seed with
# both argmax and its validation-selected contact gate; ligand atoms are uncapped.

#SBATCH --job-name=si30_e3_bench
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=10:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/si30_e3_bench_%A_%a.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/si30_e3_bench_%A_%a.err

set -euo pipefail
PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
BENCH_ROOT="${BENCH_ROOT:-${PROJECT}/benchmark}"
FEATURE_STATS="${FEATURE_STATS:-${PROJECT}/data/stats_output.pt}"
ESM3_ROOT="${ESM3_ROOT:-esm3_latents_norm}"
RESULT_ROOT="${RESULT_ROOT:-${ARCHIVE}/results/v11m13_chain_si30_esm3_benchmark_${SLURM_ARRAY_JOB_ID:-manual}}"

SEEDS=(43 44 45)
RUN_DIRS=(
  "${ARCHIVE}/outputs/pocket_gvp_fusion_seed43_chain_si30_biolipmerge_esm3gp_scratch_seed43_12620756_0_v11m13e3ft_20260903_010817"
  "${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_esm3gp_scratch_seed44_12620756_1_v11m13e3ft_20260903_011256"
  "${ARCHIVE}/outputs/pocket_gvp_fusion_seed45_chain_si30_biolipmerge_esm3gp_scratch_seed45_12620756_2_v11m13e3ft_20260903_011256"
)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$TASK_ID]:?invalid task ${TASK_ID}}"
RUN_DIR="${RUN_DIRS[$TASK_ID]}"
CHECKPOINT="${RUN_DIR}/best_f1_c0_model.pt"
CONFIG="${RUN_DIR}/config.yaml"
RUN_OUT="${RESULT_ROOT}/seed${SEED}"

mkdir -p "${PROJECT}/logs" "${RUN_OUT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

for required in "${CHECKPOINT}" "${CONFIG}" "${FEATURE_STATS}" \
  "${BENCH_ROOT}/residue_ligand_tags/atom_h/coach420_per_ligand_dmin.pt" \
  "${BENCH_ROOT}/residue_ligand_tags/atom_h/holo4k_per_ligand_dmin.pt"; do
  test -e "${required}"
done
for dataset in coach420 holo4k; do
  test -d "${BENCH_ROOT}/${ESM3_ROOT}/${dataset}"
done

echo "seed=${SEED} checkpoint=${CHECKPOINT} output=${RUN_OUT}"
echo "This is a probe: training validation loss contained non-finite logits;"
echo "feature audit itself found zero non-finite cached values."

for DECISION in argmax val_gate; do
  EXTRA=()
  if [[ "${DECISION}" == "val_gate" ]]; then
    EXTRA=(--use-checkpoint-gate)
  fi
  OUT="${RUN_OUT}/${DECISION}"
  "${PYTHON}" -u scripts/infer/infer_v11_benchmark.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${CONFIG}" \
    --benchmark-root "${BENCH_ROOT}" \
    --feature-stats "${FEATURE_STATS}" \
    --datasets coach420,holo4k \
    --tag-mode per_ligand \
    --ligand-mode real \
    --output "${OUT}" \
    --batch-size 2 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --max-seq-len 1800 \
    --max-atoms -1 \
    --atom-truncation-policy alternating_ends \
    --esm3-root "${ESM3_ROOT}" \
    --amp \
    "${EXTRA[@]}"
  cat "${OUT}/coach420_summary.json"
  cat "${OUT}/holo4k_summary.json"
done
