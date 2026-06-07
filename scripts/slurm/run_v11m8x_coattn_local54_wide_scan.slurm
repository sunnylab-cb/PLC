#!/bin/bash
# Serial local54 dependency scan for the strongest V11m8x co-attn probe.
#
# Purpose:
#   Broadly identify which local54 atom-chemistry dimensions/groups/combinations
#   carry ligand-aware residue signal. This is eval-only: no training and no
#   checkpoint mutation.
#
# Default checkpoint:
#   outputs/pocket_gvp_fusion_seed44_coattn_g60_v11m8x_*/best_f1_c0_model.pt
#
# Job layout:
#   array task 0: baseline + all 54 single local54 dims, zero/shuffle
#   array task 1: whole groups, zero/shuffle/mean/noise
#   array task 2: promising chemistry combinations, zero/shuffle/mean/noise
#
# Rationale:
#   The cluster QOS limits array task count/concurrency. Each Slurm task loops
#   serially over many eval modes on one GPU, trading wall time for low submit
#   count.

#SBATCH --job-name=v11m8x_l54ser
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/v11m8x_l54ser_%A_%a.out
#SBATCH --error=logs/v11m8x_l54ser_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m8x_coattn_local54_wide_${SLURM_ARRAY_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
SPLIT="${SPLIT:-test}"
VARIANT="${VARIANT_NAME:-seed44_coattn_g60}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

BASE_MODES=(
  "real"
  "shuffle"
  "zero"
  "zero_ligand_local54_only"
  "zero_ligand_global55_only"
  "zero_ligand_edge_attr_only"
  "shuffle_ligand_topology_only"
)

SINGLE_OPS=("zero" "shuffle")

GROUP_NAMES=(
  "base33"
  "electronic3"
  "sasa_hbond4"
  "flex_rot5"
  "metal4"
  "stereo5"
)
GROUP_RANGES=(
  "0:33"
  "33:36"
  "38:42"
  "36:38,44:47"
  "47:51"
  "42:44,51:54"
)
GROUP_OPS=("zero" "shuffle" "mean" "noise")

COMBO_NAMES=(
  "electronic3_sasa_hbond4"
  "electronic3_flex_rot5"
  "sasa_hbond4_flex_rot5"
  "base33_electronic3"
  "base33_sasa_hbond4"
  "base33_flex_rot5"
  "electronic3_sasa_hbond4_flex_rot5"
  "base33_electronic3_sasa_hbond4_flex_rot5"
  "polar_surface_electronic"
  "nonlocal_tail_metal_stereo"
)
COMBO_RANGES=(
  "33:36,38:42"
  "33:36,36:38,44:47"
  "38:42,36:38,44:47"
  "0:33,33:36"
  "0:33,38:42"
  "0:33,36:38,44:47"
  "33:36,38:42,36:38,44:47"
  "0:33,33:36,38:42,36:38,44:47"
  "33:36,38:42,47:51"
  "42:44,47:51,51:54"
)
COMBO_OPS=("zero" "shuffle" "mean" "noise")

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Variant: ${VARIANT}"
echo "Split: ${SPLIT}"
echo "Output root: ${OUT_ROOT}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "============================================"

if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -gt 2 ]; then
  echo "ERROR: invalid task id ${TASK_ID}; expected 0, 1, or 2"
  exit 2
fi

mkdir -p logs

if command -v module &> /dev/null; then
  module purge || true
fi

if command -v conda &> /dev/null; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "ERROR: conda command not found in PATH"
  exit 1
fi
conda activate af3

cd "${PROJECT_DIR}" || exit 1

RUN_DIR="$(ls -dt "outputs/pocket_gvp_fusion_${VARIANT}_v11m8x_"* 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: no output directory found for variant: ${VARIANT}"
  exit 3
fi

CKPT="${RUN_DIR}/best_f1_c0_model.pt"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT}"
  exit 4
fi

echo "Run dir: ${RUN_DIR}"
echo "Checkpoint: ${CKPT}"
echo "Python: $(which python)"
python --version

run_eval() {
  local mode_label="$1"
  local ligand_mode="$2"
  local feature_name="${3:-}"
  local feature_ranges="${4:-}"
  local feature_op="${5:-}"

  local out_dir="${OUT_ROOT}/${VARIANT}_${SPLIT}_${mode_label}"
  mkdir -p "${out_dir}"

  if [ -f "${out_dir}/ligand_dependency_summary.json" ]; then
    echo "SKIP existing result: ${out_dir}/ligand_dependency_summary.json"
    return 0
  fi

  local cmd=(
    python scripts/infer/infer_v11_internal_ligdep.py
    --checkpoint "${CKPT}"
    --split "${SPLIT}"
    --ligand_mode "${ligand_mode}"
    --output "${out_dir}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --max_samples "${MAX_SAMPLES}"
    --num_save_rows 1
    --amp
  )

  if [ "${NO_PIN_MEMORY:-0}" = "1" ]; then
    cmd+=(--no_pin_memory)
  fi

  if [ "${ligand_mode}" = "custom_local54" ]; then
    cmd+=(
      --custom_local54_name "${feature_name}"
      --custom_local54_ranges "${feature_ranges}"
      --custom_local54_op "${feature_op}"
    )
  fi

  echo "--------------------------------------------"
  echo "Eval start: $(date)"
  echo "Mode label: ${mode_label}"
  echo "Ligand mode: ${ligand_mode}"
  echo "Feature name: ${feature_name}"
  echo "Feature ranges: ${feature_ranges}"
  echo "Feature op: ${feature_op}"
  echo "Output dir: ${out_dir}"
  echo "Command: ${cmd[*]}"
  echo "--------------------------------------------"

  "${cmd[@]}"
  local exit_code=$?
  echo "Eval end: $(date), exit=${exit_code}, mode=${mode_label}"
  if [ "${exit_code}" -ne 0 ]; then
    exit "${exit_code}"
  fi
}

if [ "${TASK_ID}" -eq 0 ]; then
  echo "Task 0: baseline + single local54 dim scan"
  for mode in "${BASE_MODES[@]}"; do
    run_eval "${mode}" "${mode}"
  done
  for dim in $(seq 0 53); do
    for op in "${SINGLE_OPS[@]}"; do
      name=$(printf "dim%02d" "${dim}")
      run_eval "custom_local54_${name}_${op}" "custom_local54" "${name}" "${dim}:$((dim + 1))" "${op}"
    done
  done
elif [ "${TASK_ID}" -eq 1 ]; then
  echo "Task 1: whole local54 group scan"
  for i in "${!GROUP_NAMES[@]}"; do
    for op in "${GROUP_OPS[@]}"; do
      name="${GROUP_NAMES[$i]}"
      ranges="${GROUP_RANGES[$i]}"
      run_eval "custom_local54_${name}_${op}" "custom_local54" "${name}" "${ranges}" "${op}"
    done
  done
elif [ "${TASK_ID}" -eq 2 ]; then
  echo "Task 2: promising local54 combination scan"
  for i in "${!COMBO_NAMES[@]}"; do
    for op in "${COMBO_OPS[@]}"; do
      name="${COMBO_NAMES[$i]}"
      ranges="${COMBO_RANGES[$i]}"
      run_eval "custom_local54_${name}_${op}" "custom_local54" "${name}" "${ranges}" "${op}"
    done
  done
fi

echo "============================================"
echo "End time: $(date)"
echo "All evals completed for task ${TASK_ID}"
echo "============================================"
