#!/usr/bin/env bash
# Strict VAL only: frozen base44 and ESM3-global-prior44, sequential on ONE GPU.
# Reproduce training features/cropping/BF16 and the checkpoint's existing gate.
# Export full CA/label axes; compare original components, spatial NMS spheres,
# and geometry-only FPS coverage. No COACH/HOLO/test labels, no model training.
# Candidate oracle F1 is an upper-bound diagnostic, NOT inference performance.
#SBATCH --job-name=strict_val_prop
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
EXPERIMENT="${EXPERIMENT:-${ARCHIVE}/experiments/strict_assignment_20260919}"
CODE="${CODE:-${EXPERIMENT}/code}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
OUT="${EXPERIMENT}/val_${SLURM_JOB_ID:-manual}"
LABEL_ARGS=()
if [[ -n "${LABEL_ALIGNMENT:-}" ]]; then LABEL_ARGS=(--label-alignment "${LABEL_ALIGNMENT}"); fi
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${CODE}"
RUNS=(
  "${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_scratch_seed44_12222454_1_v11m13_20260826_055836"
  "${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_esm3gp_scratch_seed44_12620756_1_v11m13e3ft_20260903_011256"
)
ROOTS=(pdbbind_biolip_chain_si30_final pdbbind_biolip_chain_si30_esm3)
NAMES=(base44 esm3gp44)
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
for i in 0 1; do
  DATA="${ARCHIVE}/datasets/${ROOTS[$i]}"
  TARGET="${OUT}/${NAMES[$i]}"
  "${PYTHON}" -u scripts/diagnostics/export_strict_val_predictions.py \
    --checkpoint "${RUNS[$i]}/best_f1_c0_model.pt" --data-root "${DATA}" --output "${TARGET}" "${LABEL_ARGS[@]}"
  "${PYTHON}" -u scripts/diagnostics/strict_val_candidates.py \
    --predictions "${TARGET}/predictions" --splits "${DATA}/splits.txt" --output "${TARGET}/candidates"
done
echo "STRICT_VAL_JOB_COMPLETE=${OUT}"
