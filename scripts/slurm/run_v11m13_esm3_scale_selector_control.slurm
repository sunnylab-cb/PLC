#!/bin/bash
# V11m13 ESM3 scratch: fixed-vs-trainable local-delta scale control.
#
# Purpose:
#   Historical no-gate runs accidentally rebuilt logits from a detached scale
#   diagnostic, so local_delta_logit_scale stayed fixed at raw=0.5
#   (effective tanh scale=0.4621). This array changes only whether that scalar
#   can learn, while also enabling the new val AUPRC/AUROC checkpoint selectors,
#   val-best contact gate, MCC, and Top-L metrics.
#
# Array mapping:
#   0 scale_fixed      exact historical optimization behavior
#   1 scale_trainable  repaired differentiable scale path
#
# Promotion rule:
#   Compare best-F1, best-AUPRC, and best-AUROC checkpoints on test
#   real/shuffle/zero. Do not promote on val F1 alone.

#SBATCH --job-name=e3_scale_sel
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/v11m13_esm3_scale_selector_%A_%a.out
#SBATCH --error=logs/v11m13_esm3_scale_selector_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/hpc2hdd/home/shuang886/PLC}"
BASE_CONFIG="${BASE_CONFIG:-${ARCHIVE_ROOT}/outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARCHIVE_ROOT}/outputs}"
ESM3_FEATURES_DIR="${ESM3_FEATURES_DIR:-esm3_latents_norm}"
BATCH_SIZE="${BATCH_SIZE:-16}"

VARIANTS=(
  "scale_fixed"
  "scale_trainable"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid task id=${TASK_ID}"
  exit 2
fi

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/tmp/slurm_configs" "${OUTPUT_DIR}"
cd "${PROJECT_DIR}" || exit 1

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Variant: ${VARIANT}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Base config: ${BASE_CONFIG}"
echo "Output directory: ${OUTPUT_DIR}"
echo "ESM3 features: data/${ESM3_FEATURES_DIR}"
echo "============================================"

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

if [ ! -f "${BASE_CONFIG}" ]; then
  echo "ERROR: base config not found: ${BASE_CONFIG}"
  exit 3
fi
if [ ! -d "data/${ESM3_FEATURES_DIR}" ]; then
  echo "ERROR: ESM3 feature dir not found: data/${ESM3_FEATURES_DIR}"
  exit 4
fi

CONFIG_FILE="tmp/slurm_configs/v11m13_esm3_scale_selector_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${ESM3_FEATURES_DIR}" "${BATCH_SIZE}" "${OUTPUT_DIR}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
esm3_features_dir = sys.argv[4]
batch_size = int(sys.argv[5])
output_dir = sys.argv[6]

with base_path.open() as handle:
    cfg = yaml.safe_load(handle)

trainable_scale = variant == "scale_trainable"

experiment = cfg.setdefault("experiment", {})
experiment["name"] = f"pocket_gvp_fusion_seed44_esm3_scratch_{variant}_selectors"
experiment["version_suffix"] = "v11m13e3scale"
experiment["seed"] = 44
experiment["output_dir"] = output_dir

data = cfg.setdefault("data", {})
data["esm3_features_dir"] = esm3_features_dir
data["max_atoms"] = int(data.get("max_atoms", 128))
data["atom_truncation_policy"] = data.get("atom_truncation_policy", "alternating_ends")

model = cfg["model"]
model["use_esm3"] = True
model["d_esm3"] = 1536
model["esm3_fusion"] = "global_prior"
model["esm3_alpha_init"] = 0.02
model["esm3_alpha_max"] = 0.20
model["esm3_gate_bias_init"] = 1.5
model["esm3_zero_init"] = True
model["esm3_dropout_prob"] = 0.0
model["local_delta_trainable_logit_scale"] = trainable_scale

# Exact seed44 scratch architecture. Do not add a second varying factor here.
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False
model["use_compile"] = False
model["use_v12_cross_attention"] = False
model["use_v12_coattention_block"] = True
model["use_local54_interaction_kv"] = True
model["local54_interaction_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
model["local54_interaction_hidden"] = 96
model["local54_interaction_kv_alpha_init"] = 0.035
model["local54_interaction_kv_alpha_max"] = 0.20
model["use_local54_residue_chem_compat"] = True
model["local54_compat_hidden"] = 128
model["local54_compat_gate_bias_init"] = 0.01
model["local54_compat_alpha_init"] = 0.02
model["local54_compat_alpha_max"] = 0.12
model["local54_compat_normalize_read"] = True
model["local54_compat_read_scale"] = 0.35
model["local54_compat_read_clip"] = 0.0
model["use_local54_edge_context_kv"] = False

training = cfg.setdefault("training", {})
training["init_from_checkpoint"] = None
training["freeze_v11_base"] = False
training["learning_rate"] = 2.0e-4
training["min_lr"] = 1.0e-6
training["warmup_epochs"] = 3
training["num_epochs"] = 80
training["batch_size"] = batch_size
training["accumulation_steps"] = 1
training["early_stopping_patience"] = 40
training["num_workers"] = min(int(training.get("num_workers", 8)), 8)
training["use_amp"] = True
# Keep latest and metric-best files, but do not create large periodic backups.
training["save_every"] = 1000

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

print(f"Wrote config: {out_path}")
print(
    f"variant={variant} trainable_scale={trainable_scale} "
    f"seed={experiment['seed']} lr={training['learning_rate']} "
    f"output_dir={experiment['output_dir']}"
)
PY

echo "Generated config:"
sed -n '1,280p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python scripts/train/train_v11m13.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1

echo "Done: ${VARIANT} at $(date)"
