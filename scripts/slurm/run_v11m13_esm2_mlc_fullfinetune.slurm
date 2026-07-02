#!/bin/bash
# V11m13 ESM2-MLC full-parameter finetune probes.
#
# Purpose:
#   The first MLC probes showed:
#     - adapter-only warm-start: small/no val gain around the V11m13 basin
#     - scratch wsum: late small positive signal, but slow and precision-heavy
#     - ESM3 global_prior fullfinetune: clear val gain, suggesting feature
#       fusion may need trunk adaptation rather than frozen adapters.
#
# This matrix tests the analogous MLC setting:
#   Initialize from the strongest V11m13 group_rescompat checkpoint, enable ESM2
#   multi-layer cache fusion, do NOT freeze the V11 base, and finetune all
#   parameters with a low LR.
#
# Variants:
#   0 esmmlc_wsum_fullfinetune
#   1 esmmlc_concat_fullfinetune

#SBATCH --job-name=mlc_ft
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/mlc_fullfinetune_%A_%a.out
#SBATCH --error=logs/mlc_fullfinetune_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BASE_CONFIG="${BASE_CONFIG:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt}"
ESM_FEATURES_DIR="${ESM_FEATURES_DIR:-esm2_multilayer_latents}"
ESM_MLC_LAYERS="${ESM_MLC_LAYERS:-2,6,12,18,24,30,33}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-60}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-5}"

VARIANTS=(
  "esmmlc_wsum_fullfinetune"
  "esmmlc_concat_fullfinetune"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid MLC fullfinetune task id=${TASK_ID}"
  exit 2
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Variant: ${VARIANT}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Base config: ${BASE_CONFIG}"
echo "Init checkpoint: ${INIT_CHECKPOINT}"
echo "ESM features: data/${ESM_FEATURES_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "LR: ${LEARNING_RATE}"
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
if [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "ERROR: init checkpoint not found: ${INIT_CHECKPOINT}"
  exit 4
fi
if [ ! -d "data/${ESM_FEATURES_DIR}" ]; then
  echo "ERROR: ESM multi-layer feature dir not found: data/${ESM_FEATURES_DIR}"
  exit 5
fi

mkdir -p tmp/slurm_configs
CONFIG_FILE="tmp/slurm_configs/v11m13_esmmlc_fullfinetune_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${INIT_CHECKPOINT}" "${ESM_FEATURES_DIR}" "${ESM_MLC_LAYERS}" "${BATCH_SIZE}" "${ACCUMULATION_STEPS}" "${NUM_EPOCHS}" "${LEARNING_RATE}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
init_checkpoint = sys.argv[4]
esm_features_dir = sys.argv[5]
layers = [int(x) for x in sys.argv[6].split(",") if x.strip()]
batch_size = int(sys.argv[7])
accumulation_steps = int(sys.argv[8])
num_epochs = int(sys.argv[9])
learning_rate = float(sys.argv[10])

with base_path.open() as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_seed44_{variant}"
cfg["experiment"]["version_suffix"] = "v11m13mlcft"
cfg["experiment"]["seed"] = 44

data = cfg.setdefault("data", {})
data["esm_features_dir"] = esm_features_dir
data["max_atoms"] = int(data.get("max_atoms", 128))
data["atom_truncation_policy"] = data.get("atom_truncation_policy", "alternating_ends")

model = cfg.setdefault("model", {})
model["use_esm_mlc"] = True
model["esm_mlc_num_layers"] = len(layers)
model["esm_mlc_layers"] = layers
model["esm_mlc_init"] = "last"
model["esm_mlc_last_logit_bias"] = 4.0
model["esm_mlc_dropout"] = 0.05
model["esm_mlc_fusion"] = "concat_proj" if "concat" in variant else "weighted_sum"
model["esm_mlc_concat_zero_init"] = True

# Preserve the exact V11m13 architecture from BASE_CONFIG. Only change the MLC
# feature path and training mode.
train = cfg.setdefault("training", {})
train["init_from_checkpoint"] = init_checkpoint
train["freeze_v11_base"] = False
train.pop("delta_trainable_tokens", None)
train["learning_rate"] = learning_rate
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 2
train["num_epochs"] = num_epochs
train["batch_size"] = batch_size
train["accumulation_steps"] = accumulation_steps
train["early_stopping_patience"] = 40
train["num_workers"] = min(int(train.get("num_workers", 8)), 8)
train["use_amp"] = True

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote MLC fullfinetune config: {out_path}")
print(
    f"variant={variant} fusion={model['esm_mlc_fusion']} layers={layers} "
    f"init={init_checkpoint} freeze={train['freeze_v11_base']} "
    f"batch={batch_size} lr={learning_rate}"
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
  --accumulation-steps "${ACCUMULATION_STEPS}"

echo "Done: ${VARIANT} at $(date)"
