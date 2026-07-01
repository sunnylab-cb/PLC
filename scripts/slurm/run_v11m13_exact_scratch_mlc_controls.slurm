#!/bin/bash
# V11m13 exact-scratch controls for ESM2-MLC.
#
# Purpose:
#   Separate two effects that were confounded in the first MLC scratch run:
#     1. V11m13's strong baseline used a warm-started frozen-base refinement
#        from v11m8x coattn_g60.
#     2. The previous MLC scratch script started from configs/pocket_gvp_v11m6n_hpc.yaml,
#        which missed several V11m13 baseline switches such as local_ligand_film,
#        pocket_virtual_tokens, ligand_source_dropout, and center_aux.
#
# This script starts from the exact V11m13 group_rescompat config, preserves all
# model/data/loss switches, then clears init_from_checkpoint and disables
# freeze_v11_base. Variants:
#   0 v11m13_exact_scratch_no_mlc
#   1 v11m13_exact_scratch_mlc_wsum
#   2 v11m13_exact_scratch_mlc_concat

#SBATCH --job-name=v11m13_xs
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/v11m13_exact_scratch_%A_%a.out
#SBATCH --error=logs/v11m13_exact_scratch_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BASE_CONFIG="${BASE_CONFIG:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
ESM_FEATURES_DIR="${ESM_FEATURES_DIR:-esm2_multilayer_latents}"
ESM_MLC_LAYERS="${ESM_MLC_LAYERS:-2,6,12,18,24,30,33}"
BATCH_SIZE="${BATCH_SIZE:-20}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"
NUM_EPOCHS="${NUM_EPOCHS:-100}"
LEARNING_RATE="${LEARNING_RATE:-5.0e-5}"

VARIANTS=(
  "v11m13_exact_scratch_no_mlc"
  "v11m13_exact_scratch_mlc_wsum"
  "v11m13_exact_scratch_mlc_concat"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid exact-scratch task id=${TASK_ID}"
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
if [[ "${VARIANT}" == *"mlc"* ]] && [ ! -d "data/${ESM_FEATURES_DIR}" ]; then
  echo "ERROR: ESM multi-layer feature dir not found: data/${ESM_FEATURES_DIR}"
  exit 4
fi

mkdir -p tmp/slurm_configs
CONFIG_FILE="tmp/slurm_configs/v11m13_exact_scratch_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${ESM_FEATURES_DIR}" "${ESM_MLC_LAYERS}" "${BATCH_SIZE}" "${ACCUMULATION_STEPS}" "${NUM_EPOCHS}" "${LEARNING_RATE}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
esm_features_dir = sys.argv[4]
layers = [int(x) for x in sys.argv[5].split(",") if x.strip()]
batch_size = int(sys.argv[6])
accumulation_steps = int(sys.argv[7])
num_epochs = int(sys.argv[8])
learning_rate = float(sys.argv[9])

with base_path.open() as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_seed44_{variant}"
cfg["experiment"]["version_suffix"] = "v11m13xs"
cfg["experiment"]["seed"] = 44

data = cfg.setdefault("data", {})
data["max_atoms"] = int(data.get("max_atoms", 128))
data["atom_truncation_policy"] = data.get("atom_truncation_policy", "alternating_ends")

model = cfg.setdefault("model", {})
use_mlc = variant.endswith("_mlc_wsum") or variant.endswith("_mlc_concat")
model["use_esm_mlc"] = use_mlc
if use_mlc:
    data["esm_features_dir"] = esm_features_dir
    model["esm_mlc_num_layers"] = len(layers)
    model["esm_mlc_layers"] = layers
    model["esm_mlc_init"] = "uniform"
    model["esm_mlc_dropout"] = 0.05
    model["esm_mlc_fusion"] = "concat_proj" if variant.endswith("_concat") else "weighted_sum"
    model["esm_mlc_concat_zero_init"] = True
else:
    for key in (
        "esm_mlc_num_layers",
        "esm_mlc_layers",
        "esm_mlc_init",
        "esm_mlc_dropout",
        "esm_mlc_fusion",
        "esm_mlc_concat_zero_init",
        "esm_mlc_last_logit_bias",
    ):
        model.pop(key, None)

# Keep exact V11m13 architecture switches from BASE_CONFIG. Only clear the
# warm-start/frozen-base recipe so this is a true scratch control.
train = cfg.setdefault("training", {})
train["init_from_checkpoint"] = None
train["freeze_v11_base"] = False
train.pop("delta_trainable_tokens", None)
train["learning_rate"] = learning_rate
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 3
train["num_epochs"] = num_epochs
train["batch_size"] = batch_size
train["accumulation_steps"] = accumulation_steps
train["early_stopping_patience"] = num_epochs
train["num_workers"] = min(int(train.get("num_workers", 8)), 8)
train["use_amp"] = True

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote exact-scratch config: {out_path}")
print(
    f"variant={variant} use_mlc={use_mlc} fusion={model.get('esm_mlc_fusion')} "
    f"layers={layers if use_mlc else None} batch={batch_size} lr={learning_rate}"
)
print(
    "kept_v11m13_switches="
    f"local_ligand_film={model.get('use_local_ligand_film')} "
    f"pocket_virtual={model.get('use_pocket_virtual_tokens')} "
    f"ligdrop={model.get('ligand_source_dropout_p')} "
    f"center_aux={model.get('use_center_aux_head')}"
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
