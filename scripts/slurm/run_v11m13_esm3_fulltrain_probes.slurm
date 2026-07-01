#!/bin/bash
# V11m13 ESM3 full-train probes.
#
# Purpose:
#   Test whether normalized ESM3 becomes useful only when gradients are allowed
#   to update the V11m13 trunk, rather than training ESM3 adapters on top of a
#   frozen group_rescompat checkpoint.
#
# Variants:
#   0 esm3_global_prior_fullfinetune
#       Initialize from the current strongest V11m13 group_rescompat checkpoint,
#       but set freeze_v11_base=false and train all parameters with a low LR.
#   1 esm3_global_prior_scratch
#       Train the same global-prior ESM3 architecture from scratch. This is a
#       higher-risk sanity control, not the preferred production path.
#
# Data requirement:
#   data/${ESM3_FEATURES_DIR} must contain normalized ESM3 caches. Verify with:
#     python -c "import torch; x=torch.load('...pt'); print(x['embeddings'].std())"

#SBATCH --job-name=v11m13_e3ft
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/v11m13_esm3_fulltrain_%A_%a.out
#SBATCH --error=logs/v11m13_esm3_fulltrain_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BASE_CONFIG="${BASE_CONFIG:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt}"
ESM3_FEATURES_DIR="${ESM3_FEATURES_DIR:-esm3_latents_norm}"
BATCH_SIZE="${BATCH_SIZE:-16}"

VARIANTS=(
  "esm3_global_prior_fullfinetune"
  "esm3_global_prior_scratch"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid ESM3 full-train task id=${TASK_ID}"
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
echo "ESM3 features: data/${ESM3_FEATURES_DIR}"
echo "Batch size: ${BATCH_SIZE}"
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
  exit 5
fi
if [[ "${VARIANT}" == *"fullfinetune"* ]] && [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "ERROR: init checkpoint not found: ${INIT_CHECKPOINT}"
  exit 4
fi

mkdir -p tmp/slurm_configs
CONFIG_FILE="tmp/slurm_configs/v11m13_esm3_fulltrain_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${INIT_CHECKPOINT}" "${ESM3_FEATURES_DIR}" "${BATCH_SIZE}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
init_checkpoint = sys.argv[4]
esm3_features_dir = sys.argv[5]
batch_size = int(sys.argv[6])

with base_path.open() as f:
    cfg = yaml.safe_load(f)

is_scratch = variant.endswith("_scratch")

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_seed44_{variant}"
cfg["experiment"]["version_suffix"] = "v11m13e3ft"
cfg["experiment"]["seed"] = 44

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

# Preserve the current V11m13 group_rescompat architecture; only the training
# mode changes between adapter-only, full finetune, and scratch controls.
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False
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

train = cfg.setdefault("training", {})
train["init_from_checkpoint"] = None if is_scratch else init_checkpoint
train["freeze_v11_base"] = False
train["learning_rate"] = 2.0e-4 if is_scratch else 5.0e-5
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 3 if is_scratch else 2
train["num_epochs"] = 80 if is_scratch else 60
train["batch_size"] = batch_size
train["accumulation_steps"] = 1
train["early_stopping_patience"] = 40
train["num_workers"] = min(int(train.get("num_workers", 8)), 8)
train["use_amp"] = True

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote ESM3 full-train config: {out_path}")
print(
    f"variant={variant} fusion={model['esm3_fusion']} "
    f"scratch={is_scratch} init={train['init_from_checkpoint']} "
    f"freeze={train['freeze_v11_base']} lr={train['learning_rate']} "
    f"batch={train['batch_size']}"
)
PY

echo "Generated config:"
sed -n '1,260p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python scripts/train/train_v11m13.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1

echo "Done: ${VARIANT} at $(date)"
