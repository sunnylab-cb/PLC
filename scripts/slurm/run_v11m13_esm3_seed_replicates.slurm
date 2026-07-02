#!/bin/bash
# ESM3 global-prior seed replicates for the two promising modes.
#
# Variants:
#   fullfinetune: initialize from the best available group_rescompat checkpoint
#                 for the seed when present; otherwise seed44 baseline.
#   scratch:      train the same architecture from scratch.
#
# Array mapping:
#   0 seed43 fullfinetune
#   1 seed45 fullfinetune
#   2 seed46 fullfinetune
#   3 seed43 scratch
#   4 seed45 scratch
#   5 seed46 scratch

#SBATCH --job-name=e3_seed
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/v11m13_esm3_seed_%A_%a.out
#SBATCH --error=logs/v11m13_esm3_seed_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BASE_CONFIG="${BASE_CONFIG:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
DEFAULT_INIT="${DEFAULT_INIT:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt}"
ESM3_FEATURES_DIR="${ESM3_FEATURES_DIR:-esm3_latents_norm}"
BATCH_SIZE="${BATCH_SIZE:-16}"

SEEDS=(43 45 46 43 45 46)
MODES=("fullfinetune" "fullfinetune" "fullfinetune" "scratch" "scratch" "scratch")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$TASK_ID]}"
MODE="${MODES[$TASK_ID]}"
if [ -z "${SEED}" ] || [ -z "${MODE}" ]; then
  echo "ERROR: invalid task id=${TASK_ID}"
  exit 2
fi
VARIANT="esm3_global_prior_${MODE}_seed${SEED}"

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Variant: ${VARIANT}"
echo "Seed: ${SEED}"
echo "Mode: ${MODE}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Base config: ${BASE_CONFIG}"
echo "Default init: ${DEFAULT_INIT}"
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

INIT_CHECKPOINT="${DEFAULT_INIT}"
if [ "${MODE}" = "fullfinetune" ]; then
  if [ "${SEED}" = "45" ] && [ -f "outputs/pocket_gvp_fusion_seed45_group_rescompat_lr2e4_v11m13_20260614_022638/best_f1_c0_model.pt" ]; then
    INIT_CHECKPOINT="outputs/pocket_gvp_fusion_seed45_group_rescompat_lr2e4_v11m13_20260614_022638/best_f1_c0_model.pt"
  fi
  if [ ! -f "${INIT_CHECKPOINT}" ]; then
    echo "ERROR: init checkpoint not found: ${INIT_CHECKPOINT}"
    exit 5
  fi
else
  INIT_CHECKPOINT=""
fi
echo "Resolved init checkpoint: ${INIT_CHECKPOINT:-<scratch>}"

mkdir -p tmp/slurm_configs
CONFIG_FILE="tmp/slurm_configs/v11m13_esm3_seed_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${SEED}" "${MODE}" "${INIT_CHECKPOINT}" "${ESM3_FEATURES_DIR}" "${BATCH_SIZE}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
seed = int(sys.argv[4])
mode = sys.argv[5]
init_checkpoint = sys.argv[6]
esm3_features_dir = sys.argv[7]
batch_size = int(sys.argv[8])
is_scratch = mode == "scratch"

with base_path.open() as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_{variant}"
cfg["experiment"]["version_suffix"] = "v11m13e3ft"
cfg["experiment"]["seed"] = seed

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

# Preserve the current group_rescompat recipe.
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

print(f"Wrote ESM3 seed config: {out_path}")
print(
    f"variant={variant} seed={seed} mode={mode} init={train['init_from_checkpoint']} "
    f"lr={train['learning_rate']} batch={train['batch_size']}"
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
