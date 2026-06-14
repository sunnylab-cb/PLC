#!/bin/bash
# V11m14 base33-conditioned chemistry token probes.
#
# Purpose:
#   Test a small architecture follow-up to group_rescompat:
#   use base33 as anchor, but form one conditioned token from
#   base33 + electronic3 + sasa_hbond4 + flex_rot5.  This avoids treating
#   base33 as a direct group while still modeling local54 combination effects.
#
# Variants:
#   0 basecond_prev_lr2e4:
#       warm-start from seed44 group_rescompat; train only basecond token.
#   1 basecond_soft_prev_lr2e4:
#       same, but smaller bounded alpha/gate.
#   2 basecond_coattn_lr2e4:
#       warm-start from seed44 coattn; train group_rescompat + basecond modules.

#SBATCH --job-name=v11m14_bcond
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/v11m14_bcond_%A_%a.out
#SBATCH --error=logs/v11m14_bcond_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
PREV_CHECKPOINT="${PREV_CHECKPOINT:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt}"
COATTN_CHECKPOINT="${COATTN_CHECKPOINT:-outputs/pocket_gvp_fusion_seed44_coattn_g60_v11m8x_20260606_200217/best_f1_c0_model.pt}"

VARIANTS=(
  "seed44_basecond_prev_lr2e4"
  "seed44_basecond_soft_prev_lr2e4"
  "seed44_basecond_coattn_lr2e4"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m14 basecond task id: ${TASK_ID}"
  exit 2
fi

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Variant: ${VARIANT}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Base config: ${BASE_CONFIG}"
echo "Prev checkpoint: ${PREV_CHECKPOINT}"
echo "Coattn checkpoint: ${COATTN_CHECKPOINT}"
echo "============================================"

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

if [ ! -f "${BASE_CONFIG}" ]; then
  echo "ERROR: base config not found: ${BASE_CONFIG}"
  exit 3
fi
if [ ! -f "${PREV_CHECKPOINT}" ]; then
  echo "ERROR: previous group_rescompat checkpoint not found: ${PREV_CHECKPOINT}"
  exit 4
fi
if [ ! -f "${COATTN_CHECKPOINT}" ]; then
  echo "ERROR: V11m8x coattn checkpoint not found: ${COATTN_CHECKPOINT}"
  exit 5
fi

echo "Python: $(which python)"
python --version

mkdir -p tmp/slurm_configs
CONFIG_FILE="tmp/slurm_configs/v11m14_basecond_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${PREV_CHECKPOINT}" "${COATTN_CHECKPOINT}" <<'PY'
import re
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
prev_checkpoint = sys.argv[4]
coattn_checkpoint = sys.argv[5]

with base_path.open() as f:
    cfg = yaml.safe_load(f)

seed_match = re.search(r"seed(\d+)_", variant)
if not seed_match:
    raise SystemExit(f"Could not parse seed from variant: {variant}")

seed = int(seed_match.group(1))
init_checkpoint = coattn_checkpoint if "_coattn_" in variant else prev_checkpoint

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_{variant}"
cfg["experiment"]["version_suffix"] = "v11m14"
cfg["experiment"]["seed"] = seed

model = cfg["model"]

# Preserve original group_rescompat structure: base33 anchor with
# electronic/sasa/flex condition groups.
model["local54_interaction_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
model["local54_interaction_hidden"] = 96
model["use_local54_interaction_kv"] = True
model["use_local54_residue_chem_compat"] = True
model["use_local54_edge_context_kv"] = False
model["use_local54_chem_adapter"] = False
model["use_local54_chem_reweight"] = False

model["use_local54_basecond_token"] = True
model["local54_basecond_hidden"] = 96
if "soft" in variant:
    model["local54_basecond_kv_alpha_init"] = 0.010
    model["local54_basecond_kv_alpha_max"] = 0.08
    model["local54_basecond_compat_alpha_init"] = 0.008
    model["local54_basecond_compat_alpha_max"] = 0.06
    model["local54_basecond_compat_gate_bias_init"] = 0.002
    model["local54_basecond_compat_read_scale"] = 0.20
else:
    model["local54_basecond_kv_alpha_init"] = 0.020
    model["local54_basecond_kv_alpha_max"] = 0.12
    model["local54_basecond_compat_alpha_init"] = 0.015
    model["local54_basecond_compat_alpha_max"] = 0.10
    model["local54_basecond_compat_gate_bias_init"] = 0.005
    model["local54_basecond_compat_read_scale"] = 0.30
model["local54_basecond_compat_normalize_read"] = True

model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False
model["use_v12_cross_attention"] = False
model["use_v12_coattention_block"] = True

model["local54_interaction_kv_alpha_init"] = 0.035
model["local54_interaction_kv_alpha_max"] = 0.20
model["local54_compat_gate_bias_init"] = 0.010
model["local54_compat_alpha_init"] = 0.020
model["local54_compat_alpha_max"] = 0.12
model["local54_compat_normalize_read"] = True
model["local54_compat_read_scale"] = 0.35
model["local54_compat_read_clip"] = 0.0
model["use_ligand_context_prior"] = False
model["ligand_context_prior_sources"] = []

cfg.setdefault("training", {})
train = cfg["training"]
train["init_from_checkpoint"] = init_checkpoint
train["freeze_v11_base"] = True
if "_coattn_" in variant:
    train["delta_trainable_tokens"] = [
        "local54_anchor_proj.",
        "local54_group_proj.",
        "local54_interaction_",
        "local54_compat_",
        "local54_basecond_",
    ]
else:
    train["delta_trainable_tokens"] = ["local54_basecond_"]
train["learning_rate"] = 2.0e-4
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 2
train["num_epochs"] = 60
train["batch_size"] = 24
train["accumulation_steps"] = 1
train["early_stopping_patience"] = 60

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Wrote V11m14 basecond config:", out_path)
print(
    f"variant={variant} seed={seed} init={init_checkpoint} "
    f"train_tokens={train['delta_trainable_tokens']}"
)
PY

echo "Generated config:"
sed -n '1,260p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Command: python scripts/train/train_v11m14.py --config ${CONFIG_FILE} --amp --accumulation-steps 1"
python scripts/train/train_v11m14.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1

echo "Finished at: $(date)"
