#!/bin/bash
# V11m13 base33-anchor + conditional local54 evidence probes.
#
# Motivation:
#   The group_rescompat_lr2e4 model is the current strongest single-seed test
#   model (test real F1_c0 ~= 0.6158, real-shuffle ~= 0.080).  Its local54
#   subgroup scan showed:
#     - full local54 drop ~= 0.056
#     - base33 is the largest single subgroup drop
#     - electronic3 / sasa_hbond4 / flex_rot5 provide smaller but consistent
#       conditional evidence
#     - edge_attr/topology are negligible
#
# Probe:
#   Keep the V11m8x/co-attn trunk and V11m10 group-rescompat idea, but include
#   base33 as a direct interaction group in addition to its existing anchor role:
#     groups = [base33, electronic3, sasa_hbond4, flex_rot5]
#
#   Most variants warm-start from the best group_rescompat_lr2e4 checkpoint.
#   One control starts from the original V11m8x coattn checkpoint.

#SBATCH --job-name=v11m13_b4cond
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/v11m13_b4cond_%A_%a.out
#SBATCH --error=logs/v11m13_b4cond_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
PREV_CHECKPOINT="${PREV_CHECKPOINT:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt}"
COATTN_CHECKPOINT="${COATTN_CHECKPOINT:-outputs/pocket_gvp_fusion_seed44_coattn_g60_v11m8x_20260606_200217/best_f1_c0_model.pt}"

VARIANTS=(
  "seed44_base4_grc_prev_lr1e4"
  "seed44_base4_grc_prev_lr2e4"
  "seed44_base4_grc_soft_prev_lr2e4"
  "seed44_base4_rescompat_prev_lr2e4"
  "seed44_base4_groupkv_prev_lr2e4"
  "seed44_base4_grc_chem_prev_lr2e4"
  "seed44_base4_grc_coattn_lr2e4"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m13 base4 conditional task id: ${TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m13_base4_cond_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

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
lr = 1.0e-4 if "lr1e4" in variant else 2.0e-4
init_checkpoint = coattn_checkpoint if "_coattn_" in variant else prev_checkpoint

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_{variant}"
cfg["experiment"]["version_suffix"] = "v11m13"
cfg["experiment"]["seed"] = seed

model = cfg["model"]

# Main hypothesis: base33 remains the anchor, but is also allowed to contribute
# as an explicit evidence group alongside electronic/sasa/flex conditionals.
model["local54_interaction_groups"] = [
    "base33",
    "electronic3",
    "sasa_hbond4",
    "flex_rot5",
]
model["local54_interaction_hidden"] = 96

# Keep current best V11m8x/co-attn trunk behavior.
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False
model["use_v12_cross_attention"] = False
model["use_v12_coattention_block"] = True

# Edge topology was negligible in the subgroup scan; keep it disabled.
model["use_local54_edge_context_kv"] = False

# Variant switches.
model["use_local54_interaction_kv"] = "groupkv" in variant or "_grc_" in variant
model["use_local54_residue_chem_compat"] = "rescompat" in variant or "_grc_" in variant

model["use_local54_chem_reweight"] = "chem" in variant
model["use_local54_chem_adapter"] = "chem" in variant
model["local54_chem_adapter_hidden"] = 32
model["local54_chem_adapter_scale_init"] = 0.02
model["local54_chem_adapter_scale_max"] = 0.12
model["local54_chem_base_scale"] = 1.05 if "chem" in variant else 1.00
model["local54_chem_electronic_scale"] = 1.15
model["local54_chem_sasa_hbond_scale"] = 1.15
model["local54_chem_flex_rot_scale"] = 1.10
model["local54_chem_metal_scale"] = 1.00
model["local54_chem_stereo_scale"] = 1.00

if "soft" in variant:
    model["local54_interaction_kv_alpha_init"] = 0.020
    model["local54_interaction_kv_alpha_max"] = 0.12
    model["local54_compat_gate_bias_init"] = 0.005
    model["local54_compat_alpha_init"] = 0.012
    model["local54_compat_alpha_max"] = 0.08
    model["local54_compat_normalize_read"] = True
    model["local54_compat_read_scale"] = 0.25
else:
    model["local54_interaction_kv_alpha_init"] = 0.035
    model["local54_interaction_kv_alpha_max"] = 0.20
    model["local54_compat_gate_bias_init"] = 0.010
    model["local54_compat_alpha_init"] = 0.020
    model["local54_compat_alpha_max"] = 0.12
    model["local54_compat_normalize_read"] = True
    model["local54_compat_read_scale"] = 0.35
model["local54_compat_read_clip"] = 0.0

cfg.setdefault("training", {})
train = cfg["training"]
train["init_from_checkpoint"] = init_checkpoint
train["freeze_v11_base"] = True
train["delta_trainable_tokens"] = [
    "local54_chem_",
    "local54_anchor_proj.",
    "local54_group_proj.",
    "local54_interaction_",
    "local54_compat_",
]
train["learning_rate"] = lr
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 2
train["num_epochs"] = 60
train["batch_size"] = 24
train["accumulation_steps"] = 1
train["early_stopping_patience"] = 60

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Wrote V11m13 base4 conditional config:", out_path)
print(
    "variant={variant} seed={seed} lr={lr:.1e} init={init} groups={groups} "
    "kv={kv} compat={compat} chem={chem} soft={soft}".format(
        variant=variant,
        seed=seed,
        lr=lr,
        init=init_checkpoint,
        groups=model["local54_interaction_groups"],
        kv=model["use_local54_interaction_kv"],
        compat=model["use_local54_residue_chem_compat"],
        chem=model["use_local54_chem_adapter"],
        soft="soft" in variant,
    )
)
PY

echo "Generated config:"
sed -n '1,260p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Command: python scripts/train/train_v11m13.py --config ${CONFIG_FILE} --amp --accumulation-steps 1"
python scripts/train/train_v11m13.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1

echo "Finished at: $(date)"
