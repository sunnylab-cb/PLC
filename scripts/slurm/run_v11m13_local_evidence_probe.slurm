#!/bin/bash
# V11m13 local evidence enhancement probes.
#
# Current strongest single-seed test-real model is V11m8x coattn seed44
# (~0.6147 test real).  This probe warm-starts from that checkpoint, freezes the
# existing trunk/local_delta base, and trains only small local-evidence modules.
#
# Axes:
#   groupkv     : local54 group-wise K/V residual.
#   edgekv      : explicit bond-edge-context K/V residual.
#   rescompat   : residue x local54 chemistry compatibility readout.
#   chem_adapter: raw local54 chemistry adapter/reweight control.
#
# The goal is to see whether local54/graph source-split gains can be converted
# into higher real F1 without losing real-shuffle gap.

#SBATCH --job-name=v11m13_loc
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-8
#SBATCH --output=logs/v11m13_loc_%A_%a.out
#SBATCH --error=logs/v11m13_loc_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-}"

VARIANTS=(
  "seed44_groupkv_lr2e4"
  "seed44_edgekv_lr2e4"
  "seed44_rescompat_lr2e4"
  "seed44_group_edge_lr2e4"
  "seed44_group_rescompat_lr2e4"
  "seed44_edge_rescompat_lr2e4"
  "seed44_all3_lr2e4"
  "seed44_chem_adapter_lr2e4"
  "seed44_all3_lr5e4"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m13 local evidence task id: ${TASK_ID}"
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

if [ -z "${INIT_CHECKPOINT}" ]; then
  INIT_CHECKPOINT="$(ls -dt outputs/pocket_gvp_fusion_seed44_coattn_g60_v11m8x_*/best_f1_c0_model.pt 2>/dev/null | head -1 || true)"
fi
if [ -z "${INIT_CHECKPOINT}" ] || [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "ERROR: V11m8x coattn seed44 init checkpoint not found."
  echo "Set INIT_CHECKPOINT=/path/to/best_f1_c0_model.pt if the output dir name changed."
  exit 3
fi

echo "Init checkpoint: ${INIT_CHECKPOINT}"
echo "Python: $(which python)"
python --version

echo "GPU devices:"
if command -v rocm-smi &> /dev/null; then
  rocm-smi
elif command -v nvidia-smi &> /dev/null; then
  nvidia-smi
else
  echo "No rocm-smi/nvidia-smi found"
fi

mkdir -p tmp/slurm_configs
CONFIG_FILE="tmp/slurm_configs/v11m13_loc_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${INIT_CHECKPOINT}" <<'PY'
import re
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
init_checkpoint = sys.argv[4]

with base_path.open() as f:
    cfg = yaml.safe_load(f)

seed_match = re.search(r"seed(\d+)_", variant)
if not seed_match:
    raise SystemExit(f"Could not parse seed from variant: {variant}")
seed = int(seed_match.group(1))
lr = 5.0e-4 if variant.endswith("lr5e4") else 2.0e-4

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_{variant}"
cfg["experiment"]["version_suffix"] = "v11m13"
cfg["experiment"]["seed"] = seed

model = cfg["model"]
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False

# Preserve the current V11m8x co-attn/all-light base.
model["use_local_ligand_film"] = True
model["use_pocket_virtual_tokens"] = True
model["use_center_aux_head"] = True
model["ligand_source_dropout_p"] = 0.03
model["ligand_dropout_local54_p"] = 0.04
model["ligand_dropout_hidden_p"] = 0.04

model["local_ligand_film_hidden"] = 256
model["local_ligand_film_alpha_init"] = 0.05
model["local_ligand_film_alpha_max"] = 0.30
model["pocket_virtual_tokens"] = 4
model["pocket_virtual_alpha_init"] = 0.05
model["pocket_virtual_alpha_max"] = 0.30
model["center_aux_hidden"] = 128
model["center_aux_logit_scale_init"] = 0.05
model["center_aux_logit_scale_max"] = 0.30

model["use_v12_cross_attention"] = False
model["use_v12_coattention_block"] = True
model["v12_cross_attn_logit_clip"] = 30.0
model["v12_cross_use_flash_attn"] = False
model["v12_cross_flash_attn_fallback"] = True
model["v12_coattn_use_film"] = False
model["v12_coattn_use_prot_dist_bias"] = False
model["v12_coattn_use_lig_dist_bias"] = False
model["v12_coattn_self_attn_logit_clip"] = 30.0
model["v12_coattn_inner_residual_scale"] = 0.05
model["v12_coattn_cross_residual_scale"] = 0.25
model["v12_coattn_use_rezero_branch_gates"] = True
model["v12_coattn_rezero_gate_init"] = 0.05
model["v12_coattn_p2l_length_scale"] = False
model["v12_coattn_p2l_rezero_gate_max"] = 0.20
model["v12_coattn_zero_init_residuals"] = True

# Start all local evidence probes disabled; enable per variant below.
model["use_local54_chem_reweight"] = False
model["use_local54_chem_adapter"] = False
model["local54_chem_adapter_hidden"] = 32
model["local54_chem_adapter_scale_init"] = 0.025
model["local54_chem_adapter_scale_max"] = 0.15
model["local54_chem_base_scale"] = 1.00
model["local54_chem_electronic_scale"] = 1.15
model["local54_chem_sasa_hbond_scale"] = 1.15
model["local54_chem_flex_rot_scale"] = 1.10
model["local54_chem_metal_scale"] = 1.00
model["local54_chem_stereo_scale"] = 1.00

model["local54_interaction_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
model["local54_interaction_hidden"] = 96
model["local54_interaction_kv_alpha_init"] = 0.035
model["local54_interaction_kv_alpha_max"] = 0.20
model["use_local54_interaction_kv"] = False

model["use_local54_edge_context_kv"] = False
model["local54_edge_context_hidden"] = 96
model["local54_edge_context_alpha_init"] = 0.03
model["local54_edge_context_alpha_max"] = 0.20

model["use_local54_residue_chem_compat"] = False
model["local54_compat_hidden"] = 128
model["local54_compat_gate_bias_init"] = 0.01
model["local54_compat_alpha_init"] = 0.02
model["local54_compat_alpha_max"] = 0.12
model["local54_compat_normalize_read"] = True
model["local54_compat_read_scale"] = 0.35
model["local54_compat_read_clip"] = 0.0

if "groupkv" in variant or "group_edge" in variant or "group_rescompat" in variant or "all3" in variant:
    model["use_local54_interaction_kv"] = True
if "edgekv" in variant or "group_edge" in variant or "edge_rescompat" in variant or "all3" in variant:
    model["use_local54_edge_context_kv"] = True
if "rescompat" in variant or "group_rescompat" in variant or "edge_rescompat" in variant or "all3" in variant:
    model["use_local54_residue_chem_compat"] = True
if "chem_adapter" in variant:
    model["use_local54_chem_reweight"] = True
    model["use_local54_chem_adapter"] = True

model["use_ligand_context_prior"] = False
model["ligand_context_prior_sources"] = []

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
    "local54_edge_context_",
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

print("Wrote V11m13 local evidence config:", out_path)
print(
    "variant={variant} lr={lr:.1e} init={init} groupkv={groupkv} edgekv={edgekv} "
    "rescompat={rescompat} chem_adapter={chem_adapter} freeze_base=true".format(
        variant=variant,
        lr=lr,
        init=init_checkpoint,
        groupkv=model["use_local54_interaction_kv"],
        edgekv=model["use_local54_edge_context_kv"],
        rescompat=model["use_local54_residue_chem_compat"],
        chem_adapter=model["use_local54_chem_adapter"],
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
