#!/bin/bash
# V11m12 pure late-branch UniMol warm-start probes.
#
# Purpose:
#   The previous V11m12 probe was not a pure late-branch UniMol test because it
#   inherited use_unimol_ligand_input=true from V11m6n. This script keeps that
#   tensor shape for checkpoint compatibility, but zeros/freezes the direct
#   UniMol atom-input alpha after loading a V11m10 checkpoint. UniMol can then
#   influence predictions only through V11m12 late branches.
#
# Base:
#   - initialize from V11m10 l54kvcompat/coattn best checkpoint
#   - freeze the V11/V11m10 base
#   - train only unimol_* late-branch parameters
#   - keep local54/graph evidence path fixed
#
# Matrix:
#   0/1  seed44_u_rescompat_lr2e4/lr5e4
#   2/3  seed44_u_groupgate_lr2e4/lr5e4
#   4/5  seed44_u_mem1b2_lr2e4/lr5e4
#   6/7  seed44_u_mem4b2_lr2e4/lr5e4
#   8/9  seed44_u_res_group_lr2e4/lr5e4
#
# Success criteria:
#   - Val/test F1 improves over V11m10 by >=0.003-0.005.
#   - zero_unimol_only drop >=0.01 after dependency eval.
#   - Health/unimol_*_abs_mean is non-zero and stable.

#SBATCH --job-name=v11m12_uw
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/v11m12_uw_%A_%a.out
#SBATCH --error=logs/v11m12_uw_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/pocket_gvp_fusion_seed44_l54kvcompat_coattn_g60_v11m10_20260607_231717/best_f1_c0_model.pt}"

VARIANTS=(
  "seed44_u_rescompat_lr2e4"
  "seed44_u_rescompat_lr5e4"
  "seed44_u_groupgate_lr2e4"
  "seed44_u_groupgate_lr5e4"
  "seed44_u_mem1b2_lr2e4"
  "seed44_u_mem1b2_lr5e4"
  "seed44_u_mem4b2_lr2e4"
  "seed44_u_mem4b2_lr5e4"
  "seed44_u_res_group_lr2e4"
  "seed44_u_res_group_lr5e4"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m12 warm-start task id: ${TASK_ID}"
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
echo "Init checkpoint: ${INIT_CHECKPOINT}"
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

if [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "ERROR: INIT_CHECKPOINT not found: ${INIT_CHECKPOINT}"
  exit 3
fi

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
CONFIG_FILE="tmp/slurm_configs/v11m12_uw_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

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

lr = 2.0e-4 if variant.endswith("lr2e4") else 5.0e-4

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_{variant}"
cfg["experiment"]["version_suffix"] = "v11m12"
cfg["experiment"]["seed"] = seed

model = cfg["model"]

# Keep direct UniMol input enabled only for tensor-shape compatibility with the
# V11m10 checkpoint. train_v11m12 zeroes/freezes the direct alpha after loading.
model["use_unimol_ligand_input"] = True
model["d_unimol"] = 512
model["d_unimol_ligand_proj"] = 64

model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False

# Preserve the current V11m10/all-light/co-attn base.
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

model["use_local54_chem_reweight"] = False
model["use_local54_chem_adapter"] = False

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

model["local54_interaction_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
model["local54_interaction_hidden"] = 96
model["local54_interaction_kv_alpha_init"] = 0.05
model["local54_interaction_kv_alpha_max"] = 0.30
model["local54_compat_hidden"] = 128
model["local54_compat_gate_bias_init"] = 0.05
model["local54_compat_alpha_init"] = 0.05
model["local54_compat_alpha_max"] = 0.30
model["local54_compat_normalize_read"] = True
model["local54_compat_read_scale"] = 0.35
model["local54_compat_read_clip"] = 0.0
model["use_local54_interaction_kv"] = True
model["use_local54_residue_chem_compat"] = True

model["use_ligand_context_prior"] = False
model["ligand_context_prior_sources"] = []

model["use_unimol_memory_kv"] = False
model["unimol_memory_tokens"] = 1
model["unimol_memory_hidden"] = 192
model["unimol_memory_alpha_init"] = 0.05
model["unimol_memory_alpha_max"] = 0.25
model["unimol_memory_score_bias_init"] = -2.0
model["use_unimol_group_gate"] = False
model["unimol_group_gate_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
model["unimol_group_gate_hidden"] = 128
model["unimol_group_gate_alpha_init"] = 0.05
model["unimol_group_gate_alpha_max"] = 0.25
model["use_unimol_atom_gate"] = False
model["unimol_atom_gate_hidden"] = 96
model["unimol_atom_gate_alpha_init"] = 0.03
model["unimol_atom_gate_alpha_max"] = 0.20
model["use_unimol_residue_compat"] = False
model["unimol_residue_compat_hidden"] = 128
model["unimol_residue_compat_alpha_init"] = 0.05
model["unimol_residue_compat_alpha_max"] = 0.25

if "_u_rescompat_" in variant:
    model["use_unimol_residue_compat"] = True
elif "_u_groupgate_" in variant:
    model["use_unimol_group_gate"] = True
elif "_u_mem1b2_" in variant:
    model["use_unimol_memory_kv"] = True
    model["unimol_memory_tokens"] = 1
elif "_u_mem4b2_" in variant:
    model["use_unimol_memory_kv"] = True
    model["unimol_memory_tokens"] = 4
elif "_u_res_group_" in variant:
    model["use_unimol_residue_compat"] = True
    model["use_unimol_group_gate"] = True
else:
    raise SystemExit(f"Unknown V11m12 warm-start variant: {variant}")

cfg.setdefault("training", {})
train = cfg["training"]
train["init_from_checkpoint"] = init_checkpoint
train["freeze_v11_base"] = True
train["zero_direct_unimol_ligand_input"] = True
train["delta_trainable_tokens"] = [
    "unimol_memory_",
    "unimol_group_gate",
    "unimol_atom_",
    "unimol_residue_",
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

print("Wrote config:", out_path)
print(
    "Variant={variant} lr={lr:.1e} init={init} direct_unimol_shape=true "
    "direct_alpha_zeroed=true freeze_base=true trainable=unimol_*".format(
        variant=variant,
        lr=lr,
        init=init_checkpoint,
    )
)
PY

echo "Generated config:"
sed -n '1,260p' "${CONFIG_FILE}"

echo "Command: python scripts/train/train_v11m12.py --config ${CONFIG_FILE} --amp --accumulation-steps 1"
python scripts/train/train_v11m12.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1

echo "Finished at: $(date)"
