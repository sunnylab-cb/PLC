#!/bin/bash
# V11m10 local54 group interaction probes.
#
# Purpose:
#   Test co-attn + local54 group interaction, not just a wider local54 adapter.
#   The design follows current dependency evidence: base33 acts as atom-identity
#   anchor, while electronic3 + sasa_hbond4 + flex_rot5 provide conditional
#   chemistry evidence.  The evidence is injected as bounded residuals into
#   local_delta K/V and/or residue-chem compatibility readout.
#
# UniMol/global policy:
#   Prior source-split showed UniMol/global55 are weak for residue localization
#   when used as direct gates.  Here they are tested only as bounded molecule-
#   level context priors on top of the pooled ligand context; they do not
#   replace or broadcast into atom-level local54 evidence.
#
# Base:
#   - V11m8x co-attn/all_light trunk
#   - no global_delta_gate
#   - local_delta_protein_grad_scale = 0.60
#   - light ligand-source dropout retained from all_light
#
# Matrix:
#   0. seed44_l54kv_coattn_g60
#   1. seed44_l54compat_coattn_g60
#   2. seed44_l54kvcompat_coattn_g60
#   3. seed44_l54kvcompat_gctx_coattn_g60
#   4. seed44_l54kvcompat_uctx_coattn_g60
#   5. seed44_l54kvcompat_bothctx_coattn_g60

#SBATCH --job-name=v11m10_l54i
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/v11m10_l54i_%A_%a.out
#SBATCH --error=logs/v11m10_l54i_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=(
  "seed44_l54kv_coattn_g60"
  "seed44_l54compat_coattn_g60"
  "seed44_l54kvcompat_coattn_g60"
  "seed44_l54kvcompat_gctx_coattn_g60"
  "seed44_l54kvcompat_uctx_coattn_g60"
  "seed44_l54kvcompat_bothctx_coattn_g60"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m10 task id: ${TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m10_l54i_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" <<'PY'
import re
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]

with base_path.open() as f:
    cfg = yaml.safe_load(f)

seed_match = re.search(r"seed(\d+)_", variant)
if not seed_match:
    raise SystemExit(f"Could not parse seed from variant: {variant}")
seed = int(seed_match.group(1))

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_{variant}"
cfg["experiment"]["version_suffix"] = "v11m10"
cfg["experiment"]["seed"] = seed

model = cfg["model"]
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False

# Preserve the current all_light/co-attn baseline.
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

# Do not mix in earlier local54 adapter/reweight probes.
model["use_local54_chem_reweight"] = False
model["use_local54_chem_adapter"] = False

# V11 self-attn front-end unchanged; only later fusion loop uses co-attn block.
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

# V11m10 local54 interaction knobs.
model["local54_interaction_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
model["local54_interaction_hidden"] = 96
model["local54_interaction_kv_alpha_init"] = 0.05
model["local54_interaction_kv_alpha_max"] = 0.30
model["local54_compat_hidden"] = 128
model["local54_compat_alpha_init"] = 0.05
model["local54_compat_alpha_max"] = 0.30

use_kv = "l54kv" in variant
use_compat = "l54compat" in variant or "kvcompat" in variant
model["use_local54_interaction_kv"] = use_kv
model["use_local54_residue_chem_compat"] = use_compat

sources = []
if "gctx" in variant or "bothctx" in variant:
    sources.append("global55")
if "uctx" in variant or "bothctx" in variant:
    sources.append("unimol")
model["use_ligand_context_prior"] = bool(sources)
model["ligand_context_prior_sources"] = sources
model["ligand_context_prior_hidden"] = 192
model["ligand_context_prior_alpha_init"] = 0.03
model["ligand_context_prior_alpha_max"] = 0.20

cfg.setdefault("training", {})
cfg["training"]["batch_size"] = 12
cfg["training"]["init_from_checkpoint"] = None
cfg["training"]["freeze_v11_base"] = False
cfg["training"]["delta_trainable_tokens"] = [
    "local_delta_",
    "global_delta_gate.",
    "local_ligand_film",
    "local54_chem_",
    "local54_anchor_proj.",
    "local54_group_proj.",
    "local54_interaction_",
    "local54_compat_",
    "ligand_context_prior",
    "pocket_",
    "center_aux_",
    "cross_attention_layers.",
]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V11m10 config: {out_path}")
print(
    "variant={variant} seed={seed} batch={batch} grad_scale={scale} "
    "kv={kv} compat={compat} context={context} sources={sources} "
    "groups={groups} coattn={coattn} source_drop={source_drop}".format(
        variant=variant,
        seed=seed,
        batch=cfg["training"]["batch_size"],
        scale=model["local_delta_protein_grad_scale"],
        kv=model["use_local54_interaction_kv"],
        compat=model["use_local54_residue_chem_compat"],
        context=model["use_ligand_context_prior"],
        sources=model["ligand_context_prior_sources"],
        groups=model["local54_interaction_groups"],
        coattn=model["use_v12_coattention_block"],
        source_drop=model["ligand_source_dropout_p"],
    )
)
PY

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(python scripts/train/train_v11m10.py --config "${CONFIG_FILE}" --amp --accumulation-steps 2)
if [ -n "${RESUME_CHECKPOINT}" ]; then
  CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "Generated config:"
sed -n '1,340p' "${CONFIG_FILE}"
echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
