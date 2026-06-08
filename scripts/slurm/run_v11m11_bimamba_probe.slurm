#!/bin/bash
# V11m11 BiMamba-style residual probes.
#
# Purpose:
#   Test whether bidirectional state-space-style context helps the current
#   V11m10 l54kv + co-attn path.  The probes are bounded residuals and do not
#   replace residue-ligand cross attention.
#
# Base:
#   - V11m10 l54kv, no compat, no global/UniMol context prior
#   - V11m8x co-attn/all_light trunk
#   - local_delta_protein_grad_scale = 0.60
#
# Matrix:
#   0. seed44_bimamba_respost1_l54kv_coattn_g60
#   1. seed44_bimamba_respost2_l54kv_coattn_g60
#   2. seed44_bimamba_ligkv1_l54kv_coattn_g60
#   3. seed44_bimamba_ligkv2_l54kv_coattn_g60
#   4. seed44_mambamem4_l54kv_coattn_g60
#   5. seed44_mambamem8_l54kv_coattn_g60
#   6. seed44_bimamba_ligkv1_mem4_l54kv_coattn_g60

#SBATCH --job-name=v11m11_bima
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/v11m11_bima_%A_%a.out
#SBATCH --error=logs/v11m11_bima_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=(
  "seed44_bimamba_respost1_l54kv_coattn_g60"
  "seed44_bimamba_respost2_l54kv_coattn_g60"
  "seed44_bimamba_ligkv1_l54kv_coattn_g60"
  "seed44_bimamba_ligkv2_l54kv_coattn_g60"
  "seed44_mambamem4_l54kv_coattn_g60"
  "seed44_mambamem8_l54kv_coattn_g60"
  "seed44_bimamba_ligkv1_mem4_l54kv_coattn_g60"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m11 BiMamba task id: ${TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m11_bima_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

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
cfg["experiment"]["version_suffix"] = "v11m11"
cfg["experiment"]["seed"] = seed

model = cfg["model"]
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False

# Preserve V11m8x co-attn/all_light baseline.
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

# Current strong local54 path: K/V interaction only.  Do not mix in compat or
# ligand-level context priors for this architecture probe.
model["use_local54_interaction_kv"] = True
model["use_local54_residue_chem_compat"] = False
model["local54_interaction_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
model["local54_interaction_hidden"] = 96
model["local54_interaction_kv_alpha_init"] = 0.05
model["local54_interaction_kv_alpha_max"] = 0.30
model["use_ligand_context_prior"] = False
model["ligand_context_prior_sources"] = []

# BiMamba-style residual knobs.
model["bimamba_kernel_size"] = 7
model["bimamba_dropout"] = 0.0
model["use_bimamba_residue_post"] = "respost" in variant
model["bimamba_residue_layers"] = 2 if "respost2" in variant else 1
model["bimamba_residue_alpha_init"] = 0.03
model["bimamba_residue_alpha_max"] = 0.20

model["use_bimamba_ligand_kv"] = "ligkv" in variant
model["bimamba_ligand_layers"] = 2 if "ligkv2" in variant else 1
model["bimamba_ligand_alpha_init"] = 0.03
model["bimamba_ligand_alpha_max"] = 0.20

model["use_mamba_lig_memory"] = "mambamem" in variant or "mem4" in variant
model["mamba_lig_memory_tokens"] = 8 if "mem8" in variant else 4
model["mamba_lig_memory_layers"] = 1
model["mamba_lig_memory_alpha_init"] = 0.03
model["mamba_lig_memory_alpha_max"] = 0.20
model["mamba_lig_memory_score_bias_init"] = -4.0

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
    "bimamba_",
    "mamba_lig_",
    "pocket_",
    "center_aux_",
    "cross_attention_layers.",
]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V11m11 BiMamba config: {out_path}")
print(
    "variant={variant} seed={seed} respost={respost} res_layers={res_layers} "
    "ligkv={ligkv} lig_layers={lig_layers} memory={memory} mem_tokens={mem_tokens} "
    "l54kv={l54kv} coattn={coattn} grad_scale={grad_scale}".format(
        variant=variant,
        seed=seed,
        respost=model["use_bimamba_residue_post"],
        res_layers=model["bimamba_residue_layers"],
        ligkv=model["use_bimamba_ligand_kv"],
        lig_layers=model["bimamba_ligand_layers"],
        memory=model["use_mamba_lig_memory"],
        mem_tokens=model["mamba_lig_memory_tokens"],
        l54kv=model["use_local54_interaction_kv"],
        coattn=model["use_v12_coattention_block"],
        grad_scale=model["local_delta_protein_grad_scale"],
    )
)
PY

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(python scripts/train/train_v11m11.py --config "${CONFIG_FILE}" --amp --accumulation-steps 2)
if [ -n "${RESUME_CHECKPOINT}" ]; then
  CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "Generated config:"
sed -n '1,380p' "${CONFIG_FILE}"
echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
