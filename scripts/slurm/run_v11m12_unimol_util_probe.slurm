#!/bin/bash
# V11m12 UniMol utilization probes.
#
# Purpose:
#   Prior source-split/dependency eval repeatedly showed near-zero contribution
#   from zero_unimol_only (~0.002-0.005 F1 drop), while local54/ligand graph
#   dominate residue localization. This matrix tests whether UniMol becomes
#   useful when routed as a separate late branch instead of being broadcast into
#   dense atom features or mixed into a pooled context prior.
#
# External motivation:
#   UniMol/UniMol2 are molecule-level 3D representation models; their signal is
#   more naturally used as global/multi-level control over local evidence than
#   as raw atom feature replacement. FABind-style ligand-informed pocket modules
#   also support late ligand-to-pocket routing. Therefore these probes test:
#     - u_mem1/u_mem4: UniMol memory K/V tokens in local_delta attention.
#     - u_groupgate: UniMol gates local54 evidence groups.
#     - u_atomgate: UniMol-conditioned atom K/V scaling.
#     - u_rescompat: low-rank residue-UniMol compatibility logit residual.
#
# Base:
#   - V11m10 local54 K/V + residue-chem compatibility.
#   - V11m8x co-attn/all_light trunk.
#   - no global_delta_gate.
#   - local_delta_protein_grad_scale = 0.60.
#   - old ligand_context_prior disabled to isolate the new UniMol branches.
#
# Matrix:
#   0/1 seed44/45_u_mem1_g60
#   2/3 seed44/45_u_mem4_g60
#   4/5 seed44/45_u_groupgate_g60
#   6/7 seed44/45_u_atomgate_g60
#   8/9 seed44/45_u_rescompat_g60

#SBATCH --job-name=v11m12_uni
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/v11m12_uni_%A_%a.out
#SBATCH --error=logs/v11m12_uni_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=(
  "seed44_u_mem1_g60"
  "seed45_u_mem1_g60"
  "seed44_u_mem4_g60"
  "seed45_u_mem4_g60"
  "seed44_u_groupgate_g60"
  "seed45_u_groupgate_g60"
  "seed44_u_atomgate_g60"
  "seed45_u_atomgate_g60"
  "seed44_u_rescompat_g60"
  "seed45_u_rescompat_g60"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m12 task id: ${TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m12_uni_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

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
cfg["experiment"]["version_suffix"] = "v11m12"
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

# Strong local evidence base from V11m10.
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

# Disable old context-prior path. V11m12 tests explicit UniMol late branches.
model["use_ligand_context_prior"] = False
model["ligand_context_prior_sources"] = []

# Reset all V11m12 branches, then enable one per variant.
model["use_unimol_memory_kv"] = False
model["unimol_memory_tokens"] = 1
model["unimol_memory_hidden"] = 192
model["unimol_memory_alpha_init"] = 0.03
model["unimol_memory_alpha_max"] = 0.20
model["unimol_memory_score_bias_init"] = -4.0
model["use_unimol_group_gate"] = False
model["unimol_group_gate_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
model["unimol_group_gate_hidden"] = 128
model["unimol_group_gate_alpha_init"] = 0.03
model["unimol_group_gate_alpha_max"] = 0.25
model["use_unimol_atom_gate"] = False
model["unimol_atom_gate_hidden"] = 96
model["unimol_atom_gate_alpha_init"] = 0.03
model["unimol_atom_gate_alpha_max"] = 0.20
model["use_unimol_residue_compat"] = False
model["unimol_residue_compat_hidden"] = 128
model["unimol_residue_compat_alpha_init"] = 0.03
model["unimol_residue_compat_alpha_max"] = 0.20

if "_u_mem1_" in variant:
    model["use_unimol_memory_kv"] = True
    model["unimol_memory_tokens"] = 1
elif "_u_mem4_" in variant:
    model["use_unimol_memory_kv"] = True
    model["unimol_memory_tokens"] = 4
elif "_u_groupgate_" in variant:
    model["use_unimol_group_gate"] = True
elif "_u_atomgate_" in variant:
    model["use_unimol_atom_gate"] = True
elif "_u_rescompat_" in variant:
    model["use_unimol_residue_compat"] = True
else:
    raise SystemExit(f"Unknown V11m12 variant: {variant}")

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
    "unimol_memory_",
    "unimol_group_gate",
    "unimol_atom_",
    "unimol_residue_",
    "pocket_",
    "center_aux_",
    "cross_attention_layers.",
]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V11m12 config: {out_path}")
print(
    "variant={variant} seed={seed} batch={batch} grad_scale={scale} "
    "u_mem={u_mem} u_mem_tokens={u_mem_tokens} u_group={u_group} "
    "u_atom={u_atom} u_res={u_res} l54_kv={l54_kv} l54_compat={l54_compat} "
    "compat_norm={compat_norm} compat_scale={compat_scale} coattn={coattn}".format(
        variant=variant,
        seed=seed,
        batch=cfg["training"]["batch_size"],
        scale=model["local_delta_protein_grad_scale"],
        u_mem=model["use_unimol_memory_kv"],
        u_mem_tokens=model["unimol_memory_tokens"],
        u_group=model["use_unimol_group_gate"],
        u_atom=model["use_unimol_atom_gate"],
        u_res=model["use_unimol_residue_compat"],
        l54_kv=model["use_local54_interaction_kv"],
        l54_compat=model["use_local54_residue_chem_compat"],
        compat_norm=model["local54_compat_normalize_read"],
        compat_scale=model["local54_compat_read_scale"],
        coattn=model["use_v12_coattention_block"],
    )
)
PY

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(python scripts/train/train_v11m12.py --config "${CONFIG_FILE}" --amp --accumulation-steps 2)
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
