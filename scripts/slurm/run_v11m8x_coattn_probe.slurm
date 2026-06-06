#!/bin/bash
# V11m8x V12 CoAttentionBlock probe.
#
# Purpose:
#   Corrected branch for testing "V11 front-end self-attn unchanged, but the
#   later cross-fusion loop replaced by V12-style CoAttentionBlock".
#
# What stays unchanged:
#   - V11 protein context encoder before fusion
#   - V11 ligand self-attention before fusion
#   - V11m8 all_light local_delta / FiLM / pocket virtual tokens / center head
#   - no global/UniMol gate, grad_scale = 0.60
#
# What changes:
#   - Each V11 fusion-loop layer is replaced by a V12 CoAttentionBlock adapter.
#   - V12 block internal FiLM and distance biases are disabled for this probe,
#     so the new variable is the self+cross co-attention block inside fusion.
#   - Gradient checkpointing is disabled because the adapter caches the ligand
#     stream between V11's prot_to_lig() and lig_to_prot() calls.
#
# Memory policy:
#   - batch_size = 12, accumulation_steps = 2, effective batch about 24.
#
# Matrix:
#   0. seed44_coattn_g60
#   1. seed45_coattn_g60

#SBATCH --job-name=v11m8x_coa
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=36:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/v11m8x_coa_%A_%a.out
#SBATCH --error=logs/v11m8x_coa_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=(
  "seed44_coattn_g60"
  "seed45_coattn_g60"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m8x co-attn task id: ${TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m8x_coattn_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

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
cfg["experiment"]["version_suffix"] = "v11m8x"
cfg["experiment"]["seed"] = seed

model = cfg["model"]
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False

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

# Keep V11 front-end self-attn unchanged.  The V12 CoAttentionBlock is used
# only inside the later fusion loop; internal distance/UniMol FiLM branches are
# disabled to avoid reintroducing V12's full front-end assumptions.
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

cfg.setdefault("training", {})
cfg["training"]["batch_size"] = 12
cfg["training"]["init_from_checkpoint"] = None
cfg["training"]["freeze_v11_base"] = False
cfg["training"]["delta_trainable_tokens"] = [
    "local_delta_",
    "global_delta_gate.",
    "local_ligand_film",
    "local54_chem_",
    "pocket_",
    "center_aux_",
    "cross_attention_layers.",
]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V11m8x co-attn config: {out_path}")
print(
    "variant={variant} seed={seed} batch={batch} grad_scale={scale} coattn={coattn} "
    "inner={inner} cross={cross} gate_init={gate_init} zero_init={zero_init} "
    "all_light=(film={film},virtual={virtual},center={center}) source_drop={source_drop}".format(
        variant=variant,
        seed=seed,
        batch=cfg["training"]["batch_size"],
        scale=model["local_delta_protein_grad_scale"],
        coattn=model["use_v12_coattention_block"],
        inner=model["v12_coattn_inner_residual_scale"],
        cross=model["v12_coattn_cross_residual_scale"],
        gate_init=model["v12_coattn_rezero_gate_init"],
        zero_init=model["v12_coattn_zero_init_residuals"],
        film=model["use_local_ligand_film"],
        virtual=model["use_pocket_virtual_tokens"],
        center=model["use_center_aux_head"],
        source_drop=model["ligand_source_dropout_p"],
    )
)
PY

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(python scripts/train/train_v11m8x.py --config "${CONFIG_FILE}" --amp --accumulation-steps 2)
if [ -n "${RESUME_CHECKPOINT}" ]; then
  CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "Generated config:"
sed -n '1,320p' "${CONFIG_FILE}"
echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
