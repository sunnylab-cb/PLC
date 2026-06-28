#!/bin/bash
# V13m0 residue-atom compatibility bias probes.
#
# Motivation:
#   ESMFold uses pair representations as dynamic attention biases.  V13m0 tests
#   the analogous low-risk PLC idea without adding a full protein-protein pair
#   trunk: add a bounded residue-atom chemistry compatibility bias directly to
#   local_delta residue-atom attention scores.
#
# Variants:
#   0 coattn_resatom_g60
#       Warm-start from current strongest co-attn V11m13 group_rescompat.
#       Train only the new residue-atom bias module.
#   1 coattn_resatom_tri_g60
#       Same, plus lightweight triangle-inspired local coherence over the
#       residue-atom score field.
#   2 noncoattn_resatom_g60
#       Warm-start from strongest non-co-attn V11m6n grad060_no_gate baseline.
#       Tests whether the bias helps without the V12-style co-attn block.
#   3 noncoattn_resatom_tri_g60
#       Same non-co-attn baseline, plus the triangle-inspired score coherence.
#
# Training policy:
#   Freeze the warm-started model and train only newly added V13m0 modules.
#   This isolates whether residue-atom bias itself is useful.  If positive, a
#   second pass can unfreeze local_delta/classifier for ceiling.

#SBATCH --job-name=v13m0_rab
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/v13m0_rab_%A_%a.out
#SBATCH --error=logs/v13m0_rab_%A_%a.err

# Avoid `set -u`: this cluster's conda deactivate hook can read an unset
# _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED variable.
set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
COATTN_CONFIG="${COATTN_CONFIG:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
COATTN_CKPT="${COATTN_CKPT:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt}"
NONCOATTN_CONFIG="${NONCOATTN_CONFIG:-outputs/pocket_gvp_fusion_grad060_no_gate_v11m6n_20260529_202523/config.yaml}"
NONCOATTN_CKPT="${NONCOATTN_CKPT:-outputs/pocket_gvp_fusion_grad060_no_gate_v11m6n_20260529_202523/best_f1_c0_model.pt}"

VARIANTS=(
  "coattn_resatom_g60"
  "coattn_resatom_tri_g60"
  "noncoattn_resatom_g60"
  "noncoattn_resatom_tri_g60"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V13m0 task id: ${TASK_ID}"
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
echo "Project: ${PROJECT_DIR}"
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

if [[ "${VARIANT}" == coattn_* ]]; then
  BASE_CONFIG="${COATTN_CONFIG}"
  INIT_CHECKPOINT="${COATTN_CKPT}"
else
  BASE_CONFIG="${NONCOATTN_CONFIG}"
  INIT_CHECKPOINT="${NONCOATTN_CKPT}"
fi

if [ ! -f "${BASE_CONFIG}" ]; then
  echo "ERROR: base config not found: ${BASE_CONFIG}"
  exit 3
fi
if [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "ERROR: init checkpoint not found: ${INIT_CHECKPOINT}"
  exit 4
fi

echo "Base config: ${BASE_CONFIG}"
echo "Init checkpoint: ${INIT_CHECKPOINT}"
echo "Python: $(which python)"
python --version

mkdir -p tmp/slurm_configs
CONFIG_FILE="tmp/slurm_configs/v13m0_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${INIT_CHECKPOINT}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
init_checkpoint = sys.argv[4]

with base_path.open() as f:
    cfg = yaml.safe_load(f)

is_coattn = variant.startswith("coattn_")
use_triangle = "_tri_" in variant

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_{variant}"
cfg["experiment"]["version_suffix"] = "v13m0"
cfg["experiment"]["seed"] = 44

model = cfg["model"]

# Keep the current ligand-aware late-delta regime.
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False

# Explicitly define the co-attn vs non-co-attn baselines.
if is_coattn:
    model["use_v12_cross_attention"] = False
    model["use_v12_coattention_block"] = True
    model["use_local54_interaction_kv"] = True
    model["use_local54_residue_chem_compat"] = True
    model["use_local54_edge_context_kv"] = False
    model["use_ligand_context_prior"] = False
    model["ligand_context_prior_sources"] = []
    model["local54_interaction_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
    model["local54_interaction_hidden"] = 96
    model["local54_interaction_kv_alpha_init"] = 0.035
    model["local54_interaction_kv_alpha_max"] = 0.20
    model["local54_compat_hidden"] = 128
    model["local54_compat_gate_bias_init"] = 0.01
    model["local54_compat_alpha_init"] = 0.02
    model["local54_compat_alpha_max"] = 0.12
    model["local54_compat_normalize_read"] = True
    model["local54_compat_read_scale"] = 0.35
    model["local54_compat_read_clip"] = 0.0
else:
    model["use_v12_cross_attention"] = False
    model["use_v12_coattention_block"] = False
    model["use_local54_interaction_kv"] = False
    model["use_local54_residue_chem_compat"] = False
    model["use_local54_edge_context_kv"] = False
    model["use_local54_chem_adapter"] = False
    model["use_local54_chem_reweight"] = False
    model["use_ligand_context_prior"] = False
    model["ligand_context_prior_sources"] = []

# V13m0 residue-atom compatibility score bias.
model["use_residue_atom_bias"] = True
model["residue_atom_bias_hidden"] = 48
model["residue_atom_bias_local54_groups"] = [
    "base33",
    "electronic3",
    "sasa_hbond4",
    "flex_rot5",
]
model["residue_atom_bias_alpha_init"] = 0.03
model["residue_atom_bias_alpha_max"] = 0.18
model["residue_atom_bias_clip"] = 6.0

# Lightweight triangle-inspired local score coherence, not a full [L,L,C] trunk.
model["use_residue_atom_triangle_bias"] = use_triangle
model["residue_atom_triangle_window"] = 7
model["residue_atom_triangle_alpha_init"] = 0.02
model["residue_atom_triangle_alpha_max"] = 0.12

train = cfg.setdefault("training", {})
train["init_from_checkpoint"] = init_checkpoint
train["freeze_v11_base"] = True
train["delta_trainable_tokens"] = ["residue_atom_bias_"]
if use_triangle:
    train["delta_trainable_tokens"].append("residue_atom_triangle_")
train["learning_rate"] = 2.0e-4
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 2
train["num_epochs"] = 50
train["batch_size"] = 24
train["accumulation_steps"] = 1
train["early_stopping_patience"] = 50
train["num_workers"] = min(int(train.get("num_workers", 8)), 8)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V13m0 config: {out_path}")
print(f"variant={variant} coattn={is_coattn} triangle={use_triangle}")
print(f"trainable={train['delta_trainable_tokens']}")
PY

echo "Generated config:"
sed -n '1,260p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Training command: python scripts/train/train_v13m0.py --config ${CONFIG_FILE} --amp --accumulation-steps 1"
python scripts/train/train_v13m0.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1

TRAIN_EXIT=$?
if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "ERROR: V13m0 training failed with exit code ${TRAIN_EXIT}"
  exit "${TRAIN_EXIT}"
fi

RUN_DIR="$(ls -dt outputs/pocket_gvp_fusion_${VARIANT}_v13m0_* 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: could not locate output dir for ${VARIANT}"
  exit 5
fi

echo "Completed V13m0 run: ${RUN_DIR}"
echo "Best checkpoint: ${RUN_DIR}/best_f1_c0_model.pt"
echo "End time: $(date)"
