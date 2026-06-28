#!/bin/bash
# V13 parallel structural probes.
#
# These are sibling V13 branches to run alongside V13m0 scratch/full-train:
#
#   V13m1 slots:
#     Compress ligand atom/local54 evidence into K learned ligand slots, then
#     let residues attend to slots instead of raw atoms.  This is the
#     O(L_res x K) bottleneck line and should be more memory/generalization
#     friendly than full residue-atom attention.
#
#   V13m2 QK reparameterization:
#     Keep atom-level local_delta, but use ligand/local54 context to shift the
#     residue query and atom key spaces.  This tests "attention geometry warp"
#     rather than scalar score bias.
#
# Variants:
#   0 coattn_slots16_b16_acc2_h128
#   1 noncoattn_slots16_b16_acc2_h128
#   2 coattn_qkreparam_b16_acc2_h96
#   3 noncoattn_qkreparam_b16_acc2_h96

#SBATCH --job-name=v13m12_pr
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/v13m12_pr_%A_%a.out
#SBATCH --error=logs/v13m12_pr_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-2}"

VARIANTS=(
  "coattn_slots16_b16_acc2_h128"
  "noncoattn_slots16_b16_acc2_h128"
  "coattn_qkreparam_b16_acc2_h96"
  "noncoattn_qkreparam_b16_acc2_h96"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V13 parallel task id: ${TASK_ID}"
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
echo "Base config: ${BASE_CONFIG}"
echo "Batch size: ${BATCH_SIZE}"
echo "Accumulation steps: ${ACCUMULATION_STEPS}"
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

echo "Python: $(which python)"
python --version

mkdir -p tmp/slurm_configs
CONFIG_FILE="tmp/slurm_configs/v13m12_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${BATCH_SIZE}" "${ACCUMULATION_STEPS}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
batch_size = int(sys.argv[4])
accumulation_steps = int(sys.argv[5])

with base_path.open() as f:
    cfg = yaml.safe_load(f)

is_coattn = variant.startswith("coattn_")
is_slots = "slots" in variant
is_qk = "qkreparam" in variant
suffix = "v13m1" if is_slots else "v13m2"

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_{variant}"
cfg["experiment"]["version_suffix"] = suffix
cfg["experiment"]["seed"] = 44

data = cfg.setdefault("data", {})
data["max_atoms"] = int(data.get("max_atoms", 128))
data["atom_truncation_policy"] = data.get("atom_truncation_policy", "alternating_ends")

model = cfg["model"]
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False
model["local_delta_use_qk_norm"] = bool(model.get("local_delta_use_qk_norm", False))

model["use_v12_cross_attention"] = False
model["use_v12_coattention_block"] = is_coattn
if is_coattn:
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

model["use_ligand_context_prior"] = False
model["ligand_context_prior_sources"] = []
model["use_local54_edge_context_kv"] = False
model["use_local54_chem_adapter"] = False
model["use_local54_chem_reweight"] = False
model["use_residue_atom_bias"] = False
model["use_residue_atom_triangle_bias"] = False

if is_slots:
    model["use_ligand_slots_delta"] = True
    model["ligand_slots_num"] = 16
    model["ligand_slots_hidden"] = 128
    model["ligand_slots_local54_groups"] = [
        "base33",
        "electronic3",
        "sasa_hbond4",
        "flex_rot5",
    ]
    model["use_local54_qk_reparam"] = False
    # Slots replace atom-level local_delta; keep co-attn only as a mid-level
    # protein-ligand trunk option, not as local54 group K/V/compat.
    model["use_local54_interaction_kv"] = False
    model["use_local54_residue_chem_compat"] = False
else:
    model["use_ligand_slots_delta"] = False
    model["use_local54_qk_reparam"] = True
    model["local54_qk_reparam_hidden"] = 96
    model["local54_qk_reparam_alpha_init"] = 0.03
    model["local54_qk_reparam_alpha_max"] = 0.20
    model["local54_qk_reparam_local54_groups"] = [
        "base33",
        "electronic3",
        "sasa_hbond4",
        "flex_rot5",
    ]
    if is_coattn:
        model["use_local54_interaction_kv"] = True
        model["local54_interaction_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
        model["local54_interaction_hidden"] = 128
        model["local54_interaction_kv_alpha_init"] = 0.035
        model["local54_interaction_kv_alpha_max"] = 0.20
        model["use_local54_residue_chem_compat"] = True
        model["local54_compat_hidden"] = 160
        model["local54_compat_gate_bias_init"] = 0.01
        model["local54_compat_alpha_init"] = 0.02
        model["local54_compat_alpha_max"] = 0.12
        model["local54_compat_normalize_read"] = True
        model["local54_compat_read_scale"] = 0.35
        model["local54_compat_read_clip"] = 0.0
    else:
        model["use_local54_interaction_kv"] = False
        model["use_local54_residue_chem_compat"] = False

train = cfg.setdefault("training", {})
train["init_from_checkpoint"] = None
train["init_from_v11m3"] = None
train["freeze_v11_base"] = False
train["batch_size"] = batch_size
train["accumulation_steps"] = accumulation_steps
train["learning_rate"] = 5.0e-5
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 3
train["num_epochs"] = 100
train["early_stopping_patience"] = 100
train["num_workers"] = min(int(train.get("num_workers", 8)), 8)
train["use_amp"] = True

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V13 parallel config: {out_path}")
print(f"variant={variant} suffix={suffix} coattn={is_coattn} slots={is_slots} qk={is_qk}")
print(f"batch_size={batch_size} accumulation_steps={accumulation_steps}")
PY

echo "Generated config:"
sed -n '1,280p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

echo "Training command: python scripts/train/train_v13m0.py --config ${CONFIG_FILE} --amp --accumulation-steps ${ACCUMULATION_STEPS}"
python scripts/train/train_v13m0.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps "${ACCUMULATION_STEPS}"

TRAIN_EXIT=$?
if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "ERROR: V13 parallel probe failed with exit code ${TRAIN_EXIT}"
  exit "${TRAIN_EXIT}"
fi

RUN_DIR="$(ls -dt outputs/pocket_gvp_fusion_${VARIANT}_${suffix}_* 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: could not locate output dir for ${VARIANT}"
  exit 5
fi

echo "Completed V13 parallel run: ${RUN_DIR}"
echo "Best checkpoint: ${RUN_DIR}/best_f1_c0_model.pt"
echo "End time: $(date)"
