#!/bin/bash
# V13 warm-start adapter/value-enrichment probes.
#
# Motivation:
#   V13 scratch/full-train has not yet recovered the V11m13 group_rescompat
#   basin.  This job tests V13 ideas under the same low-risk recipe that made
#   V11m13 strong: warm-start from group_rescompat seed44, freeze the base, and
#   train only small zero-initialized adapters.
#
# Variants:
#   0 v13m0_resatom_adapter_only
#       V11m13 group_rescompat + residue_atom_bias, train only bias module.
#   1 v13m0_resatom_adapter_head
#       Same, plus local_delta_head adaptation at lower LR.
#   2 v13m0_value_enrich_only
#       V11m13 group_rescompat + post-attention local54 value enrichment.
#   3 v13m0_value_enrich_head
#       Same, plus local_delta_head adaptation at lower LR.

#SBATCH --job-name=v13m0_wv
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=18:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/v13m0_wv_%A_%a.out
#SBATCH --error=logs/v13m0_wv_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BASE_CONFIG="${BASE_CONFIG:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt}"

VARIANTS=(
  "v13m0_resatom_adapter_only"
  "v13m0_resatom_adapter_head"
  "v13m0_value_enrich_only"
  "v13m0_value_enrich_head"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V13 warm-start task id: ${TASK_ID}"
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
echo "Init checkpoint: ${INIT_CHECKPOINT}"
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
if [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "ERROR: init checkpoint not found: ${INIT_CHECKPOINT}"
  exit 4
fi

echo "Python: $(which python)"
python --version

mkdir -p tmp/slurm_configs
CONFIG_FILE="tmp/slurm_configs/v13m0_wv_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

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

use_resatom = "resatom" in variant
use_value = "value_enrich" in variant
train_head = variant.endswith("_head")

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_seed44_{variant}"
cfg["experiment"]["version_suffix"] = "v13m0"
cfg["experiment"]["seed"] = 44

data = cfg.setdefault("data", {})
data["max_atoms"] = int(data.get("max_atoms", 128))
data["atom_truncation_policy"] = data.get("atom_truncation_policy", "alternating_ends")

model = cfg["model"]

# Preserve V11m13 group_rescompat behavior exactly, then add one V13 probe path.
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False
model["use_v12_cross_attention"] = False
model["use_v12_coattention_block"] = True
model["use_local54_interaction_kv"] = True
model["local54_interaction_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
model["local54_interaction_hidden"] = 96
model["local54_interaction_kv_alpha_init"] = 0.035
model["local54_interaction_kv_alpha_max"] = 0.20
model["use_local54_residue_chem_compat"] = True
model["local54_compat_hidden"] = 128
model["local54_compat_gate_bias_init"] = 0.01
model["local54_compat_alpha_init"] = 0.02
model["local54_compat_alpha_max"] = 0.12
model["local54_compat_normalize_read"] = True
model["local54_compat_read_scale"] = 0.35
model["local54_compat_read_clip"] = 0.0
model["use_local54_edge_context_kv"] = False
model["use_local54_chem_adapter"] = False
model["use_local54_chem_reweight"] = False
model["use_ligand_context_prior"] = False
model["ligand_context_prior_sources"] = []
model["use_ligand_slots_delta"] = False
model["use_local54_qk_reparam"] = False
model["use_residue_atom_triangle_bias"] = False

model["use_residue_atom_bias"] = use_resatom
model["residue_atom_bias_hidden"] = 64
model["residue_atom_bias_local54_groups"] = [
    "base33",
    "electronic3",
    "sasa_hbond4",
    "flex_rot5",
]
model["residue_atom_bias_alpha_init"] = 0.03
model["residue_atom_bias_alpha_max"] = 0.18
model["residue_atom_bias_clip"] = 6.0

model["use_residue_chem_value_enrich"] = use_value
model["residue_chem_value_enrich_groups"] = [
    "base33",
    "electronic3",
    "sasa_hbond4",
    "flex_rot5",
]
model["residue_chem_value_enrich_hidden"] = 128
model["residue_chem_value_enrich_gate_bias_init"] = 0.01
model["residue_chem_value_enrich_alpha_init"] = 0.02
model["residue_chem_value_enrich_alpha_max"] = 0.12
model["residue_chem_value_enrich_normalize"] = True
model["residue_chem_value_enrich_scale"] = 0.35

train = cfg.setdefault("training", {})
train["init_from_checkpoint"] = init_checkpoint
train["freeze_v11_base"] = True
tokens = []
if use_resatom:
    tokens.append("residue_atom_bias_")
if use_value:
    tokens.append("residue_chem_value_enrich_")
if train_head:
    tokens.append("local_delta_head")
train["delta_trainable_tokens"] = tokens
train["learning_rate"] = 5.0e-5 if train_head else 2.0e-4
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 2
train["num_epochs"] = 60
train["batch_size"] = 24
train["accumulation_steps"] = 1
train["early_stopping_patience"] = 60
train["num_workers"] = min(int(train.get("num_workers", 8)), 8)
train["use_amp"] = True

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V13 warm-start config: {out_path}")
print(
    f"variant={variant} use_resatom={use_resatom} use_value={use_value} "
    f"train_head={train_head} lr={train['learning_rate']} tokens={tokens}"
)
PY

echo "Generated config:"
sed -n '1,280p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

echo "Training command: python scripts/train/train_v13m0.py --config ${CONFIG_FILE} --amp --accumulation-steps 1"
python scripts/train/train_v13m0.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1

TRAIN_EXIT=$?
if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "ERROR: V13 warm-start probe failed with exit code ${TRAIN_EXIT}"
  exit "${TRAIN_EXIT}"
fi

RUN_DIR="$(ls -dt outputs/pocket_gvp_fusion_seed44_${VARIANT}_v13m0_* 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: could not locate output dir for ${VARIANT}"
  exit 5
fi

echo "Completed V13 warm-start run: ${RUN_DIR}"
echo "Best checkpoint: ${RUN_DIR}/best_f1_c0_model.pt"
echo "End time: $(date)"
