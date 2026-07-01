#!/bin/bash
# V11m13 ESM2-MLC full retrain probes, not warm-start/resume.
#
# Motivation:
#   The warm-start MLC probes produced a small val-F1 gain but kept ~90-97% of
#   layer mass on the final ESM2 layer because they were initialized from the
#   existing V11m13 basin.  These probes test whether MLC becomes more useful
#   when trained end-to-end from scratch with uniform layer initialization.
#
# Variants:
#   0 esmmlc_scratch_wsum_uniform
#   1 esmmlc_scratch_concat_uniform

#SBATCH --job-name=mlc_scratch
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/mlc_scratch_%A_%a.out
#SBATCH --error=logs/mlc_scratch_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
ESM_FEATURES_DIR="${ESM_FEATURES_DIR:-esm2_multilayer_latents}"
ESM_MLC_LAYERS="${ESM_MLC_LAYERS:-2,6,12,18,24,30,33}"
BATCH_SIZE="${BATCH_SIZE:-20}"
ACCUMULATION_STEPS="${ACCUMULATION_STEPS:-1}"

VARIANTS=(
  "esmmlc_scratch_wsum_uniform"
  "esmmlc_scratch_concat_uniform"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid MLC scratch task id=${TASK_ID}"
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
echo "Base config: ${BASE_CONFIG}"
echo "ESM features: data/${ESM_FEATURES_DIR}"
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
if [ ! -d "data/${ESM_FEATURES_DIR}" ]; then
  echo "ERROR: ESM multi-layer feature dir not found: data/${ESM_FEATURES_DIR}"
  exit 4
fi

mkdir -p tmp/slurm_configs
CONFIG_FILE="tmp/slurm_configs/v11m13_esmmlc_scratch_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${ESM_FEATURES_DIR}" "${ESM_MLC_LAYERS}" "${BATCH_SIZE}" "${ACCUMULATION_STEPS}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
esm_features_dir = sys.argv[4]
layers = [int(x) for x in sys.argv[5].split(",") if x.strip()]
batch_size = int(sys.argv[6])
accumulation_steps = int(sys.argv[7])

with base_path.open() as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_seed44_{variant}"
cfg["experiment"]["version_suffix"] = "v11m13"
cfg["experiment"]["seed"] = 44

data = cfg.setdefault("data", {})
data["esm_features_dir"] = esm_features_dir
data["max_atoms"] = int(data.get("max_atoms", 128))
data["atom_truncation_policy"] = data.get("atom_truncation_policy", "alternating_ends")

model = cfg["model"]
model["use_esm_mlc"] = True
model["esm_mlc_num_layers"] = len(layers)
model["esm_mlc_layers"] = layers
model["esm_mlc_init"] = "uniform"
model["esm_mlc_dropout"] = 0.05
model["esm_mlc_fusion"] = "concat_proj" if "concat" in variant else "weighted_sum"
model["esm_mlc_concat_zero_init"] = True

# Current V11m13 group_rescompat recipe, trained end-to-end from scratch.
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

train = cfg.setdefault("training", {})
train["init_from_checkpoint"] = None
train["freeze_v11_base"] = False
train["learning_rate"] = 5.0e-5
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 3
train["num_epochs"] = 100
train["batch_size"] = batch_size
train["accumulation_steps"] = accumulation_steps
train["early_stopping_patience"] = 100
train["num_workers"] = min(int(train.get("num_workers", 8)), 8)
train["use_amp"] = True

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote MLC scratch config: {out_path}")
print(
    f"variant={variant} fusion={model['esm_mlc_fusion']} layers={layers} "
    f"batch={batch_size} accum={accumulation_steps}"
)
PY

echo "Generated config:"
sed -n '1,260p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python scripts/train/train_v11m13.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps "${ACCUMULATION_STEPS}"

echo "Done: ${VARIANT} at $(date)"
