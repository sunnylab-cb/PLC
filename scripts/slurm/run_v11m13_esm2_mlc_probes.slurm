#!/bin/bash
# V11m13 ESM2 multi-layer concat/fusion probes.
#
# Prerequisite:
#   Extract ESM2 multi-layer features to data/${ESM_FEATURES_DIR}, with
#   embeddings shaped [N, 7, 1280] and repr_layers=[2,6,12,18,24,30,33].
#
# Local/remote extraction command template:
#   python preparation/protein/esm2_extract_latents_loose.py \
#     --input <pdb_root> \
#     --pattern '*_protein.pdb' \
#     --output data/esm2_multilayer_latents \
#     --model esm2_t33_650M_UR50D \
#     --repr-layers 2,6,12,18,24,30,33 \
#     --devices 0,1
#
# Variants:
#   0 esmmlc_wsum_only
#       Warm-start V11m13 group_rescompat, freeze base, train only MLC weights.
#   1 esmmlc_concat_only
#       Same, but concat selected layers -> projection + residual weighted sum.
#   2 esmmlc_concat_esmproj
#       Same as concat, but also unfreeze esm_proj to adapt the old ESM path.

#SBATCH --job-name=v11m13_mlc
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/v11m13_mlc_%A_%a.out
#SBATCH --error=logs/v11m13_mlc_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BASE_CONFIG="${BASE_CONFIG:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt}"
ESM_FEATURES_DIR="${ESM_FEATURES_DIR:-esm2_multilayer_latents}"
ESM_MLC_LAYERS="${ESM_MLC_LAYERS:-2,6,12,18,24,30,33}"

VARIANTS=(
  "esmmlc_wsum_only"
  "esmmlc_concat_only"
  "esmmlc_concat_esmproj"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid ESM MLC task id: ${TASK_ID}"
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
echo "ESM features: data/${ESM_FEATURES_DIR}"
echo "ESM MLC layers: ${ESM_MLC_LAYERS}"
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
if [ ! -d "data/${ESM_FEATURES_DIR}" ]; then
  echo "ERROR: ESM multi-layer feature dir not found: data/${ESM_FEATURES_DIR}"
  echo "Run the extraction command in this script header before submitting."
  exit 5
fi

echo "Python: $(which python)"
python --version

mkdir -p tmp/slurm_configs
CONFIG_FILE="tmp/slurm_configs/v11m13_esmmlc_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${INIT_CHECKPOINT}" "${ESM_FEATURES_DIR}" "${ESM_MLC_LAYERS}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
init_checkpoint = sys.argv[4]
esm_features_dir = sys.argv[5]
layers = [int(x) for x in sys.argv[6].split(",") if x.strip()]

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
model["esm_mlc_init"] = "last"
model["esm_mlc_last_logit_bias"] = 4.0
model["esm_mlc_dropout"] = 0.05
model["esm_mlc_fusion"] = "concat_proj" if "concat" in variant else "weighted_sum"
model["esm_mlc_concat_zero_init"] = True

# Preserve the current V11m13 group_rescompat recipe.
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

train = cfg.setdefault("training", {})
train["init_from_checkpoint"] = init_checkpoint
train["freeze_v11_base"] = True
tokens = ["esm_mlc_"]
if variant.endswith("_esmproj"):
    tokens.append("esm_proj")
train["delta_trainable_tokens"] = tokens
train["learning_rate"] = 1.0e-4 if "esmproj" in variant else 2.0e-4
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 2
train["num_epochs"] = 60
train["batch_size"] = 20
train["accumulation_steps"] = 1
train["early_stopping_patience"] = 60
train["num_workers"] = min(int(train.get("num_workers", 8)), 8)
train["use_amp"] = True

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote ESM MLC config: {out_path}")
print(
    f"variant={variant} fusion={model['esm_mlc_fusion']} layers={layers} "
    f"train_tokens={tokens} lr={train['learning_rate']}"
)
PY

echo "Generated config:"
sed -n '1,260p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

echo "Training command: python scripts/train/train_v11m13.py --config ${CONFIG_FILE} --amp --accumulation-steps 1"
python scripts/train/train_v11m13.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1

echo "Done: ${VARIANT} at $(date)"
