#!/bin/bash
# V11m13 ESM-C 600M replacement/fusion probes.
#
# Purpose:
#   Replace the ESM2-650M final residue cache with exact-sequence ESM-C 600M
#   block 12/24/36 features while preserving the proven V11m13
#   group_rescompat + co-attention + local54/graph architecture.  This array
#   separates ESM-C layer fusion from the optional weak ESM3 global prior.
#
# Array mapping:
#   0 lastinit_wsum       learned weighted sum, initialized near block 36
#   1 uniform_wsum        learned weighted sum, uniform 12/24/36 initialization
#   2 concat              trainable concat projection, standard initialization
#   3 lastinit_wsum_esm3  task 0 plus the existing weak ESM3 global prior
#   4 lastinit_wsum_warmft initialize all shape-compatible V11m13 weights, then
#                          full-finetune (the ESM input projection is new)
#   5 lastinit_wsum_warmadapter
#                          initialize V11m13 and train only esm_proj + ESM-C MLC
#
# Promotion rule:
#   Compare best-AUPRC and best-F1 checkpoints on test real/shuffle/zero.
#   ESM-C is promoted only if real F1 improves without reducing real-shuffle.

#SBATCH --job-name=v11m13_esmc
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/v11m13_esmc_%A_%a.out
#SBATCH --error=logs/v11m13_esmc_%A_%a.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/hpc2hdd/home/shuang886/PLC}"
BASE_CONFIG="${BASE_CONFIG:-${ARCHIVE_ROOT}/outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARCHIVE_ROOT}/outputs}"
ESMC_FEATURES_DIR="${ESMC_FEATURES_DIR:-esmc_600m_2024_12_mlc_l12_l24_l36_layered_fp16}"
ESM3_FEATURES_DIR="${ESM3_FEATURES_DIR:-esm3_latents_norm}"
BATCH_SIZE="${BATCH_SIZE:-16}"

VARIANTS=(
  "lastinit_wsum"
  "uniform_wsum"
  "concat"
  "lastinit_wsum_esm3"
  "lastinit_wsum_warmft"
  "lastinit_wsum_warmadapter"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid task id=${TASK_ID}"
  exit 2
fi

mkdir -p "${PROJECT_DIR}/logs" "${PROJECT_DIR}/tmp/slurm_configs" "${OUTPUT_DIR}"
cd "${PROJECT_DIR}" || exit 1

echo "============================================"
echo "Experiment: ESM-C 600M PDBbind probes"
echo "Variant: ${VARIANT}"
echo "Job: ${SLURM_JOB_ID}_${TASK_ID} node=${SLURMD_NODENAME}"
echo "ESMC cache: data/${ESMC_FEATURES_DIR}"
echo "Base config: ${BASE_CONFIG}"
echo "Start: $(date)"
echo "============================================"

if command -v module &> /dev/null; then
  module purge || true
fi
if command -v conda &> /dev/null; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "ERROR: conda command not found"
  exit 1
fi
conda activate af3

if [ ! -f "${BASE_CONFIG}" ]; then
  echo "ERROR: base config not found: ${BASE_CONFIG}"
  exit 3
fi
if [ ! -f "data/${ESMC_FEATURES_DIR}/pdb_manifest.tsv" ]; then
  echo "ERROR: ESM-C manifest not found: data/${ESMC_FEATURES_DIR}/pdb_manifest.tsv"
  exit 4
fi

CONFIG_FILE="tmp/slurm_configs/v11m13_esmc_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${ESMC_FEATURES_DIR}" \
  "${ESM3_FEATURES_DIR}" "${BATCH_SIZE}" "${OUTPUT_DIR}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
esmc_features_dir = sys.argv[4]
esm3_features_dir = sys.argv[5]
batch_size = int(sys.argv[6])
output_dir = sys.argv[7]

with base_path.open() as handle:
    cfg = yaml.safe_load(handle)

use_esm3 = variant.endswith("_esm3")
is_warmft = variant.endswith("_warmft")
is_warmadapter = variant.endswith("_warmadapter")
fusion = "concat_proj" if variant.startswith("concat") else "weighted_sum"
init = "uniform" if variant.startswith("uniform_wsum") else "last"

experiment = cfg.setdefault("experiment", {})
experiment["name"] = f"pocket_gvp_fusion_seed44_esmc_{variant}"
experiment["version_suffix"] = "v11m13esmc"
experiment["seed"] = 44
experiment["output_dir"] = output_dir

data = cfg.setdefault("data", {})
data["esm_features_dir"] = esmc_features_dir
data["esm_alignment_features_dir"] = "esm2_latents" if use_esm3 else None
data["esm3_features_dir"] = esm3_features_dir if use_esm3 else None
data["max_atoms"] = int(data.get("max_atoms", 128))
data["atom_truncation_policy"] = data.get("atom_truncation_policy", "alternating_ends")

model = cfg["model"]
model["d_esm"] = 1152
model["use_esm_mlc"] = True
model["esm_mlc_num_layers"] = 3
model["esm_mlc_layers"] = [12, 24, 36]
model["esm_mlc_fusion"] = fusion
model["esm_mlc_init"] = init
model["esm_mlc_last_logit_bias"] = 4.0
model["esm_mlc_dropout"] = 0.0
# The concat branch is residual over the weighted sum. Zero-init starts from
# the stable last-layer-weighted baseline and learns only an incremental path.
model["esm_mlc_concat_zero_init"] = True

model["use_esm3"] = use_esm3
model["d_esm3"] = 1536
model["esm3_fusion"] = "global_prior"
model["esm3_alpha_init"] = 0.02
model["esm3_alpha_max"] = 0.20
model["esm3_gate_bias_init"] = 1.5
model["esm3_zero_init"] = True
model["esm3_dropout_prob"] = 0.0

# Exact currently promoted V11m13 local-evidence architecture.
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["local_delta_trainable_logit_scale"] = False
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False
model["use_compile"] = False
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

training = cfg.setdefault("training", {})
training["init_from_checkpoint"] = (
    str(base_path.with_name("best_f1_c0_model.pt"))
    if (is_warmft or is_warmadapter)
    else None
)
training["freeze_v11_base"] = is_warmadapter
if is_warmadapter:
    # Learn a supervised ESM-C-to-established-V11m13 representation adapter
    # without allowing the proven protein/ligand decision path to drift.
    training["delta_trainable_tokens"] = ["esm_proj.", "esm_mlc_"]
training["learning_rate"] = 1.0e-4 if is_warmft else 2.0e-4
training["min_lr"] = 1.0e-6
training["warmup_epochs"] = 2 if (is_warmft or is_warmadapter) else 3
training["num_epochs"] = 60 if (is_warmft or is_warmadapter) else 80
training["batch_size"] = batch_size
training["accumulation_steps"] = 1
training["early_stopping_patience"] = 40
training["num_workers"] = min(int(training.get("num_workers", 8)), 8)
training["use_amp"] = True
training["save_every"] = 1000

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

print(
    f"wrote={out_path} variant={variant} fusion={fusion} init={init} "
    f"use_esm3={use_esm3} warmft={is_warmft} "
    f"warmadapter={is_warmadapter} d_esm={model['d_esm']} batch={batch_size}"
)
PY

sed -n '1,300p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

python scripts/train/train_v11m13.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1

echo "Done: ${VARIANT} at $(date)"
