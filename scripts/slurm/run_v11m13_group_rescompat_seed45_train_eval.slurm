#!/bin/bash
# V11m13 original group_rescompat seed45 confirmation run.
#
# Motivation:
#   Keep the original group_rescompat_lr2e4 design as the current main line.
#   The seed44 result is the current strongest single-seed test model:
#     test real F1_c0 ~= 0.6158, real-shuffle ~= 0.080.
#   The base4 direct-base33 follow-up did not improve val, so the next useful
#   supplement is not another architecture tweak, but a matched seed45
#   confirmation from the existing seed45 V11m8x coattn checkpoint.
#
# This job trains:
#   seed45_group_rescompat_lr2e4_v11m13
# and then immediately runs test ligand-dependency:
#   all, source_split, feature_split

#SBATCH --job-name=v11m13_grc45
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/v11m13_grc45_%j.out
#SBATCH --error=logs/v11m13_grc45_%j.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/pocket_gvp_fusion_seed45_coattn_g60_v11m8x_20260606_200217/best_f1_c0_model.pt}"
VARIANT="seed45_group_rescompat_lr2e4"
OUT_ROOT="${OUT_ROOT:-results/ligdep_v11m13_group_rescompat_seed45_${SLURM_JOB_ID:-manual}}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Variant: ${VARIANT}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Base config: ${BASE_CONFIG}"
echo "Init checkpoint: ${INIT_CHECKPOINT}"
echo "Eval output root: ${OUT_ROOT}"
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
CONFIG_FILE="tmp/slurm_configs/v11m13_${VARIANT}_${SLURM_JOB_ID}.yaml"

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

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_{variant}"
cfg["experiment"]["version_suffix"] = "v11m13"
cfg["experiment"]["seed"] = 45

model = cfg["model"]

# Original group_rescompat design: base33 remains the anchor; condition groups
# are electronic/sasa/flex only.  Do not include base33 as a direct group.
model["local54_interaction_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
model["local54_interaction_hidden"] = 96
model["use_local54_interaction_kv"] = True
model["use_local54_residue_chem_compat"] = True
model["use_local54_edge_context_kv"] = False
model["use_local54_chem_adapter"] = False
model["use_local54_chem_reweight"] = False

# Preserve current co-attn/no-gate/grad060 base behavior.
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False
model["use_v12_cross_attention"] = False
model["use_v12_coattention_block"] = True

model["local54_interaction_kv_alpha_init"] = 0.035
model["local54_interaction_kv_alpha_max"] = 0.20
model["local54_compat_hidden"] = 128
model["local54_compat_gate_bias_init"] = 0.01
model["local54_compat_alpha_init"] = 0.02
model["local54_compat_alpha_max"] = 0.12
model["local54_compat_normalize_read"] = True
model["local54_compat_read_scale"] = 0.35
model["local54_compat_read_clip"] = 0.0
model["use_ligand_context_prior"] = False
model["ligand_context_prior_sources"] = []

cfg.setdefault("training", {})
train = cfg["training"]
train["init_from_checkpoint"] = init_checkpoint
train["freeze_v11_base"] = True
train["delta_trainable_tokens"] = [
    "local54_anchor_proj.",
    "local54_group_proj.",
    "local54_interaction_",
    "local54_compat_",
]
train["learning_rate"] = 2.0e-4
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 2
train["num_epochs"] = 60
train["batch_size"] = 24
train["accumulation_steps"] = 1
train["early_stopping_patience"] = 60

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print("Wrote V11m13 group_rescompat seed45 config:", out_path)
print(f"variant={variant} seed=45 lr=2e-4 init={init_checkpoint}")
PY

echo "Generated config:"
sed -n '1,240p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Training command: python scripts/train/train_v11m13.py --config ${CONFIG_FILE} --amp --accumulation-steps 1"
python scripts/train/train_v11m13.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1
TRAIN_EXIT=$?
if [ "${TRAIN_EXIT}" -ne 0 ]; then
  echo "ERROR: training failed with exit code ${TRAIN_EXIT}"
  exit "${TRAIN_EXIT}"
fi

RUN_DIR="$(ls -dt outputs/pocket_gvp_fusion_${VARIANT}_v11m13_* 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: could not locate output dir for ${VARIANT}"
  exit 5
fi
CKPT="${RUN_DIR}/best_f1_c0_model.pt"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: best checkpoint not found after training: ${CKPT}"
  exit 6
fi

echo "Training output dir: ${RUN_DIR}"
echo "Best checkpoint: ${CKPT}"

for MODE in all source_split feature_split; do
  OUT_DIR="${OUT_ROOT}/${VARIANT}_test_${MODE}"
  mkdir -p "${OUT_DIR}"
  CMD=(
    python scripts/infer/infer_v11_internal_ligdep.py
    --checkpoint "${CKPT}"
    --split test
    --ligand_mode "${MODE}"
    --output "${OUT_DIR}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --max_samples "${MAX_SAMPLES}"
    --num_save_rows 1
    --amp
  )
  echo "Eval command: ${CMD[*]}"
  "${CMD[@]}"
  EVAL_EXIT=$?
  if [ "${EVAL_EXIT}" -ne 0 ]; then
    echo "ERROR: eval mode ${MODE} failed with exit code ${EVAL_EXIT}"
    exit "${EVAL_EXIT}"
  fi
done

echo "Finished at: $(date)"
