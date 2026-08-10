#!/usr/bin/env bash
# V11m16 deterministic candidate-pocket ranking probe.
#
# Purpose:
#   V11m13 already detects residue-level pockets and uses paired ligand chemistry,
#   but HOLO4K shows weak ligand-to-pocket assignment on multi-site proteins.
#   This probe forms four spatially separated, protein-only candidates from the
#   V4 pocketness channels, ranks them with a pre-cross ligand query, and applies
#   a bounded class-0 logit residual. The V11m13 trunk and residue head stay frozen.
#
# This is intentionally one task on one GPU. It can be submitted with sbatch or
# invoked directly on Star by scripts/run/run_v11m16_candidate_rank_gate_star.sh.

#SBATCH --job-name=v11m16_rankgate
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=20:00:00
#SBATCH --output=logs/v11m16_rankgate_%j.out
#SBATCH --error=logs/v11m16_rankgate_%j.err

set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-${PROJECT_DIR}}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
CONDA_ENV="${CONDA_ENV:-af3}"
BASELINE_NAME="${BASELINE_NAME:-pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114}"
BASE_CONFIG="${BASE_CONFIG:-${ARCHIVE_ROOT}/outputs/${BASELINE_NAME}/config.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-${ARCHIVE_ROOT}/outputs/${BASELINE_NAME}/best_f1_c0_model.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${ARCHIVE_ROOT}/outputs}"
RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
VARIANT="candidate_rankgate_k4_aux002"

mkdir -p "${ARCHIVE_ROOT}/logs" "${ARCHIVE_ROOT}/tmp/slurm_configs" "${OUTPUT_DIR}"
cd "${PROJECT_DIR}" || exit 1

if [ ! -f "${BASE_CONFIG}" ] || [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "ERROR: missing V11m13 baseline config/checkpoint"
  echo "  config=${BASE_CONFIG}"
  echo "  checkpoint=${INIT_CHECKPOINT}"
  exit 2
fi

export PATH="${CONDA_ROOT}/bin:${PATH}"
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
fi

CONFIG_FILE="${ARCHIVE_ROOT}/tmp/slurm_configs/v11m16_${VARIANT}_${RUN_ID}.yaml"
python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${INIT_CHECKPOINT}" "${OUTPUT_DIR}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path, out_path, checkpoint, output_dir = map(Path, sys.argv[1:])
with base_path.open() as handle:
    cfg = yaml.safe_load(handle)

cfg.setdefault("experiment", {})
cfg["experiment"].update(
    {
        "name": "pocket_gvp_fusion_seed44_candidate_rankgate_k4_aux002",
        "version_suffix": "v11m16",
        "seed": 44,
        "output_dir": str(output_dir),
    }
)

model = cfg["model"]
model.update(
    {
        "use_assignment_embed_loss": False,
        "assignment_samevalid_mode": "none",
        "use_candidate_rank_gate": True,
        "candidate_num_sites": 4,
        "candidate_min_center_distance": 10.0,
        "candidate_membership_sigma": 8.0,
        "candidate_embed_dim": 128,
        "candidate_temperature": 0.10,
        "candidate_gate_alpha_init": 0.02,
        "candidate_gate_alpha_max": 0.30,
        "candidate_aux_weight": 0.02,
        "candidate_aux_min_coverage": 0.05,
        "use_compile": False,
        "use_gradient_checkpointing": False,
    }
)

train = cfg.setdefault("training", {})
train.update(
    {
        "init_from_checkpoint": str(checkpoint),
        "freeze_v11_base": True,
        "delta_trainable_tokens": ["assignment_candidate_"],
        "learning_rate": 2.0e-4,
        "min_lr": 1.0e-6,
        "warmup_epochs": 2,
        "num_epochs": 40,
        "early_stopping_patience": 20,
        "batch_size": int(train.get("batch_size", 24)),
        "accumulation_steps": 1,
        "num_workers": min(int(train.get("num_workers", 8)), 8),
        "use_bf16": True,
        "use_amp": False,
    }
)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
print(out_path)
PY

echo "V11m16 config: ${CONFIG_FILE}"
sed -n '1,260p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python scripts/train/train_v11m16.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1

RUN_DIR="$(find "${OUTPUT_DIR}" -maxdepth 1 -type d -name '*candidate_rankgate_k4_aux002*v11m16*' -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
CKPT="${RUN_DIR}/best_f1_c0_model.pt"
if [ -z "${RUN_DIR}" ] || [ ! -f "${CKPT}" ]; then
  echo "ERROR: V11m16 best checkpoint not found under ${OUTPUT_DIR}"
  exit 3
fi

echo "V11m16 run complete: ${RUN_DIR}"
echo "Best checkpoint: ${CKPT}"

if [ "${RUN_LIGDEP:-1}" = "1" ]; then
  LIGDEP_OUT="${ARCHIVE_ROOT}/results/ligdep_v11m16_${RUN_ID}/test_all"
  mkdir -p "${LIGDEP_OUT}"
  python scripts/infer/infer_v11_internal_ligdep.py \
    --checkpoint "${CKPT}" \
    --config "${RUN_DIR}/config.yaml" \
    --split test \
    --ligand_mode all \
    --batch_size 8 \
    --num_workers 8 \
    --output "${LIGDEP_OUT}" \
    --amp
fi
