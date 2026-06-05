#!/bin/bash
# V11m8 all_light mild ligand-source dropout regularization probe.
#
# Purpose:
#   Refine the current all_light candidate with very small source-dropout
#   perturbations around the default all_light regularization.  This is not a
#   new architecture sweep; it tests whether slightly weaker/stronger ligand
#   source regularization improves paired-ligand real F1 while preserving the
#   real-shuffle gap.
#
# Baseline all_light:
#   ligand_source_dropout_p = 0.03
#   ligand_dropout_local54_p = 0.04
#   ligand_dropout_hidden_p = 0.04
#
# Matrix:
#   0. seed44_all_light_src02_g60 -> source=0.02, local/hidden=0.03
#   1. seed45_all_light_src02_g60 -> source=0.02, local/hidden=0.03
#   2. seed44_all_light_src04_g60 -> source=0.04, local/hidden=0.05
#   3. seed45_all_light_src04_g60 -> source=0.04, local/hidden=0.05

#SBATCH --job-name=v11m8_src_reg
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/v11m8_src_reg_%A_%a.out
#SBATCH --error=logs/v11m8_src_reg_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=(
  "seed44_all_light_src02_g60"
  "seed45_all_light_src02_g60"
  "seed44_all_light_src04_g60"
  "seed45_all_light_src04_g60"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m8 source-reg task id: ${TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m8_src_reg_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

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

if "_src02_" in variant:
    source_drop = 0.02
    branch_drop = 0.03
elif "_src04_" in variant:
    source_drop = 0.04
    branch_drop = 0.05
else:
    raise SystemExit(f"Unknown source-reg variant: {variant}")

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_v11m8_{variant}"
cfg["experiment"]["version_suffix"] = "v11m8"
cfg["experiment"]["seed"] = seed

model = cfg["model"]
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = True

model["use_local_ligand_film"] = True
model["use_pocket_virtual_tokens"] = True
model["use_center_aux_head"] = True
model["ligand_source_dropout_p"] = source_drop
model["ligand_dropout_local54_p"] = branch_drop
model["ligand_dropout_hidden_p"] = branch_drop

model["local_ligand_film_hidden"] = 256
model["local_ligand_film_alpha_init"] = 0.05
model["local_ligand_film_alpha_max"] = 0.30
model["pocket_virtual_tokens"] = 4
model["pocket_virtual_alpha_init"] = 0.05
model["pocket_virtual_alpha_max"] = 0.30
model["center_aux_hidden"] = 128
model["center_aux_logit_scale_init"] = 0.05
model["center_aux_logit_scale_max"] = 0.30

cfg.setdefault("training", {})
cfg["training"]["init_from_checkpoint"] = None
cfg["training"]["freeze_v11_base"] = False
cfg["training"]["delta_trainable_tokens"] = [
    "local_delta_",
    "global_delta_gate.",
    "local_ligand_film",
    "pocket_",
    "center_aux_",
]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V11m8 source-reg config: {out_path}")
print(
    "variant={variant} seed={seed} source_drop={source} branch_drop={branch} "
    "grad_scale={scale}".format(
        variant=variant,
        seed=seed,
        source=source_drop,
        branch=branch_drop,
        scale=model["local_delta_protein_grad_scale"],
    )
)
PY

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(python scripts/train/train_v11m8.py --config "${CONFIG_FILE}" --amp --gradient-checkpointing)
if [ -n "${RESUME_CHECKPOINT}" ]; then
  CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "Generated config:"
sed -n '1,260p' "${CONFIG_FILE}"
echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
