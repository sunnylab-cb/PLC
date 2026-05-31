#!/bin/bash
# V11m7 local54 group-wise adapter pilot.
#
# What this runs:
#   V11m6n grad060/no_gate mainline plus a small local54 group adapter:
#     base33        -> medium capacity
#     electronic3   -> high capacity
#     sasa_hbond4   -> highest capacity
#     flex_rot5     -> high capacity
#     metal4/stereo5-> small capacity
#
# Why:
#   The subgroup ablation showed full local54 is important, with strongest
#   single-group drops from SASA/H-bond and flexibility/rotatable features.
#   Edge attrs/topology alone had ~0 drop, so this run strengthens chemistry
#   representation without adding UniMol/global gates or larger fusion.

#SBATCH --job-name=v11m7_l54ad
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/v11m7_l54ad_%A_%a.out
#SBATCH --error=logs/v11m7_l54ad_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=("seed42_g60" "seed45_g60")
SEEDS=(42 45)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANT:-${VARIANTS[$TASK_ID]}}"
SEED="${SEED:-${SEEDS[$TASK_ID]}}"

if [ -z "${VARIANT}" ] || [ -z "${SEED}" ]; then
  echo "ERROR: invalid V11m7 task id: ${TASK_ID}"
  exit 2
fi

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Variant: ${VARIANT}"
echo "Seed: ${SEED}"
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
CONFIG_FILE="tmp/slurm_configs/v11m7_local54_adapter_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${SEED}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
seed = int(sys.argv[4])

with base_path.open() as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_v11m7_local54_adapter_{variant}"
cfg["experiment"]["version_suffix"] = "v11m7"
cfg["experiment"]["seed"] = seed

model = cfg["model"]
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_local54_group_adapter"] = True
model["local54_group_adapter_dim"] = 128
model["local54_group_dropout"] = 0.05
model["local54_group_dim_base33"] = 48
model["local54_group_dim_electronic3"] = 64
model["local54_group_dim_sasa_hbond4"] = 96
model["local54_group_dim_flex_rot5"] = 80
model["local54_group_dim_metal4"] = 16
model["local54_group_dim_stereo5"] = 16

cfg.setdefault("training", {})
cfg["training"]["init_from_checkpoint"] = None
cfg["training"]["freeze_v11_base"] = False
cfg["training"]["delta_trainable_tokens"] = [
    "local_delta_",
    "local54_group_adapter.",
    "global_delta_gate.",
]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Wrote V11m7 config: {out_path}")
PY

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(python scripts/train/train_v11m7.py --config "${CONFIG_FILE}" --amp)
if [ -n "${RESUME_CHECKPOINT}" ]; then
  CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "Generated config:"
sed -n '1,230p' "${CONFIG_FILE}"
echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
