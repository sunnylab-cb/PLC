#!/bin/bash
# V11m6n grad060/no_gate + very mild ligand-source dropout probe.
#
# Purpose:
#   Test whether tiny training-time dropout on the local_delta ligand sources
#   can reduce zero-ligand fragility while preserving the established
#   real-shuffle gap around ~0.08.
#
# Design:
#   Keep the original V11m6n mainline fixed:
#     - local_delta_detach_protein = false
#     - local_delta_protein_grad_scale = 0.60
#     - use_global_delta_gate = false
#   Add only ligand-source dropout in the local_delta branch:
#     - drop02: p=0.02 for whole source, local54 source, and ligand-hidden source
#     - drop03: p=0.03 for whole source, local54 source, and ligand-hidden source
#
# Matrix:
#   0. seed42_ligdrop02_g60_no_gate
#   1. seed42_ligdrop03_g60_no_gate
#   2. seed45_ligdrop02_g60_no_gate
#   3. seed45_ligdrop03_g60_no_gate

#SBATCH --job-name=v11m6n_ligdrop
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/v11m6n_ligdrop_%A_%a.out
#SBATCH --error=logs/v11m6n_ligdrop_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=(
  "seed42_ligdrop02_g60_no_gate"
  "seed42_ligdrop03_g60_no_gate"
  "seed45_ligdrop02_g60_no_gate"
  "seed45_ligdrop03_g60_no_gate"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANT:-${VARIANTS[$TASK_ID]}}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m6n ligand-dropout task id: ${TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m6n_ligdrop_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

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
drop_match = re.search(r"ligdrop0?(\d+)", variant)
if not seed_match or not drop_match:
    raise SystemExit(f"Could not parse seed/dropout from variant: {variant}")

seed = int(seed_match.group(1))
drop_p = int(drop_match.group(1)) / 100.0

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_v11m6n_{variant}"
cfg["experiment"]["version_suffix"] = "v11m6n"
cfg["experiment"]["seed"] = seed

model = cfg["model"]
model["local_delta_detach_protein"] = False
model["use_global_delta_gate"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["ligand_source_dropout_p"] = drop_p
model["ligand_dropout_local54_p"] = drop_p
model["ligand_dropout_hidden_p"] = drop_p

cfg.setdefault("training", {})
cfg["training"]["init_from_checkpoint"] = None
cfg["training"]["freeze_v11_base"] = False

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V11m6n ligand-dropout config: {out_path}")
print(
    "variant={variant} seed={seed} grad_scale={scale} no_gate={no_gate} "
    "source_drop={source} local54_drop={local54} hidden_drop={hidden}".format(
        variant=variant,
        seed=seed,
        scale=model["local_delta_protein_grad_scale"],
        no_gate=not model["use_global_delta_gate"],
        source=model["ligand_source_dropout_p"],
        local54=model["ligand_dropout_local54_p"],
        hidden=model["ligand_dropout_hidden_p"],
    )
)
PY

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(python scripts/train/train_v11m6.py --config "${CONFIG_FILE}" --amp)
if [ -n "${RESUME_CHECKPOINT}" ]; then
  CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "Generated config:"
sed -n '1,240p' "${CONFIG_FILE}"
echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
