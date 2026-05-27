#!/bin/bash
#SBATCH --job-name=v11m6_ablate
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/v11m6_ablate_%A_%a.out
#SBATCH --error=logs/v11m6_ablate_%A_%a.err

# ============================================================
# Pocket-GVP-Fusion V11m6 control matrix
# ============================================================
# Submit all four controls:
#   sbatch scripts/slurm/run_v11m6_ablation_matrix.slurm
#
# Submit one control:
#   sbatch --array=0 scripts/slurm/run_v11m6_ablation_matrix.slurm  # no_gate
#   sbatch --array=1 scripts/slurm/run_v11m6_ablation_matrix.slurm  # local54_only
#   sbatch --array=2 scripts/slurm/run_v11m6_ablation_matrix.slurm  # lig_hidden_only
#   sbatch --array=3 scripts/slurm/run_v11m6_ablation_matrix.slurm  # no_detach_protein
#
# Override base config if needed:
#   sbatch --export=ALL,BASE_CONFIG=configs/pocket_gvp_v11m6_hpc.yaml \
#     scripts/slurm/run_v11m6_ablation_matrix.slurm
#
# Matching interactive allocation:
#   srun -p i64m1tga800u -n 1 --mem=64G -c 8 --gres=gpu:1 --time=24:00:00 --pty bash
# ============================================================

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=("no_gate" "local54_only" "lig_hidden_only" "no_detach_protein")
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANT:-${VARIANTS[$TASK_ID]}}"

if [ -z "${VARIANT}" ]; then
    echo "ERROR: Unknown V11m6 ablation task id: ${TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m6_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]

with base_path.open() as f:
    cfg = yaml.safe_load(f)

cfg["experiment"]["name"] = f"pocket_gvp_fusion_v11m6_{variant}"
model = cfg["model"]

if variant == "no_gate":
    # Tests whether global55/UniMol calibration gate is necessary.
    model["use_global_delta_gate"] = False
elif variant == "local54_only":
    # Tests raw local54 chemistry contribution without encoded ligand hidden memory.
    model["local_delta_use_ligand_hidden"] = False
    model["local_delta_use_local_features"] = True
elif variant == "lig_hidden_only":
    # Tests graph/self-attention ligand hidden memory without direct local54 value/key.
    model["local_delta_use_ligand_hidden"] = True
    model["local_delta_use_local_features"] = False
elif variant == "no_detach_protein":
    # Tests whether allowing delta loss into the protein trunk helps or damages floor.
    model["local_delta_detach_protein"] = False
else:
    raise SystemExit(f"Unknown variant: {variant}")

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Wrote ablation config: {out_path}")
PY

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(python scripts/train/train_v11m6.py --config "${CONFIG_FILE}" --amp)
if [ -n "${RESUME_CHECKPOINT}" ]; then
    CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "Generated config:"
sed -n '1,220p' "${CONFIG_FILE}"
echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"

exit ${EXIT_CODE}
