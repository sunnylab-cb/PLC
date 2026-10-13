#!/bin/bash
#SBATCH --job-name=v11m6n_follow
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/v11m6n_follow_%A_%a.out
#SBATCH --error=logs/v11m6n_follow_%A_%a.err

# V11m6n follow-up matrix:
#   0: seed45_grad05_no_gate       extra seed for robustness
#   1: seed46_grad05_no_gate       extra seed for robustness
#   2: grad05_tversky075_no_gate   recall-compensated loss at grad scale 0.50
#   3: grad065_tversky075_no_gate  recall-compensated loss at grad scale 0.65

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=(
  "seed45_grad05_no_gate"
  "seed46_grad05_no_gate"
  "grad05_tversky075_no_gate"
  "grad065_tversky075_no_gate"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANT:-${VARIANTS[$TASK_ID]}}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: Unknown V11m6n follow-up task id: ${TASK_ID}"
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
echo "Partition: i64m1tga800u"
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
CONFIG_FILE="tmp/slurm_configs/v11m6n_follow_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]

with base_path.open() as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_v11m6n_{variant}"
cfg["experiment"]["version_suffix"] = "v11m6n"

model = cfg["model"]
model["local_delta_detach_protein"] = False
model["use_global_delta_gate"] = False
model["local_delta_protein_grad_scale"] = 0.50

training = cfg["training"]
seed = 42

if variant == "seed45_grad05_no_gate":
    seed = 45
elif variant == "seed46_grad05_no_gate":
    seed = 46
elif variant == "grad05_tversky075_no_gate":
    training["tversky_alpha"] = 0.25
    training["tversky_beta"] = 0.75
elif variant == "grad065_tversky075_no_gate":
    model["local_delta_protein_grad_scale"] = 0.65
    training["tversky_alpha"] = 0.25
    training["tversky_beta"] = 0.75
else:
    raise SystemExit(f"Unknown V11m6n follow-up variant: {variant}")

cfg["experiment"]["seed"] = seed

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"Wrote V11m6n follow-up config: {out_path}")
print(
    "variant={variant} seed={seed} grad_scale={scale} no_gate={no_gate} "
    "tversky_alpha={alpha} tversky_beta={beta}".format(
        variant=variant,
        seed=seed,
        scale=model["local_delta_protein_grad_scale"],
        no_gate=not model["use_global_delta_gate"],
        alpha=training["tversky_alpha"],
        beta=training["tversky_beta"],
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
sed -n '1,220p' "${CONFIG_FILE}"
echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
