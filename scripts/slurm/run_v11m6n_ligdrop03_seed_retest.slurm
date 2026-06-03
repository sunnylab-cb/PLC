#!/bin/bash
# V11m6n grad060/no_gate + ligand-source dropout p=0.03 seed retest.
#
# Purpose:
#   Complete the robustness-regularization claim for the light ligand-source
#   dropout line.  The previous seed42/45 results preserved real-shuffle while
#   improving zero-ligand robustness.  This retest focuses on seed43/44/46,
#   especially seed44 whose original grad060 zero F1 was unusually low.
#
# Fixed design:
#   - local_delta_detach_protein = false
#   - local_delta_protein_grad_scale = 0.60
#   - use_global_delta_gate = false
#   - ligand_source_dropout_p = 0.03
#   - ligand_dropout_local54_p = 0.03
#   - ligand_dropout_hidden_p = 0.03
#
# Matrix:
#   0. seed43_ligdrop03_g60_no_gate
#   1. seed44_ligdrop03_g60_no_gate
#   2. seed46_ligdrop03_g60_no_gate

#SBATCH --job-name=v11m6n_ld03
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/v11m6n_ld03_%A_%a.out
#SBATCH --error=logs/v11m6n_ld03_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=(
  "seed43_ligdrop03_g60_no_gate"
  "seed44_ligdrop03_g60_no_gate"
  "seed46_ligdrop03_g60_no_gate"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANT:-${VARIANTS[$TASK_ID]}}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m6n ligdrop03 retest task id: ${TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m6n_ligdrop03_retest_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

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
drop_p = 0.03

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

print(f"Wrote V11m6n ligdrop03 retest config: {out_path}")
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
