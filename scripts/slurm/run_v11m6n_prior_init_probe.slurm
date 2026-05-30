#!/bin/bash
#SBATCH --job-name=v11m6n_prior_init
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/v11m6n_prior_init_%A_%a.out
#SBATCH --error=logs/v11m6n_prior_init_%A_%a.err

# V11m6n prior-initialization probes:
#   0 tiny_final_1e4_g60: tiny non-zero delta head for early delta-path gradients.
#   1 tiny_final_3e4_g60: stronger tiny delta-head initialization.
#   2 qknorm_g60: Q/K normalization for entropy-controlled local attention.

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) VARIANT="tiny_final_1e4_g60" ;;
  1) VARIANT="tiny_final_3e4_g60" ;;
  2) VARIANT="qknorm_g60" ;;
  *) echo "Unknown task id: ${SLURM_ARRAY_TASK_ID}" >&2; exit 2 ;;
esac

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m6n_prior_init_${VARIANT}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"

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
cfg["experiment"]["seed"] = 42

model = cfg["model"]
model["local_delta_detach_protein"] = False
model["use_global_delta_gate"] = False
model["local_delta_protein_grad_scale"] = 0.60

if variant == "tiny_final_1e4_g60":
    model["local_delta_final_init_std"] = 1.0e-4
    model["local_delta_zero_init_final"] = False
elif variant == "tiny_final_3e4_g60":
    model["local_delta_final_init_std"] = 3.0e-4
    model["local_delta_zero_init_final"] = False
elif variant == "qknorm_g60":
    model["local_delta_use_qk_norm"] = True
    model["local_delta_qk_norm_scale_init"] = 11.313708498984761
else:
    raise ValueError(f"Unknown variant: {variant}")

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V11m6n prior-init probe config: {out_path}")
print(
    "variant={variant} seed={seed} grad_scale={scale} no_gate={no_gate} "
    "tiny_std={tiny} qk_norm={qk}".format(
        variant=variant,
        seed=cfg["experiment"]["seed"],
        scale=model["local_delta_protein_grad_scale"],
        no_gate=not model["use_global_delta_gate"],
        tiny=model.get("local_delta_final_init_std", 0.0),
        qk=model.get("local_delta_use_qk_norm", False),
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
