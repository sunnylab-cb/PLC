#!/bin/bash
#SBATCH --job-name=v11m6n_moment_init
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/v11m6n_moment_init_%A_%a.out
#SBATCH --error=logs/v11m6n_moment_init_%A_%a.err

# Moment-guided initialization probes for V11m6n no_gate grad060.
#
# Background:
#   The scale-moment extraction over trained checkpoints showed the healthy
#   V11m6n region has valid local_delta score_std ~= 1.22-1.28,
#   attention entropy ~= 3.53-3.57, and top1 ~= 0.055-0.058.
#   With local_delta_dim=128, a QKNorm target score_std near 1.25 implies
#   alpha ~= 1.25 * sqrt(128) ~= 14.1.
#
# What this array tests:
#   0 qknorm_a12_g60:
#       Conservative score-temperature prior, slightly below the fitted alpha.
#   1 qknorm_a14_g60:
#       Moment-matched QKNorm alpha; primary score-scale prior candidate.
#   2 qknorm_a16_g60:
#       Slightly sharper score prior, but still close to the healthy region.
#   3 qknorm_a14_tiny1e4_g60:
#       Moment-matched QKNorm plus tiny non-zero delta head std=1e-4.
#       Tests whether early delta-path gradients help without breaking base.
#   4 qknorm_a14_tiny3e4_g60:
#       Same as above, stronger tiny delta-head std=3e-4.
#   5 delta_scale025_g60:
#       Conservative residual-delta multiplier control.  The trained healthy
#       models keep raw scale around 0.5, so this tests whether smaller early
#       correction is safer or just underuses ligand evidence.
#
# All variants keep:
#   seed=42, no_gate=True, local_delta_detach_protein=False,
#   local_delta_protein_grad_scale=0.60, partition=i64m1tga800u.

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

case "${SLURM_ARRAY_TASK_ID}" in
  0) VARIANT="qknorm_a12_g60" ;;
  1) VARIANT="qknorm_a14_g60" ;;
  2) VARIANT="qknorm_a16_g60" ;;
  3) VARIANT="qknorm_a14_tiny1e4_g60" ;;
  4) VARIANT="qknorm_a14_tiny3e4_g60" ;;
  5) VARIANT="delta_scale025_g60" ;;
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
CONFIG_FILE="tmp/slurm_configs/v11m6n_moment_init_${VARIANT}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.yaml"

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

if variant.startswith("qknorm_"):
    model["local_delta_use_qk_norm"] = True
    if "a12" in variant:
        model["local_delta_qk_norm_scale_init"] = 12.0
    elif "a14" in variant:
        model["local_delta_qk_norm_scale_init"] = 14.0
    elif "a16" in variant:
        model["local_delta_qk_norm_scale_init"] = 16.0
    else:
        raise ValueError(f"Unknown QKNorm alpha in variant: {variant}")

if "tiny1e4" in variant:
    model["local_delta_final_init_std"] = 1.0e-4
    model["local_delta_zero_init_final"] = False
elif "tiny3e4" in variant:
    model["local_delta_final_init_std"] = 3.0e-4
    model["local_delta_zero_init_final"] = False
elif variant == "delta_scale025_g60":
    model["local_delta_logit_scale_init"] = 0.25
elif not variant.startswith("qknorm_"):
    raise ValueError(f"Unknown variant: {variant}")

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V11m6n moment-init config: {out_path}")
print(
    "variant={variant} seed={seed} grad_scale={grad_scale} no_gate={no_gate} "
    "qk_norm={qk_norm} qk_alpha={qk_alpha} raw_delta_scale={raw_scale} "
    "tiny_std={tiny_std}".format(
        variant=variant,
        seed=cfg["experiment"]["seed"],
        grad_scale=model["local_delta_protein_grad_scale"],
        no_gate=not model["use_global_delta_gate"],
        qk_norm=model.get("local_delta_use_qk_norm", False),
        qk_alpha=model.get("local_delta_qk_norm_scale_init", "NA"),
        raw_scale=model.get("local_delta_logit_scale_init", "default"),
        tiny_std=model.get("local_delta_final_init_std", 0.0),
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
