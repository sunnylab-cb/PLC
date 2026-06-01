#!/bin/bash
# V11m8 architecture probes on the current V11m6n mainline.
#
# Purpose:
#   Test small ligand-aware residual mechanisms without replacing the proven
#   raw V11m6n local_delta path.  All variants use grad060, no global/UniMol
#   delta gate, and gradient checkpointing.
#
# Matrix:
#   0. film_g60
#      local54/ligand-graph pooled context lightly FiLMs delta-branch protein
#      hidden states.  This probes whether ligand context should modulate the
#      residue decision surface before residue-atom attention.
#   1. virtual4_g60
#      four ligand-conditioned pocket virtual tokens aggregate protein pocket
#      evidence and write a small residual into the ligand readout.
#   2. film_virtual4_g60
#      combination of 0 and 1, still bounded by small residual scales.
#   3. center_head_g60
#      bounded class-0 coherence residual head supervised through the existing
#      residue labels.  No extra loss term is introduced in this first pass.
#   4. source_dropout_g60
#      training-time dropout of local54/ligand-hidden delta sources to test
#      whether dependency gaps become more robust without losing real F1.
#   5. all_light_g60
#      film + virtual tokens + center residual + mild source dropout.

#SBATCH --job-name=v11m8_arch
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/v11m8_arch_%A_%a.out
#SBATCH --error=logs/v11m8_arch_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=(
  "film_g60"
  "virtual4_g60"
  "film_virtual4_g60"
  "center_head_g60"
  "source_dropout_g60"
  "all_light_g60"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANT:-${VARIANTS[$TASK_ID]}}"
SEED="${SEED:-42}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m8 architecture task id: ${TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m8_arch_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

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
cfg["experiment"]["name"] = f"pocket_gvp_fusion_v11m8_{variant}"
cfg["experiment"]["version_suffix"] = "v11m8"
cfg["experiment"]["seed"] = seed

model = cfg["model"]

# Keep the V11m6n mainline fixed across the probe matrix.
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = True
model["use_local_ligand_film"] = False
model["use_pocket_virtual_tokens"] = False
model["use_center_aux_head"] = False
model["ligand_source_dropout_p"] = 0.0
model["ligand_dropout_local54_p"] = 0.0
model["ligand_dropout_hidden_p"] = 0.0

# Conservative bounded residual defaults.  If a single branch wins, later
# probes can sweep these scales independently.
model["local_ligand_film_hidden"] = 256
model["local_ligand_film_alpha_init"] = 0.05
model["local_ligand_film_alpha_max"] = 0.30
model["pocket_virtual_tokens"] = 4
model["pocket_virtual_alpha_init"] = 0.05
model["pocket_virtual_alpha_max"] = 0.30
model["center_aux_hidden"] = 128
model["center_aux_logit_scale_init"] = 0.05
model["center_aux_logit_scale_max"] = 0.30

if variant == "film_g60":
    model["use_local_ligand_film"] = True
elif variant == "virtual4_g60":
    model["use_pocket_virtual_tokens"] = True
elif variant == "film_virtual4_g60":
    model["use_local_ligand_film"] = True
    model["use_pocket_virtual_tokens"] = True
elif variant == "center_head_g60":
    model["use_center_aux_head"] = True
elif variant == "source_dropout_g60":
    model["ligand_source_dropout_p"] = 0.04
    model["ligand_dropout_local54_p"] = 0.06
    model["ligand_dropout_hidden_p"] = 0.06
elif variant == "all_light_g60":
    model["use_local_ligand_film"] = True
    model["use_pocket_virtual_tokens"] = True
    model["use_center_aux_head"] = True
    model["ligand_source_dropout_p"] = 0.03
    model["ligand_dropout_local54_p"] = 0.04
    model["ligand_dropout_hidden_p"] = 0.04
else:
    raise SystemExit(f"Unknown V11m8 architecture variant: {variant}")

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

print(f"Wrote V11m8 architecture config: {out_path}")
print(
    "variant={variant} seed={seed} grad_scale={scale} film={film} virtual={virtual} "
    "center={center} source_drop={source_drop} local_drop={local_drop} hidden_drop={hidden_drop}".format(
        variant=variant,
        seed=seed,
        scale=model["local_delta_protein_grad_scale"],
        film=model["use_local_ligand_film"],
        virtual=model["use_pocket_virtual_tokens"],
        center=model["use_center_aux_head"],
        source_drop=model["ligand_source_dropout_p"],
        local_drop=model["ligand_dropout_local54_p"],
        hidden_drop=model["ligand_dropout_hidden_p"],
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
