#!/bin/bash
# V11m9 structured local evidence probes.
#
# Purpose:
#   Move beyond raw local54 scale tuning.  Keep the V11m8 all_light trunk fixed
#   and test two structured local-evidence routes:
#     1) group_kv_delta: base/electronic/sasa_hbond/flex_rot local54 groups add
#        bounded residual key/value evidence into local_delta atom attention.
#     2) residue_chem_compat: residue hidden state gates group-specific local54
#        chemistry readouts, then adds a bounded readout residual.
#
# Fixed baseline:
#   - all_light: film + pocket virtual tokens + center residual
#   - no global/UniMol gate
#   - grad_scale = 0.60
#   - ligand source dropout = 0.03; local54/hidden dropout = 0.04
#
# Matrix:
#   0. seed44_group_kv_delta_g60
#   1. seed45_group_kv_delta_g60
#   2. seed44_residue_chem_compat_g60
#   3. seed45_residue_chem_compat_g60

#SBATCH --job-name=v11m9_le
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=30:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/v11m9_le_%A_%a.out
#SBATCH --error=logs/v11m9_le_%A_%a.err

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
BASE_CONFIG="${BASE_CONFIG:-configs/pocket_gvp_v11m6n_hpc.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

VARIANTS=(
  "seed44_group_kv_delta_g60"
  "seed45_group_kv_delta_g60"
  "seed44_residue_chem_compat_g60"
  "seed45_residue_chem_compat_g60"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"

if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid V11m9 local evidence task id: ${TASK_ID}"
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
CONFIG_FILE="tmp/slurm_configs/v11m9_local_evidence_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"

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

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_{variant}"
cfg["experiment"]["version_suffix"] = "v11m9"
cfg["experiment"]["seed"] = seed

model = cfg["model"]
model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = True

model["use_local_ligand_film"] = True
model["use_pocket_virtual_tokens"] = True
model["use_center_aux_head"] = True
model["ligand_source_dropout_p"] = 0.03
model["ligand_dropout_local54_p"] = 0.04
model["ligand_dropout_hidden_p"] = 0.04

model["local_ligand_film_hidden"] = 256
model["local_ligand_film_alpha_init"] = 0.05
model["local_ligand_film_alpha_max"] = 0.30
model["pocket_virtual_tokens"] = 4
model["pocket_virtual_alpha_init"] = 0.05
model["pocket_virtual_alpha_max"] = 0.30
model["center_aux_hidden"] = 128
model["center_aux_logit_scale_init"] = 0.05
model["center_aux_logit_scale_max"] = 0.30

model["use_local54_chem_reweight"] = False
model["use_local54_chem_adapter"] = False

model["local_evidence_groups"] = ["base33", "electronic3", "sasa_hbond4", "flex_rot5"]
model["use_local_evidence_group_kv_delta"] = "group_kv_delta" in variant
model["local_evidence_group_kv_alpha_init"] = 0.05
model["local_evidence_group_kv_alpha_max"] = 0.30
model["use_local_evidence_residue_chem_compat"] = "residue_chem_compat" in variant
model["local_evidence_compat_hidden"] = 128
model["local_evidence_compat_alpha_init"] = 0.05
model["local_evidence_compat_alpha_max"] = 0.30

cfg.setdefault("training", {})
cfg["training"]["init_from_checkpoint"] = None
cfg["training"]["freeze_v11_base"] = False
cfg["training"]["delta_trainable_tokens"] = [
    "local_delta_",
    "global_delta_gate.",
    "local_ligand_film",
    "local54_chem_",
    "local_evidence_",
    "pocket_",
    "center_aux_",
]

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote V11m9 local evidence config: {out_path}")
print(
    "variant={variant} seed={seed} group_kv={group_kv} chem_compat={compat} "
    "groups={groups} grad_scale={scale}".format(
        variant=variant,
        seed=seed,
        group_kv=model["use_local_evidence_group_kv_delta"],
        compat=model["use_local_evidence_residue_chem_compat"],
        groups=",".join(model["local_evidence_groups"]),
        scale=model["local_delta_protein_grad_scale"],
    )
)
PY

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(python scripts/train/train_v11m9.py --config "${CONFIG_FILE}" --amp --gradient-checkpointing)
if [ -n "${RESUME_CHECKPOINT}" ]; then
  CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "Generated config:"
sed -n '1,300p' "${CONFIG_FILE}"
echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"
exit "${EXIT_CODE}"
