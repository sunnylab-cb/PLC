#!/bin/bash
# Re-run current top ligand-aware candidates with a larger ligand atom budget.
#
# Motivation:
#   The legacy data.max_atoms cap was introduced as an early padding/memory
#   engineering limit (100 in V2, 128 since V10).  It truncates ligand atoms
#   during training and becomes especially problematic for external aggregate
#   benchmark samples where multiple ligands are merged before truncation.
#
# Primary variable:
#   data.max_atoms is raised from the checkpoint/config value (usually 128) to
#   512.  Architecture, warm-start checkpoint, freeze policy, LR, and loss are
#   inherited from each run's original output config.
#
# Truncation policy:
#   data.atom_truncation_policy=alternating_ends.  If a cap is still hit, atoms
#   are removed front/back/front/back, leaving a contiguous middle segment and
#   remapping ligand graph edges safely.
#
# OOM fallback policy:
#   1) max_atoms=512, original batch/accumulation
#   2) max_atoms=512, batch=12 with accumulation adjusted to keep effective batch
#   3) max_atoms=512, batch=6 with accumulation adjusted
#   4) max_atoms=384, batch=12
#   5) max_atoms=256, original batch/accumulation
#
# Top candidates are selected by current best_val_f1_c0 metadata among completed
# strong runs, not by benchmark-specific post hoc filtering.

#SBATCH --job-name=top5_atoms
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=48:00:00
#SBATCH --output=logs/top5_atoms_%j.out
#SBATCH --error=logs/top5_atoms_%j.err

set -o pipefail

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"

RUN_NAMES=(
  "seed44_u_mem4b2_lr2e4_maxatoms"
  "seed44_u_mem1b2_lr2e4_maxatoms"
  "seed44_all3_lr5e4_maxatoms"
  "seed44_group_rescompat_lr2e4_maxatoms"
  "seed44_chem_adapter_lr2e4_maxatoms"
)

BASE_CONFIGS=(
  "outputs/pocket_gvp_fusion_seed44_u_mem4b2_lr2e4_v11m12_20260611_111611/config.yaml"
  "outputs/pocket_gvp_fusion_seed44_u_mem1b2_lr2e4_v11m12_20260611_052606/config.yaml"
  "outputs/pocket_gvp_fusion_seed44_all3_lr5e4_v11m13_20260611_205603/config.yaml"
  "outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml"
  "outputs/pocket_gvp_fusion_seed44_chem_adapter_lr2e4_v11m13_20260611_205604/config.yaml"
)

TRAIN_SCRIPTS=(
  "scripts/train/train_v11m12.py"
  "scripts/train/train_v11m12.py"
  "scripts/train/train_v11m13.py"
  "scripts/train/train_v11m13.py"
  "scripts/train/train_v11m13.py"
)

TASK_ID="${TOP5_TASK_ID:-${SLURM_ARRAY_TASK_ID:-0}}"
RUN_NAME="${RUN_NAMES[$TASK_ID]}"
BASE_CONFIG="${BASE_CONFIGS[$TASK_ID]}"
TRAIN_SCRIPT="${TRAIN_SCRIPTS[$TASK_ID]}"

if [ -z "${RUN_NAME}" ] || [ -z "${BASE_CONFIG}" ] || [ -z "${TRAIN_SCRIPT}" ]; then
  echo "ERROR: invalid task id ${TASK_ID}"
  exit 2
fi

mkdir -p logs

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Run name: ${RUN_NAME}"
echo "Base config: ${BASE_CONFIG}"
echo "Train script: ${TRAIN_SCRIPT}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start: $(date)"
echo "============================================"

if command -v module &> /dev/null; then
  module purge || true
fi

if command -v conda &> /dev/null; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
  echo "ERROR: conda command not found in PATH"
  exit 1
fi
conda activate af3
set -u

cd "${PROJECT_DIR}" || exit 1

if [ ! -f "${BASE_CONFIG}" ]; then
  echo "ERROR: base config not found: ${BASE_CONFIG}"
  exit 3
fi
if [ ! -f "${TRAIN_SCRIPT}" ]; then
  echo "ERROR: train script not found: ${TRAIN_SCRIPT}"
  exit 4
fi

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

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CONFIG_DIR="tmp/slurm_configs"
mkdir -p "${CONFIG_DIR}"

make_config() {
  local max_atoms="$1"
  local batch_size="$2"
  local accum_steps="$3"
  local out_config="$4"

  python - "${BASE_CONFIG}" "${out_config}" "${RUN_NAME}" "${max_atoms}" "${batch_size}" "${accum_steps}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
run_name = sys.argv[3]
max_atoms = int(sys.argv[4])
batch_size = int(sys.argv[5])
accum_steps = int(sys.argv[6])

with base_path.open() as handle:
    cfg = yaml.safe_load(handle)

cfg.setdefault("data", {})
old_max_atoms = int(cfg["data"].get("max_atoms", 128))
cfg["data"]["max_atoms"] = max_atoms
old_policy = cfg["data"].get("atom_truncation_policy", "head")
cfg["data"]["atom_truncation_policy"] = "alternating_ends"

cfg.setdefault("experiment", {})
old_name = cfg["experiment"].get("name", run_name)
cfg["experiment"]["name"] = f"{old_name}_maxatoms{max_atoms}_{run_name}"

cfg.setdefault("training", {})
old_batch = int(cfg["training"].get("batch_size", batch_size))
old_accum = int(cfg["training"].get("accumulation_steps", 1))
cfg["training"]["batch_size"] = batch_size
cfg["training"]["accumulation_steps"] = accum_steps

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

print(f"Wrote config: {out_path}")
print(
    "max_atoms {old_max}->{new_max}, batch {old_batch}->{new_batch}, "
    "accum {old_accum}->{new_accum}, effective_batch={eff}, "
    "atom_policy {old_policy}->alternating_ends".format(
        old_max=old_max_atoms,
        new_max=max_atoms,
        old_batch=old_batch,
        new_batch=batch_size,
        old_accum=old_accum,
        new_accum=accum_steps,
        eff=batch_size * accum_steps,
        old_policy=old_policy,
    )
)
print(f"experiment.name={cfg['experiment']['name']}")
PY
}

read_original_batch() {
  python - "${BASE_CONFIG}" <<'PY'
import sys, yaml
with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)
train = cfg.get("training", {})
print(int(train.get("batch_size", 24)))
PY
}

read_original_accum() {
  python - "${BASE_CONFIG}" <<'PY'
import sys, yaml
with open(sys.argv[1]) as handle:
    cfg = yaml.safe_load(handle)
train = cfg.get("training", {})
print(int(train.get("accumulation_steps", 1)))
PY
}

ceil_div() {
  local a="$1"
  local b="$2"
  echo $(( (a + b - 1) / b ))
}

ORIG_BATCH="$(read_original_batch)"
ORIG_ACCUM="$(read_original_accum)"
EFFECTIVE_BATCH=$(( ORIG_BATCH * ORIG_ACCUM ))
ACCUM_12="$(ceil_div "${EFFECTIVE_BATCH}" 12)"
ACCUM_6="$(ceil_div "${EFFECTIVE_BATCH}" 6)"

ATTEMPTS=(
  "512:${ORIG_BATCH}:${ORIG_ACCUM}:max512_origbatch"
  "512:12:${ACCUM_12}:max512_batch12"
  "512:6:${ACCUM_6}:max512_batch6"
  "384:12:${ACCUM_12}:max384_batch12"
  "256:${ORIG_BATCH}:${ORIG_ACCUM}:max256_origbatch"
)

echo "Original batch=${ORIG_BATCH}, accum=${ORIG_ACCUM}, effective=${EFFECTIVE_BATCH}"
echo "Attempts:"
printf '  %s\n' "${ATTEMPTS[@]}"

FINAL_RC=1
for attempt in "${ATTEMPTS[@]}"; do
  IFS=":" read -r MAX_ATOMS BATCH_SIZE ACCUM_STEPS LABEL <<< "${attempt}"
  CONFIG_FILE="${CONFIG_DIR}/top5_atoms_${RUN_NAME}_${LABEL}_${SLURM_JOB_ID}_${TASK_ID}.yaml"
  ATTEMPT_LOG="logs/top5_atoms_${SLURM_JOB_ID}_${TASK_ID}_${LABEL}.attempt.log"

  echo "============================================"
  echo "Attempt ${LABEL}: max_atoms=${MAX_ATOMS}, batch=${BATCH_SIZE}, accum=${ACCUM_STEPS}"
  echo "Config: ${CONFIG_FILE}"
  echo "Attempt log: ${ATTEMPT_LOG}"
  echo "============================================"

  make_config "${MAX_ATOMS}" "${BATCH_SIZE}" "${ACCUM_STEPS}" "${CONFIG_FILE}"
  sed -n '1,220p' "${CONFIG_FILE}"

  set +e
  python "${TRAIN_SCRIPT}" --config "${CONFIG_FILE}" --amp --accumulation-steps "${ACCUM_STEPS}" 2>&1 | tee "${ATTEMPT_LOG}"
  RC=${PIPESTATUS[0]}
  set -u

  if [ "${RC}" -eq 0 ]; then
    echo "Attempt ${LABEL} completed successfully."
    FINAL_RC=0
    break
  fi

  echo "Attempt ${LABEL} failed with exit code ${RC}."
  if grep -Eiq "out of memory|OOM|hip.*memory|CUDA.*memory|MIOpen.*memory|MemoryError" "${ATTEMPT_LOG}"; then
    echo "OOM-like failure detected; trying fallback."
    FINAL_RC="${RC}"
    continue
  fi

  echo "Non-OOM failure detected; stopping instead of masking a real error."
  FINAL_RC="${RC}"
  break
done

echo "============================================"
echo "End: $(date)"
echo "Final exit code: ${FINAL_RC}"
echo "============================================"
exit "${FINAL_RC}"
