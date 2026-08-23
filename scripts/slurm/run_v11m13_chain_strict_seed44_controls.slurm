#!/usr/bin/env bash
# Seed44 strict-split controls, both trained from random initialization:
#   task 0: PDBbind-only
#   task 1: PDBbind + sequence-clean BioLiP train augmentation
# No checkpoint initialization, resume, base freezing, or delta-only tuning.

#SBATCH --job-name=v11m13_si30_s44
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=36:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_si30_s44_%A_%a.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_si30_s44_%A_%a.err

set -euo pipefail
PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
SPLIT_ROOT="${SPLIT_ROOT:?set SPLIT_ROOT to the audited chain split output}"
BASE_CONFIG="${BASE_CONFIG:-${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${TASK_ID}"

mkdir -p "${PROJECT}/logs" "${PROJECT}/tmp/slurm_configs" "${ARCHIVE}/outputs"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts/diagnostics:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [ "${TASK_ID}" = 0 ]; then
  TAG="chain_si30_pdbonly_scratch_seed44_${RUN_ID}"
  DATA_ROOT="${PROJECT}/data"
  SPLIT_FILE="${SPLIT_ROOT}/pdbbind_chain_si30_splits.txt"
elif [ "${TASK_ID}" = 1 ]; then
  TAG="chain_si30_biolipmerge_scratch_seed44_${RUN_ID}"
  DATA_ROOT="${ARCHIVE}/datasets/pdbbind_biolip_chain_si30_${SLURM_ARRAY_JOB_ID:-manual}"
  "${PYTHON}" scripts/diagnostics/materialize_chain_strict_merged_root.py \
    --split-root "${SPLIT_ROOT}" --pdbbind-root "${PROJECT}/data" \
    --biolip-root "${BIOLIP}" --output-root "${DATA_ROOT}"
  SPLIT_FILE="${DATA_ROOT}/splits.txt"
else
  echo "invalid task ${TASK_ID}" >&2
  exit 2
fi

CONFIG="${PROJECT}/tmp/slurm_configs/v11m13_${TAG}.yaml"
"${PYTHON}" - "${BASE_CONFIG}" "${CONFIG}" "${TAG}" "${DATA_ROOT}" "${SPLIT_FILE}" "${ARCHIVE}/outputs" <<'PY'
import sys
from pathlib import Path
import yaml

base, output, tag, data_root, split_file, output_root = sys.argv[1:]
cfg = yaml.safe_load(Path(base).read_text())
cfg["experiment"].update({
    "name": f"pocket_gvp_fusion_seed44_{tag}",
    "seed": 44,
    "version_suffix": "v11m13",
    "output_dir": output_root,
})
cfg["data"].update({
    "data_root": data_root,
    "split_file": split_file,
    "distances_file": str(Path(data_root) / "residue_ligand_distances_dcenter.pt"),
    "protein_features_dir": "complete_features_v4_nma",
    "nma_features_dir": "complete_features_v4_nma",
    "esm_features_dir": "esm2_latents",
    "ligand_features_dir": "ligand_features_v5",
    "max_atoms": 256,
    "atom_truncation_policy": "alternating_ends",
    "nma_n_confs": 1,
    "nma_max_conf": 0,
})
cfg["model"].update({
    "use_compile": False,
    "use_gradient_checkpointing": False,
    "use_unimol_ligand_input": False,
    "allow_zero_unimol_fallback": False,
})
train = cfg["training"]
train.update({
    "init_from_checkpoint": None,
    "freeze_v11_base": False,
    "learning_rate": 1.0e-4,
    "min_lr": 1.0e-6,
    "warmup_epochs": 5,
    "num_epochs": 120,
    "early_stopping_patience": 30,
    "batch_size": 16,
    "accumulation_steps": 1,
    "num_workers": 8,
    "save_every": 10,
    "use_bf16": True,
    "use_amp": False,
})
for key in ("delta_trainable_tokens", "resume", "resume_from_checkpoint", "checkpoint_path"):
    train.pop(key, None)
Path(output).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"scratch config={output} data={data_root} split={split_file}")
assert train["init_from_checkpoint"] is None
assert train["freeze_v11_base"] is False
PY

"${PYTHON}" scripts/train/train_v11m13.py --config "${CONFIG}" --amp --accumulation-steps 1
echo "Completed scratch control ${TAG}"
