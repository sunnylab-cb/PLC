#!/usr/bin/env bash
# Train V11m13 + weak ESM3 global prior from scratch on the audited strict
# PDBbind+BioLiP mainline. No missing-feature fallback is permitted.
# Array mapping: seed43, seed44, seed45.

#SBATCH --job-name=v11m13_si30_e3
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=36:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_si30_e3_%A_%a.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_si30_e3_%A_%a.err

set -euo pipefail
PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
SPLIT_ROOT="${SPLIT_ROOT:-/hpc2hdd/home/shuang886/biolip/pdbbind_chain_si30_split_11283646_final}"
DATA_ROOT="${DATA_ROOT:-${ARCHIVE}/datasets/pdbbind_biolip_chain_si30_esm3}"
BASE_CONFIG="${BASE_CONFIG:-${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
SEEDS=(43 44 45)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$TASK_ID]:?invalid array task ${TASK_ID}}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${TASK_ID}"
TAG="chain_si30_biolipmerge_esm3gp_scratch_seed${SEED}_${RUN_ID}"

mkdir -p "${PROJECT}/logs" "${PROJECT}/tmp/slurm_configs" "${ARCHIVE}/outputs"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PROJECT}/scripts/diagnostics:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

test -f "${DATA_ROOT}/summary.json"
test -d "${DATA_ROOT}/esm3_latents_norm"

CONFIG="${PROJECT}/tmp/slurm_configs/v11m13_${TAG}.yaml"
"${PYTHON}" - "${BASE_CONFIG}" "${CONFIG}" "${TAG}" "${SEED}" "${DATA_ROOT}" "${ARCHIVE}/outputs" <<'PY'
import sys
from collections import Counter
from pathlib import Path
import yaml

base, output, tag, seed, data_root, output_root = sys.argv[1:]
seed = int(seed)
data_root = Path(data_root)
counts = Counter(line.split()[0] for line in (data_root / "splits.txt").read_text().splitlines() if line and not line.startswith("#"))
expected = {"train": 10317, "val": 647, "test": 647}
if dict(counts) != expected:
    raise RuntimeError(f"strict merged split mismatch: {dict(counts)} != {expected}")

cfg = yaml.safe_load(Path(base).read_text())
cfg["experiment"].update({
    "name": f"pocket_gvp_fusion_seed{seed}_{tag}", "seed": seed,
    "version_suffix": "v11m13e3ft", "output_dir": output_root,
})
cfg["data"].update({
    "data_root": str(data_root), "split_file": str(data_root / "splits.txt"),
    "distances_file": str(data_root / "residue_ligand_distances_dcenter.pt"),
    "protein_features_dir": "complete_features_v4_nma", "nma_features_dir": "complete_features_v4_nma",
    "esm_features_dir": "esm2_latents", "esm3_features_dir": "esm3_latents_norm",
    "ligand_features_dir": "ligand_features_v5", "max_atoms": 256,
    "atom_truncation_policy": "alternating_ends", "nma_n_confs": 1, "nma_max_conf": 0,
})
cfg["model"].update({
    "use_compile": False, "use_gradient_checkpointing": False,
    "use_unimol_ligand_input": False, "allow_zero_unimol_fallback": False,
    "use_esm3": True, "d_esm3": 1536, "esm3_fusion": "global_prior",
    "esm3_alpha_init": 0.02, "esm3_alpha_max": 0.20,
    "esm3_gate_bias_init": 1.5, "esm3_zero_init": True, "esm3_dropout_prob": 0.0,
})
train = cfg["training"]
train.update({
    "init_from_checkpoint": None, "freeze_v11_base": False,
    "learning_rate": 2.0e-4, "min_lr": 1.0e-6, "warmup_epochs": 3,
    "num_epochs": 120, "early_stopping_patience": 30,
    "batch_size": 16, "accumulation_steps": 1, "num_workers": 8,
    "save_every": 10, "use_bf16": True, "use_amp": False,
})
for key in ("delta_trainable_tokens", "resume", "resume_from_checkpoint", "checkpoint_path"):
    train.pop(key, None)
Path(output).write_text(yaml.safe_dump(cfg, sort_keys=False))
assert train["init_from_checkpoint"] is None and not train["freeze_v11_base"]
assert cfg["model"]["use_esm3"] and cfg["data"]["esm3_features_dir"] == "esm3_latents_norm"
print(f"config={output} seed={seed} split_counts={dict(counts)}")
PY

"${PYTHON}" scripts/train/train_v11m13.py --config "${CONFIG}" --amp --accumulation-steps 1
