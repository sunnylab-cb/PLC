#!/usr/bin/env bash
# BioLip side-path pilot: candidate-head pretraining, then PDBbind fine-tuning.
#
# Scientific question:
#   Can broader BioLip ligand/site supervision teach V11m16 to rank several
#   protein-only pocket candidates before the small PDBbind training set refines
#   the final residue decision? This intentionally leaves the validated V11m13
#   trunk frozen and trains only assignment_candidate_* parameters.
#
# Stage A:
#   - PDBbind-val/test sequence-clean BioLip pairs selected by BIOLIP_DATA
#   - no BioLip Uni-Mol cache: preserve Uni-Mol layers but pass the documented
#     zero fallback, avoiding checkpoint shape changes
#   - candidate auxiliary weight 0.05 to establish the ranking representation
# Stage B:
#   - restore the original PDBbind split and real Uni-Mol cache
#   - transfer Stage A query/key projections but reset gate alpha to 0.02;
#     otherwise a source-domain alpha collapse can erase the learned ranking
#   - fine-tune the same candidate head with auxiliary weight 0.02
#
# This is a pilot, not benchmark-clean evidence: sequence/similarity filtering
# against COACH420/HOLO4K must be completed before external benchmark claims.

#SBATCH --job-name=v11m16_biolip
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m16_biolip_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m16_biolip_%j.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
BIOLIP_DATA="${BIOLIP_DATA:-/hpc2hdd/home/shuang886/biolip/v11_pilot_strict_internalclean_20260811}"
BASELINE_NAME="pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114"
BASE_CONFIG="${ARCHIVE}/outputs/${BASELINE_NAME}/config.yaml"
BASE_CHECKPOINT="${ARCHIVE}/outputs/${BASELINE_NAME}/best_f1_c0_model.pt"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
RUN_ID="${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}"
CONFIG_ROOT="${PROJECT}/tmp/slurm_configs"
OUTPUT_ROOT="${ARCHIVE}/outputs"
RESULT_ROOT="${ARCHIVE}/results"

mkdir -p "${PROJECT}/logs" "${CONFIG_ROOT}" "${OUTPUT_ROOT}" "${RESULT_ROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

for required in \
  "${BASE_CONFIG}" \
  "${BASE_CHECKPOINT}" \
  "${BIOLIP_DATA}/summary.json" \
  "${BIOLIP_DATA}/dataset_qc.json" \
  "${BIOLIP_DATA}/splits.txt" \
  "${BIOLIP_DATA}/residue_ligand_distances_dcenter.pt"; do
  if [ ! -f "${required}" ]; then
    echo "ERROR: required input missing: ${required}" >&2
    exit 2
  fi
done

BIO_CONFIG="${CONFIG_ROOT}/v11m16_biolip_pretrain_${RUN_ID}.yaml"
"${PYTHON}" - "${BASE_CONFIG}" "${BIO_CONFIG}" "${BASE_CHECKPOINT}" \
  "${OUTPUT_ROOT}" "${BIOLIP_DATA}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path, out_path, checkpoint, output_root, data_root = map(Path, sys.argv[1:])
with base_path.open() as handle:
    cfg = yaml.safe_load(handle)

cfg["experiment"].update(
    {
        "name": "pocket_gvp_fusion_seed44_biolip_candidate_pretrain_aux005",
        "version_suffix": "v11m16",
        "seed": 44,
        "output_dir": str(output_root),
    }
)
cfg["data"].update(
    {
        "data_root": str(data_root),
        "split_file": str(data_root / "splits.txt"),
        "distances_file": str(data_root / "residue_ligand_distances_dcenter.pt"),
        "protein_features_dir": "complete_features_v4",
        "esm_features_dir": "esm2_latents",
        "ligand_features_dir": "ligand_features_v5",
        "nma_features_dir": "complete_features_v4_nma",
        "nma_n_confs": 1,
        "nma_max_conf": 0,
        "max_seq_len": 1800,
        "max_atoms": 128,
    }
)
cfg["model"].update(
    {
        "allow_zero_unimol_fallback": True,
        "use_assignment_embed_loss": False,
        "assignment_samevalid_mode": "none",
        "use_candidate_rank_gate": True,
        "candidate_num_sites": 4,
        "candidate_min_center_distance": 10.0,
        "candidate_membership_sigma": 8.0,
        "candidate_embed_dim": 128,
        "candidate_temperature": 0.10,
        "candidate_gate_alpha_init": 0.02,
        "candidate_gate_alpha_max": 0.30,
        "candidate_aux_weight": 0.05,
        "candidate_aux_min_coverage": 0.05,
        "use_compile": False,
        "use_gradient_checkpointing": False,
    }
)
cfg["training"].update(
    {
        "init_from_checkpoint": str(checkpoint),
        "freeze_v11_base": True,
        "delta_trainable_tokens": ["assignment_candidate_"],
        "learning_rate": 2.0e-4,
        "min_lr": 1.0e-6,
        "warmup_epochs": 2,
        "num_epochs": 30,
        "early_stopping_patience": 12,
        "batch_size": 24,
        "accumulation_steps": 1,
        "num_workers": 8,
        "save_every": 1000,
        "use_bf16": True,
        "use_amp": False,
    }
)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
print(out_path)
PY

echo "Stage A: BioLip candidate-head pretraining"
"${PYTHON}" scripts/train/train_v11m16.py \
  --config "${BIO_CONFIG}" \
  --amp \
  --accumulation-steps 1

BIO_RUN="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d \
  -name '*biolip_candidate_pretrain_aux005*v11m16*' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
BIO_CHECKPOINT="${BIO_RUN}/checkpoint_latest.pt"
if [ -z "${BIO_RUN}" ] || [ ! -f "${BIO_CHECKPOINT}" ]; then
  echo "ERROR: BioLip latest checkpoint not found" >&2
  exit 3
fi

# Alpha is a target-domain coupling parameter, not a transferable BioLip
# representation. Preserve the pretrained query/key projections while resetting
# the bounded scale to the same low-risk 0.02 used by the PDBbind-only control.
BIO_TRANSFER_CHECKPOINT="${BIO_RUN}/candidate_transfer_alpha002.pt"
"${PYTHON}" - "${BIO_CHECKPOINT}" "${BIO_TRANSFER_CHECKPOINT}" <<'PY'
import math
import sys

import torch

source, target = sys.argv[1:]
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
state = checkpoint.get("model_state_dict", checkpoint)
key = "assignment_candidate_gate_raw_alpha"
if key not in state:
    raise KeyError(f"{key} missing from BioLip checkpoint")
alpha_init = 0.02
alpha_max = 0.30
raw_alpha = math.atanh(alpha_init / alpha_max)
state[key] = torch.full_like(state[key], raw_alpha)
checkpoint["candidate_transfer"] = {
    "source": source,
    "gate_alpha_reset": alpha_init,
    "reason": "transfer query/key representation, recalibrate coupling on PDBbind",
}
torch.save(checkpoint, target)
print(f"Wrote {target}; reset {key} to bounded alpha={alpha_init}")
PY

FINETUNE_CONFIG="${CONFIG_ROOT}/v11m16_biolip_finetune_${RUN_ID}.yaml"
"${PYTHON}" - "${BASE_CONFIG}" "${FINETUNE_CONFIG}" "${BIO_TRANSFER_CHECKPOINT}" \
  "${OUTPUT_ROOT}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path, out_path, checkpoint, output_root = map(Path, sys.argv[1:])
with base_path.open() as handle:
    cfg = yaml.safe_load(handle)

cfg["experiment"].update(
    {
        "name": "pocket_gvp_fusion_seed44_biolip_candidate_finetune_aux002",
        "version_suffix": "v11m16",
        "seed": 44,
        "output_dir": str(output_root),
    }
)
cfg["model"].update(
    {
        "allow_zero_unimol_fallback": False,
        "use_assignment_embed_loss": False,
        "assignment_samevalid_mode": "none",
        "use_candidate_rank_gate": True,
        "candidate_num_sites": 4,
        "candidate_min_center_distance": 10.0,
        "candidate_membership_sigma": 8.0,
        "candidate_embed_dim": 128,
        "candidate_temperature": 0.10,
        "candidate_gate_alpha_init": 0.02,
        "candidate_gate_alpha_max": 0.30,
        "candidate_aux_weight": 0.02,
        "candidate_aux_min_coverage": 0.05,
        "use_compile": False,
        "use_gradient_checkpointing": False,
    }
)
cfg["training"].update(
    {
        "init_from_checkpoint": str(checkpoint),
        "freeze_v11_base": True,
        "delta_trainable_tokens": ["assignment_candidate_"],
        "learning_rate": 1.0e-4,
        "min_lr": 1.0e-6,
        "warmup_epochs": 2,
        "num_epochs": 40,
        "early_stopping_patience": 20,
        "batch_size": 24,
        "accumulation_steps": 1,
        "num_workers": 8,
        "save_every": 1000,
        "use_bf16": True,
        "use_amp": False,
    }
)
out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
print(out_path)
PY

echo "Stage B: PDBbind candidate-head fine-tuning with real Uni-Mol"
"${PYTHON}" scripts/train/train_v11m16.py \
  --config "${FINETUNE_CONFIG}" \
  --amp \
  --accumulation-steps 1

FINETUNE_RUN="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d \
  -name '*biolip_candidate_finetune_aux002*v11m16*' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)"
FINETUNE_CHECKPOINT="${FINETUNE_RUN}/best_f1_c0_model.pt"
if [ -z "${FINETUNE_RUN}" ] || [ ! -f "${FINETUNE_CHECKPOINT}" ]; then
  echo "ERROR: PDBbind fine-tuned best checkpoint not found" >&2
  exit 4
fi

LIGDEP_OUT="${RESULT_ROOT}/ligdep_v11m16_biolip_${RUN_ID}/test_all"
mkdir -p "${LIGDEP_OUT}"
"${PYTHON}" scripts/infer/infer_v11_internal_ligdep.py \
  --checkpoint "${FINETUNE_CHECKPOINT}" \
  --config "${FINETUNE_RUN}/config.yaml" \
  --split test \
  --ligand_mode all \
  --batch_size 8 \
  --num_workers 8 \
  --output "${LIGDEP_OUT}" \
  --amp

echo "BioLip candidate pretrain + PDBbind fine-tune complete"
echo "  BioLip run: ${BIO_RUN}"
echo "  PDBbind run: ${FINETUNE_RUN}"
echo "  ligand dependency: ${LIGDEP_OUT}"
