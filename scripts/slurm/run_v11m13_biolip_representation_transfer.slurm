#!/usr/bin/env bash
# BioLip representation-transfer experiments on the validated V11m13 mainline.
#
# Scientific question:
#   Does BioLip improve the ligand-conditioned residue representation when its
#   supervision is allowed to update the proven local54/ligand-graph pathway,
#   rather than only a 133k-parameter candidate-ranking head?
#
# Array variants:
#   0 latefusion_warm
#     Initialize from the strongest V11m13 group_rescompat checkpoint. Freeze
#     the protein feature trunk, but train ligand GNN/self-attention, co-attn,
#     local54/group_rescompat/local-delta, pocket residuals, and classifier.
#
#   1 scratch_full
#     The large control requested by the user: train every V11m13 parameter
#     from random initialization on BioLip, then fine-tune every parameter on
#     PDBbind. UniMol is disabled in both stages because BioLip has no matching
#     cache and prior source ablations showed only a negligible UniMol effect.
#
# Both variants:
#   Stage A: 5,786 sequence-clean, regular-organic BioLip pairs.
#   Stage B: PDBbind using the exact V11m13 loss/data recipe.
#   Final: test real/shuffle/zero ligand-dependency evaluation.

#SBATCH --job-name=v11m13_biolip_repr
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_biolip_repr_%A_%a.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_biolip_repr_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
BIOLIP_DATA="${BIOLIP_DATA:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
BASELINE_NAME="pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114"
BASE_CONFIG="${BASE_CONFIG:-${ARCHIVE}/outputs/${BASELINE_NAME}/config.yaml}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-${ARCHIVE}/outputs/${BASELINE_NAME}/best_f1_c0_model.pt}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARCHIVE}/outputs}"
RESULT_ROOT="${RESULT_ROOT:-${ARCHIVE}/results}"
CONFIG_ROOT="${PROJECT}/tmp/slurm_configs"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${TASK_ID}"

VARIANTS=(
  "latefusion_warm"
  "scratch_full"
)
VARIANT="${VARIANTS[$TASK_ID]:-}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid array task ${TASK_ID}" >&2
  exit 2
fi

mkdir -p "${PROJECT}/logs" "${CONFIG_ROOT}" "${OUTPUT_ROOT}" "${RESULT_ROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

for required in \
  "${BASE_CONFIG}" \
  "${BASE_CHECKPOINT}" \
  "${BIOLIP_DATA}/summary.json" \
  "${BIOLIP_DATA}/dataset_qc.json" \
  "${BIOLIP_DATA}/splits.txt" \
  "${BIOLIP_DATA}/residue_ligand_distances_dcenter.pt"; do
  if [ ! -f "${required}" ]; then
    echo "ERROR: required input missing: ${required}" >&2
    exit 3
  fi
done

echo "BioLip V11m13 representation transfer"
echo "  job=${SLURM_JOB_ID:-manual} task=${TASK_ID} variant=${VARIANT}"
echo "  node=${SLURMD_NODENAME:-local}"
echo "  BioLip=${BIOLIP_DATA}"
echo "  baseline=${BASE_CHECKPOINT}"
echo "  output=${OUTPUT_ROOT}"

LATE_TOKENS=(
  "ligand_encoder."
  "ligand_self_attn."
  "cross_attention_layers."
  "cross_attn_res."
  "lig_attn_res."
  "local_delta_"
  "local_ligand_film"
  "pocket_"
  "local54_anchor_proj."
  "local54_group_proj."
  "local54_interaction_"
  "local54_compat_"
  "classifier_input_norm."
  "classifier."
  "dist_aux_head."
)
LATE_TOKENS_CSV="$(IFS=,; echo "${LATE_TOKENS[*]}")"

BIO_TAG="biolip_${VARIANT}_pretrain_${RUN_ID}"
BIO_CONFIG="${CONFIG_ROOT}/v11m13_${BIO_TAG}.yaml"

"${PYTHON}" - \
  "${BASE_CONFIG}" "${BIO_CONFIG}" "${VARIANT}" "${BIOLIP_DATA}" \
  "${BASE_CHECKPOINT}" "${OUTPUT_ROOT}" "${BIO_TAG}" "${LATE_TOKENS_CSV}" <<'PY'
import sys
from pathlib import Path

import yaml

(
    base_config,
    output_config,
    variant,
    biolip_root,
    base_checkpoint,
    output_root,
    run_tag,
    late_tokens_csv,
) = sys.argv[1:]

with Path(base_config).open() as handle:
    cfg = yaml.safe_load(handle)

cfg["experiment"].update(
    {
        "name": f"pocket_gvp_fusion_seed44_{run_tag}",
        "version_suffix": "v11m13",
        "seed": 44,
        "output_dir": output_root,
    }
)
cfg["data"].update(
    {
        "data_root": biolip_root,
        "split_file": str(Path(biolip_root) / "splits.txt"),
        "distances_file": str(Path(biolip_root) / "residue_ligand_distances_dcenter.pt"),
        "protein_features_dir": "complete_features_v4",
        "esm_features_dir": "esm2_latents",
        "ligand_features_dir": "ligand_features_v5",
        "nma_features_dir": "complete_features_v4_nma",
        "nma_n_confs": 1,
        "nma_max_conf": 0,
        "max_seq_len": 1800,
        "max_atoms": 256,
        "atom_truncation_policy": "alternating_ends",
    }
)

model = cfg["model"]
model["use_compile"] = False
model["use_gradient_checkpointing"] = False
model["use_candidate_rank_gate"] = False
model["use_assignment_embed_loss"] = False
model["assignment_samevalid_mode"] = "none"

train = cfg["training"]
train.update(
    {
        "min_lr": 1.0e-6,
        "num_workers": 8,
        "accumulation_steps": 1,
        "save_every": 10,
        "use_bf16": True,
        "use_amp": False,
    }
)

if variant == "latefusion_warm":
    # Preserve checkpoint tensor shapes while BioLip supplies an explicit zero
    # UniMol fallback. This is the matched low-risk representation control.
    model["use_unimol_ligand_input"] = True
    model["allow_zero_unimol_fallback"] = True
    train.update(
        {
            "init_from_checkpoint": base_checkpoint,
            "freeze_v11_base": True,
            "delta_trainable_tokens": [
                token for token in late_tokens_csv.split(",") if token
            ],
            "learning_rate": 5.0e-5,
            "warmup_epochs": 3,
            "num_epochs": 50,
            "early_stopping_patience": 12,
            "batch_size": 16,
        }
    )
elif variant == "scratch_full":
    # Keep the modality set identical across BioLip and PDBbind. UniMol was a
    # near-zero source in dependency ablations and has no BioLip cache.
    model["use_unimol_ligand_input"] = False
    model["allow_zero_unimol_fallback"] = False
    train.update(
        {
            "init_from_checkpoint": None,
            "freeze_v11_base": False,
            "learning_rate": 1.0e-4,
            "warmup_epochs": 5,
            "num_epochs": 80,
            "early_stopping_patience": 18,
            "batch_size": 16,
        }
    )
    train.pop("delta_trainable_tokens", None)
else:
    raise ValueError(f"unknown variant: {variant}")

output_path = Path(output_config)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

print(output_path)
print(
    f"variant={variant} init={train.get('init_from_checkpoint')} "
    f"freeze={train['freeze_v11_base']} lr={train['learning_rate']} "
    f"epochs={train['num_epochs']} batch={train['batch_size']} "
    f"unimol={model['use_unimol_ligand_input']}"
)
PY

if [ "${CONFIG_ONLY:-0}" = "1" ]; then
  echo "CONFIG_ONLY=1: generated Stage-A config without launching training"
  sed -n '1,280p' "${BIO_CONFIG}"
  exit 0
fi

echo "Stage A: BioLip ${VARIANT} representation pretraining"
"${PYTHON}" scripts/train/train_v11m13.py \
  --config "${BIO_CONFIG}" \
  --amp \
  --accumulation-steps 1

BIO_RUN="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d \
  -name "*${BIO_TAG}*v11m13*" -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)"
if [ -z "${BIO_RUN}" ]; then
  echo "ERROR: BioLip output directory not found for ${BIO_TAG}" >&2
  exit 4
fi
BIO_CHECKPOINT="${BIO_RUN}/best_auprc_c0_model.pt"
if [ ! -f "${BIO_CHECKPOINT}" ]; then
  BIO_CHECKPOINT="${BIO_RUN}/best_f1_c0_model.pt"
fi
if [ ! -f "${BIO_CHECKPOINT}" ]; then
  echo "ERROR: no transferable BioLip checkpoint in ${BIO_RUN}" >&2
  exit 5
fi

FINETUNE_TAG="biolip_${VARIANT}_pdbbind_finetune_${RUN_ID}"
FINETUNE_CONFIG="${CONFIG_ROOT}/v11m13_${FINETUNE_TAG}.yaml"

"${PYTHON}" - \
  "${BASE_CONFIG}" "${FINETUNE_CONFIG}" "${VARIANT}" "${BIO_CHECKPOINT}" \
  "${OUTPUT_ROOT}" "${FINETUNE_TAG}" "${LATE_TOKENS_CSV}" <<'PY'
import sys
from pathlib import Path

import yaml

(
    base_config,
    output_config,
    variant,
    bio_checkpoint,
    output_root,
    run_tag,
    late_tokens_csv,
) = sys.argv[1:]

with Path(base_config).open() as handle:
    cfg = yaml.safe_load(handle)

cfg["experiment"].update(
    {
        "name": f"pocket_gvp_fusion_seed44_{run_tag}",
        "version_suffix": "v11m13",
        "seed": 44,
        "output_dir": output_root,
    }
)
model = cfg["model"]
model["use_compile"] = False
model["use_gradient_checkpointing"] = False
model["use_candidate_rank_gate"] = False
model["use_assignment_embed_loss"] = False
model["assignment_samevalid_mode"] = "none"
model["allow_zero_unimol_fallback"] = False

train = cfg["training"]
train.update(
    {
        "init_from_checkpoint": bio_checkpoint,
        "min_lr": 1.0e-6,
        "warmup_epochs": 3,
        "num_epochs": 60,
        "early_stopping_patience": 20,
        "batch_size": 16,
        "accumulation_steps": 1,
        "num_workers": 8,
        "save_every": 10,
        "use_bf16": True,
        "use_amp": False,
    }
)

if variant == "latefusion_warm":
    model["use_unimol_ligand_input"] = True
    train["freeze_v11_base"] = True
    train["delta_trainable_tokens"] = [
        token for token in late_tokens_csv.split(",") if token
    ]
    train["learning_rate"] = 2.0e-5
elif variant == "scratch_full":
    model["use_unimol_ligand_input"] = False
    train["freeze_v11_base"] = False
    train.pop("delta_trainable_tokens", None)
    train["learning_rate"] = 2.0e-5
else:
    raise ValueError(f"unknown variant: {variant}")

output_path = Path(output_config)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

print(output_path)
print(
    f"variant={variant} transfer={bio_checkpoint} "
    f"freeze={train['freeze_v11_base']} lr={train['learning_rate']} "
    f"unimol={model['use_unimol_ligand_input']}"
)
PY

echo "Stage B: PDBbind ${VARIANT} fine-tuning"
"${PYTHON}" scripts/train/train_v11m13.py \
  --config "${FINETUNE_CONFIG}" \
  --amp \
  --accumulation-steps 1

FINETUNE_RUN="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d \
  -name "*${FINETUNE_TAG}*v11m13*" -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)"
FINETUNE_CHECKPOINT="${FINETUNE_RUN}/best_f1_c0_model.pt"
if [ -z "${FINETUNE_RUN}" ] || [ ! -f "${FINETUNE_CHECKPOINT}" ]; then
  echo "ERROR: PDBbind best checkpoint not found for ${FINETUNE_TAG}" >&2
  exit 6
fi

LIGDEP_OUT="${RESULT_ROOT}/ligdep_v11m13_biolip_repr_${RUN_ID}/${VARIANT}_test_all"
mkdir -p "${LIGDEP_OUT}"
echo "Final test ligand-dependency evaluation"
"${PYTHON}" scripts/infer/infer_v11_internal_ligdep.py \
  --checkpoint "${FINETUNE_CHECKPOINT}" \
  --config "${FINETUNE_RUN}/config.yaml" \
  --split test \
  --ligand_mode all \
  --batch_size 8 \
  --num_workers 8 \
  --output "${LIGDEP_OUT}" \
  --amp

echo "BioLip representation transfer completed"
echo "  variant=${VARIANT}"
echo "  BioLip run=${BIO_RUN}"
echo "  PDBbind run=${FINETUNE_RUN}"
echo "  ligand dependency=${LIGDEP_OUT}"
