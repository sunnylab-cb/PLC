#!/usr/bin/env bash
# Replicate the successful BioLip scratch-transfer recipe for seeds 43 and 45.
#
# Each task runs the complete three-stage path used by seed44:
#   A. full V11m13 scratch pretraining on sequence-clean BioLip
#   B. full-model PDBbind transfer from BioLip best AUPRC
#   C. fresh-optimizer continuation from Stage-B best F1
# Then evaluate PDBbind ligand dependency and strict per-ligand
# COACH420/HOLO4K generalization with no ligand atom cap.

#SBATCH --job-name=v11m13_bio_repl
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_biolip_repl_%A_%a.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_biolip_repl_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
BASE_CONFIG="${BASE_CONFIG:-${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARCHIVE}/outputs}"
RESULT_ROOT="${RESULT_ROOT:-${ARCHIVE}/results}"
BENCH_ROOT="${BENCH_ROOT:-${PROJECT}/benchmark}"
FEATURE_STATS="${FEATURE_STATS:-${PROJECT}/data/stats_output.pt}"
SEEDS=(43 45)
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
SEED="${SEEDS[$TASK_ID]:-}"
RUN_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}_${TASK_ID}"

if [ -z "${SEED}" ]; then
  echo "ERROR: invalid task ${TASK_ID}" >&2
  exit 2
fi

mkdir -p "${PROJECT}/logs" "${PROJECT}/tmp/slurm_configs" "${OUTPUT_ROOT}" "${RESULT_ROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

for required in \
  "${BASE_CONFIG}" \
  "${BIOLIP}/splits.txt" \
  "${BIOLIP}/residue_ligand_distances_dcenter.pt" \
  "${FEATURE_STATS}" \
  "${BENCH_ROOT}/residue_ligand_tags/atom_h/coach420_per_ligand_dmin.pt" \
  "${BENCH_ROOT}/residue_ligand_tags/atom_h/holo4k_per_ligand_dmin.pt"; do
  if [ ! -e "${required}" ]; then
    echo "ERROR: missing ${required}" >&2
    exit 3
  fi
done

make_config() {
  local stage="$1"
  local init_checkpoint="$2"
  local output_config="$3"
  local run_tag="$4"

  "${PYTHON}" - \
    "${BASE_CONFIG}" "${output_config}" "${stage}" "${init_checkpoint}" \
    "${OUTPUT_ROOT}" "${run_tag}" "${SEED}" "${BIOLIP}" <<'PY'
import sys
from pathlib import Path

import yaml

base, output, stage, init_checkpoint, output_root, run_tag, seed, biolip = sys.argv[1:]
with Path(base).open() as handle:
    cfg = yaml.safe_load(handle)

cfg["experiment"].update(
    {
        "name": f"pocket_gvp_fusion_seed{seed}_{run_tag}",
        "version_suffix": "v11m13",
        "seed": int(seed),
        "output_dir": output_root,
    }
)
model = cfg["model"]
model.update(
    {
        "use_compile": False,
        "use_gradient_checkpointing": False,
        "use_unimol_ligand_input": False,
        "allow_zero_unimol_fallback": False,
    }
)
train = cfg["training"]
train.update(
    {
        "init_from_checkpoint": None if init_checkpoint == "NONE" else init_checkpoint,
        "freeze_v11_base": False,
        "batch_size": 16,
        "accumulation_steps": 1,
        "num_workers": 8,
        "save_every": 10,
        "use_bf16": True,
        "use_amp": False,
    }
)
train.pop("delta_trainable_tokens", None)

if stage == "biolip":
    root = Path(biolip)
    cfg["data"].update(
        {
            "data_root": str(root),
            "split_file": str(root / "splits.txt"),
            "distances_file": str(root / "residue_ligand_distances_dcenter.pt"),
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
    train.update(
        {
            "learning_rate": 1.0e-4,
            "min_lr": 1.0e-6,
            "warmup_epochs": 5,
            "num_epochs": 80,
            "early_stopping_patience": 18,
        }
    )
elif stage == "transfer":
    train.update(
        {
            "learning_rate": 2.0e-5,
            "min_lr": 1.0e-6,
            "warmup_epochs": 3,
            "num_epochs": 60,
            "early_stopping_patience": 20,
        }
    )
elif stage == "continue":
    train.update(
        {
            "learning_rate": 2.0e-5,
            "min_lr": 5.0e-7,
            "warmup_epochs": 3,
            "num_epochs": 80,
            "early_stopping_patience": 80,
        }
    )
else:
    raise ValueError(stage)

out = Path(output)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)
print(
    f"stage={stage} seed={seed} config={out} init={train['init_from_checkpoint']} "
    f"lr={train['learning_rate']} epochs={train['num_epochs']}"
)
PY
}

find_run() {
  local tag="$1"
  find "${OUTPUT_ROOT}" -maxdepth 1 -type d -name "*${tag}*v11m13*" \
    -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-
}

echo "BioLip replication seed=${SEED} job=${RUN_ID}"

STAGE_A_TAG="biolip_scratch_repl_pretrain_${RUN_ID}"
STAGE_A_CONFIG="${PROJECT}/tmp/slurm_configs/v11m13_${STAGE_A_TAG}.yaml"
make_config biolip NONE "${STAGE_A_CONFIG}" "${STAGE_A_TAG}"
"${PYTHON}" scripts/train/train_v11m13.py --config "${STAGE_A_CONFIG}" --amp --accumulation-steps 1
STAGE_A_RUN="$(find_run "${STAGE_A_TAG}")"
STAGE_A_CKPT="${STAGE_A_RUN}/best_auprc_c0_model.pt"
test -f "${STAGE_A_CKPT}" || STAGE_A_CKPT="${STAGE_A_RUN}/best_f1_c0_model.pt"
test -f "${STAGE_A_CKPT}"

STAGE_B_TAG="biolip_scratch_repl_transfer_${RUN_ID}"
STAGE_B_CONFIG="${PROJECT}/tmp/slurm_configs/v11m13_${STAGE_B_TAG}.yaml"
make_config transfer "${STAGE_A_CKPT}" "${STAGE_B_CONFIG}" "${STAGE_B_TAG}"
"${PYTHON}" scripts/train/train_v11m13.py --config "${STAGE_B_CONFIG}" --amp --accumulation-steps 1
STAGE_B_RUN="$(find_run "${STAGE_B_TAG}")"
STAGE_B_CKPT="${STAGE_B_RUN}/best_f1_c0_model.pt"
test -f "${STAGE_B_CKPT}"

STAGE_C_TAG="biolip_scratch_repl_continue_${RUN_ID}"
STAGE_C_CONFIG="${PROJECT}/tmp/slurm_configs/v11m13_${STAGE_C_TAG}.yaml"
make_config continue "${STAGE_B_CKPT}" "${STAGE_C_CONFIG}" "${STAGE_C_TAG}"
"${PYTHON}" scripts/train/train_v11m13.py --config "${STAGE_C_CONFIG}" --amp --accumulation-steps 1
STAGE_C_RUN="$(find_run "${STAGE_C_TAG}")"
FINAL_CKPT="${STAGE_C_RUN}/best_f1_c0_model.pt"
test -f "${FINAL_CKPT}"

LIGDEP_OUT="${RESULT_ROOT}/ligdep_v11m13_biolip_scratch_repl_${RUN_ID}/test_all"
mkdir -p "${LIGDEP_OUT}"
"${PYTHON}" scripts/infer/infer_v11_internal_ligdep.py \
  --checkpoint "${FINAL_CKPT}" --config "${STAGE_C_RUN}/config.yaml" \
  --split test --ligand_mode all --batch_size 8 --num_workers 8 \
  --output "${LIGDEP_OUT}" --amp

BENCH_OUT="${RESULT_ROOT}/v11m13_biolip_scratch_repl_benchmark_perlig_${RUN_ID}"
mkdir -p "${BENCH_OUT}"
"${PYTHON}" -u scripts/infer/infer_v11_benchmark.py \
  --checkpoint "${FINAL_CKPT}" \
  --benchmark-root "${BENCH_ROOT}" --feature-stats "${FEATURE_STATS}" \
  --datasets coach420,holo4k --tag-mode per_ligand --ligand-mode real \
  --output "${BENCH_OUT}" --batch-size 2 --num-workers 8 \
  --max-seq-len 1800 --max-atoms -1 --amp

echo "Replication completed seed=${SEED}"
echo "  pretrain=${STAGE_A_RUN}"
echo "  transfer=${STAGE_B_RUN}"
echo "  continue=${STAGE_C_RUN}"
echo "  ligdep=${LIGDEP_OUT}"
echo "  benchmark=${BENCH_OUT}"
cat "${BENCH_OUT}/coach420_summary.json"
cat "${BENCH_OUT}/holo4k_summary.json"
