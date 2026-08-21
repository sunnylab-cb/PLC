#!/usr/bin/env bash
# Continue the BioLip scratch-full PDBbind transfer and benchmark the result.
#
# Motivation:
#   The first scratch-full transfer was stopped by val-loss patience at epoch
#   22 while val F1/AUPRC were still improving. This run starts a fresh
#   optimizer/scheduler from that run's best-F1 weights, trains all parameters
#   for a full convergence window, then runs:
#     1. PDBbind test real/shuffle/zero ligand dependency
#     2. COACH420/HOLO4K strict per-ligand benchmark with no atom cap

#SBATCH --job-name=v11m13_bio_cont
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_biolip_scratch_continue_%j.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_biolip_scratch_continue_%j.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
SOURCE_RUN="${SOURCE_RUN:-${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_biolip_scratch_full_pdbbind_finetune_10530083_1_v11m13_20260821_145944}"
SOURCE_CONFIG="${SOURCE_CONFIG:-${SOURCE_RUN}/config.yaml}"
SOURCE_CHECKPOINT="${SOURCE_CHECKPOINT:-${SOURCE_RUN}/best_f1_c0_model.pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ARCHIVE}/outputs}"
RESULT_ROOT="${RESULT_ROOT:-${ARCHIVE}/results}"
BENCH_ROOT="${BENCH_ROOT:-${PROJECT}/benchmark}"
FEATURE_STATS="${FEATURE_STATS:-${PROJECT}/data/stats_output.pt}"
RUN_ID="${SLURM_JOB_ID:-manual}"
RUN_TAG="biolip_scratch_full_pdbbind_continue_${RUN_ID}"
CONFIG="${PROJECT}/tmp/slurm_configs/v11m13_${RUN_TAG}.yaml"

mkdir -p "${PROJECT}/logs" "${PROJECT}/tmp/slurm_configs" "${OUTPUT_ROOT}" "${RESULT_ROOT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

for required in \
  "${SOURCE_CONFIG}" \
  "${SOURCE_CHECKPOINT}" \
  "${FEATURE_STATS}" \
  "${BENCH_ROOT}/residue_ligand_tags/atom_h/coach420_per_ligand_dmin.pt" \
  "${BENCH_ROOT}/residue_ligand_tags/atom_h/holo4k_per_ligand_dmin.pt"; do
  if [ ! -e "${required}" ]; then
    echo "ERROR: required input missing: ${required}" >&2
    exit 2
  fi
done

"${PYTHON}" - "${SOURCE_CONFIG}" "${CONFIG}" "${SOURCE_CHECKPOINT}" "${OUTPUT_ROOT}" "${RUN_TAG}" <<'PY'
import sys
from pathlib import Path

import yaml

source_config, output_config, checkpoint, output_root, run_tag = sys.argv[1:]
with Path(source_config).open() as handle:
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
model["use_unimol_ligand_input"] = False
model["allow_zero_unimol_fallback"] = False

train = cfg["training"]
train.update(
    {
        # Initialize weights only. A fresh optimizer/scheduler avoids carrying
        # the exhausted cosine schedule and val-loss patience from stage one.
        "init_from_checkpoint": checkpoint,
        "freeze_v11_base": False,
        "learning_rate": 2.0e-5,
        "min_lr": 5.0e-7,
        "warmup_epochs": 3,
        "num_epochs": 80,
        "early_stopping_patience": 80,
        "batch_size": 16,
        "accumulation_steps": 1,
        "num_workers": 8,
        "save_every": 10,
        "use_bf16": True,
        "use_amp": False,
    }
)
train.pop("delta_trainable_tokens", None)

out = Path(output_config)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w") as handle:
    yaml.safe_dump(cfg, handle, sort_keys=False)

print(
    f"config={out} init={checkpoint} freeze={train['freeze_v11_base']} "
    f"lr={train['learning_rate']} epochs={train['num_epochs']} "
    f"patience={train['early_stopping_patience']}"
)
PY

if [ "${CONFIG_ONLY:-0}" = "1" ]; then
  sed -n '1,300p' "${CONFIG}"
  exit 0
fi

echo "Stage A: continue full-model PDBbind fine-tuning"
"${PYTHON}" scripts/train/train_v11m13.py \
  --config "${CONFIG}" \
  --amp \
  --accumulation-steps 1

RUN_DIR="$(find "${OUTPUT_ROOT}" -maxdepth 1 -type d \
  -name "*${RUN_TAG}*v11m13*" -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)"
CHECKPOINT="${RUN_DIR}/best_f1_c0_model.pt"
if [ -z "${RUN_DIR}" ] || [ ! -f "${CHECKPOINT}" ]; then
  echo "ERROR: continued best-F1 checkpoint not found for ${RUN_TAG}" >&2
  exit 3
fi

echo "Stage B: PDBbind test ligand dependency"
LIGDEP_OUT="${RESULT_ROOT}/ligdep_v11m13_biolip_scratch_continue_${RUN_ID}/test_all"
mkdir -p "${LIGDEP_OUT}"
"${PYTHON}" scripts/infer/infer_v11_internal_ligdep.py \
  --checkpoint "${CHECKPOINT}" \
  --config "${RUN_DIR}/config.yaml" \
  --split test \
  --ligand_mode all \
  --batch_size 8 \
  --num_workers 8 \
  --output "${LIGDEP_OUT}" \
  --amp

echo "Stage C: COACH420/HOLO4K strict per-ligand benchmark"
BENCH_OUT="${RESULT_ROOT}/v11m13_biolip_scratch_continue_benchmark_perlig_${RUN_ID}"
mkdir -p "${BENCH_OUT}"
"${PYTHON}" -u scripts/infer/infer_v11_benchmark.py \
  --checkpoint "${CHECKPOINT}" \
  --benchmark-root "${BENCH_ROOT}" \
  --feature-stats "${FEATURE_STATS}" \
  --datasets coach420,holo4k \
  --tag-mode per_ligand \
  --ligand-mode real \
  --output "${BENCH_OUT}" \
  --batch-size 2 \
  --num-workers 8 \
  --max-seq-len 1800 \
  --max-atoms -1 \
  --amp

echo "Completed BioLip scratch continuation"
echo "  run=${RUN_DIR}"
echo "  checkpoint=${CHECKPOINT}"
echo "  ligand_dependency=${LIGDEP_OUT}"
echo "  benchmark=${BENCH_OUT}"
cat "${BENCH_OUT}/coach420_summary.json"
cat "${BENCH_OUT}/holo4k_summary.json"
