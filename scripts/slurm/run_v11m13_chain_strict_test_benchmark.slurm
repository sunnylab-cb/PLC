#!/usr/bin/env bash
# Evaluate the two seed44 V11m13 scratch controls trained on the audited
# chain-level SI<0.30 split.
#
# Array tasks:
#   0: PDBbind-only strict training control
#   1: PDBbind + sequence-clean BioLiP training control
#
# Each task runs, in order:
#   1. Strict PDBbind test real/shuffle/zero ligand-dependency evaluation.
#   2. COACH420 and HOLO4K strict per-ligand benchmark evaluation.
#
# Both evaluations report legacy three-class argmax and the contact gate saved
# from the corresponding validation checkpoint. Test/benchmark ligand atoms are
# not truncated (max_atoms=-1); ligand V5 normalization uses training stats.

#SBATCH --job-name=v11m13_si30_eval
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_si30_eval_%A_%a.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_si30_eval_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
BENCH_ROOT="${BENCH_ROOT:-${PROJECT}/benchmark}"
FEATURE_STATS="${FEATURE_STATS:-${PROJECT}/data/stats_output.pt}"
RESULT_ROOT="${RESULT_ROOT:-${ARCHIVE}/results/v11m13_chain_si30_eval_${SLURM_ARRAY_JOB_ID:-manual}}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

RUN_NAMES=(
  "pdbbind_only"
  "pdbbind_biolip"
)

RUN_DIRS=(
  "${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_pdbonly_scratch_seed44_12222454_0_v11m13_20260826_043149"
  "${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_scratch_seed44_12222454_1_v11m13_20260826_055836"
)

if (( TASK_ID < 0 || TASK_ID >= ${#RUN_NAMES[@]} )); then
  echo "ERROR: invalid array task ${TASK_ID}" >&2
  exit 2
fi

RUN_NAME="${RUN_NAMES[$TASK_ID]}"
RUN_DIR="${RUN_DIRS[$TASK_ID]}"
CHECKPOINT="${RUN_DIR}/best_f1_c0_model.pt"
SOURCE_CONFIG="${RUN_DIR}/config.yaml"
EVAL_CONFIG="${PROJECT}/tmp/slurm_configs/v11m13_chain_si30_eval_${RUN_NAME}_${SLURM_ARRAY_JOB_ID:-manual}.yaml"
RUN_OUT="${RESULT_ROOT}/${RUN_NAME}"

mkdir -p "${PROJECT}/logs" "${PROJECT}/tmp/slurm_configs" "${RUN_OUT}"
cd "${PROJECT}"
export PYTHONPATH="${PROJECT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

for required in \
  "${CHECKPOINT}" \
  "${SOURCE_CONFIG}" \
  "${FEATURE_STATS}" \
  "${BENCH_ROOT}/residue_ligand_tags/atom_h/coach420_per_ligand_dmin.pt" \
  "${BENCH_ROOT}/residue_ligand_tags/atom_h/holo4k_per_ligand_dmin.pt"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: required input missing: ${required}" >&2
    exit 3
  fi
done

# Keep the trained architecture and strict split, changing only the inference
# atom cap. This implements the agreed no-truncation test/benchmark policy.
"${PYTHON}" - "${SOURCE_CONFIG}" "${EVAL_CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

source, output = map(Path, sys.argv[1:])
config = yaml.safe_load(source.read_text())
config["data"]["max_atoms"] = -1
config["data"]["atom_truncation_policy"] = "alternating_ends"
config["model"]["use_compile"] = False
config["model"]["use_gradient_checkpointing"] = False

split_path = Path(config["data"]["split_file"])
data_root = Path(config["data"]["data_root"])
if not split_path.exists():
    raise FileNotFoundError(f"strict split missing: {split_path}")
if not data_root.exists():
    raise FileNotFoundError(f"data root missing: {data_root}")
if config["training"].get("init_from_checkpoint") is not None:
    raise RuntimeError("expected scratch-trained checkpoint provenance")
if config["training"].get("freeze_v11_base") is not False:
    raise RuntimeError("expected fully trainable scratch checkpoint provenance")

output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(yaml.safe_dump(config, sort_keys=False))
print(f"eval_config={output}")
print(f"data_root={data_root}")
print(f"strict_split={split_path}")
print("max_atoms=-1")
PY

echo "============================================================"
echo "Run: ${RUN_NAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Config: ${EVAL_CONFIG}"
echo "Output: ${RUN_OUT}"
echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Start: $(date)"
echo "============================================================"

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "Preflight completed for ${RUN_NAME}; evaluation was not launched."
  exit 0
fi

for DECISION in argmax val_gate; do
  LIGDEP_GATE_ARGS=()
  BENCH_GATE_ARGS=()
  if [[ "${DECISION}" == "val_gate" ]]; then
    LIGDEP_GATE_ARGS=(--use_checkpoint_gate)
    BENCH_GATE_ARGS=(--use-checkpoint-gate)
  fi

  LIGDEP_OUT="${RUN_OUT}/strict_test_${DECISION}"
  mkdir -p "${LIGDEP_OUT}"
  echo "Strict test ligand dependency: ${DECISION}"
  "${PYTHON}" scripts/infer/infer_v11_internal_ligdep.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${EVAL_CONFIG}" \
    --split test \
    --ligand_mode all \
    --batch_size 8 \
    --num_workers 8 \
    --max_samples 0 \
    --num_save_rows 1 \
    --output "${LIGDEP_OUT}" \
    --amp \
    "${LIGDEP_GATE_ARGS[@]}"

  BENCH_OUT="${RUN_OUT}/benchmark_per_ligand_${DECISION}"
  mkdir -p "${BENCH_OUT}"
  echo "COACH420/HOLO4K benchmark: ${DECISION}"
  "${PYTHON}" -u scripts/infer/infer_v11_benchmark.py \
    --checkpoint "${CHECKPOINT}" \
    --config "${EVAL_CONFIG}" \
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
    --atom-truncation-policy alternating_ends \
    --amp \
    "${BENCH_GATE_ARGS[@]}"

  echo "${RUN_NAME} ${DECISION} strict-test summary"
  cat "${LIGDEP_OUT}/ligand_dependency_summary.json"
  echo "${RUN_NAME} ${DECISION} COACH420 summary"
  cat "${BENCH_OUT}/coach420_summary.json"
  echo "${RUN_NAME} ${DECISION} HOLO4K summary"
  cat "${BENCH_OUT}/holo4k_summary.json"
done

echo "Completed ${RUN_NAME} at $(date)"
