#!/bin/bash
# V11m15 ligand-pocket assignment auxiliary probes.
#
# Why this is being run:
#   The same-protein negative preflight showed that standalone same-protein
#   logit-region supervision is too sparse in PDBbind:
#     only ~10.9% train samples have a valid masked same-protein negative.
#   Therefore the main assignment signal is in-batch ligand-pocket embedding
#   InfoNCE, with same-protein valid negatives only used as a sparse bonus.
#
# Variants:
#   0 assign_inbatch_embed_lam005
#       residue CE + embedding InfoNCE, lambda=0.05.
#   1 assign_inbatch_embed_plus_samevalid_lam005_sp002
#       in-batch InfoNCE + sparse same-valid weighted denominator, lambda_sp=0.02.
#   2 assign_inbatch_embed_plus_pairvalid_lam005_pair002
#       in-batch InfoNCE + sparse same-valid pairwise BT bonus, lambda_pair=0.02.
#
# Training policy:
#   Warm-start from the current V11m13 group_rescompat checkpoint.
#   Freeze the heavy protein trunk; train assignment head, co-attn/local-delta,
#   local54 interaction modules, and classifier at a conservative LR=5e-5.
#   This is necessary because training only the assignment projection head would
#   not change residue logits or benchmark LAA.
#
# Post-train eval:
#   Runs COACH420/HOLO4K per-ligand benchmark with --max-atoms -1 and reports
#   sample/protein metrics, PDR, LAA, BP from infer_v11_benchmark.py.

#SBATCH --job-name=v11m15_assign
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=28:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/v11m15_assign_%A_%a.out
#SBATCH --error=logs/v11m15_assign_%A_%a.err

# Avoid `set -u`: this cluster's conda deactivate hook reads an unset
# _CONDA_PYTHON_SYSCONFIGDATA_NAME_USED variable and exits before training.
set -eo pipefail

PROJECT_DIR="${PROJECT_DIR:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
BASE_CONFIG="${BASE_CONFIG:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/config.yaml}"
INIT_CHECKPOINT="${INIT_CHECKPOINT:-outputs/pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114/best_f1_c0_model.pt}"
BENCH_ROOT="${BENCH_ROOT:-${PROJECT_DIR}/benchmark}"
FEATURE_STATS="${FEATURE_STATS:-${PROJECT_DIR}/data/stats_output.pt}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"

VARIANTS=(
  "assign_inbatch_embed_lam005"
  "assign_inbatch_embed_plus_samevalid_lam005_sp002"
  "assign_inbatch_embed_plus_pairvalid_lam005_pair002"
)

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
VARIANT="${VARIANTS[$TASK_ID]}"
if [ -z "${VARIANT}" ]; then
  echo "ERROR: invalid task id ${TASK_ID}"
  exit 2
fi

mkdir -p "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}" || exit 1

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array task: ${TASK_ID}"
echo "Variant: ${VARIANT}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Base config: ${BASE_CONFIG}"
echo "Init checkpoint: ${INIT_CHECKPOINT}"
echo "============================================"

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

if [ ! -f "${BASE_CONFIG}" ]; then
  echo "ERROR: base config not found: ${BASE_CONFIG}"
  exit 3
fi
if [ ! -f "${INIT_CHECKPOINT}" ]; then
  echo "ERROR: init checkpoint not found: ${INIT_CHECKPOINT}"
  exit 4
fi

echo "Python: $(which python)"
python --version

mkdir -p tmp/slurm_configs tmp/assignment_negatives
CONFIG_FILE="tmp/slurm_configs/v11m15_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.yaml"
ASSIGN_SUMMARY="tmp/assignment_negatives/${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.json"
ASSIGN_ARTIFACT="tmp/assignment_negatives/${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}.pt"

python - "${BASE_CONFIG}" "${CONFIG_FILE}" "${VARIANT}" "${INIT_CHECKPOINT}" "${ASSIGN_ARTIFACT}" <<'PY'
import sys
from pathlib import Path

import yaml

base_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
variant = sys.argv[3]
init_checkpoint = sys.argv[4]
artifact = Path(sys.argv[5])

with base_path.open() as f:
    cfg = yaml.safe_load(f)

cfg.setdefault("experiment", {})
cfg["experiment"]["name"] = f"pocket_gvp_fusion_seed44_{variant}"
cfg["experiment"]["version_suffix"] = "v11m15"
cfg["experiment"]["seed"] = 44

data = cfg.setdefault("data", {})
data["max_atoms"] = int(data.get("max_atoms", 128))
data["atom_truncation_policy"] = data.get("atom_truncation_policy", "alternating_ends")
data["assignment_max_negatives"] = 4
if "samevalid" in variant or "pairvalid" in variant:
    data["assignment_negatives_file"] = str(artifact)
else:
    data.pop("assignment_negatives_file", None)

model = cfg["model"]

# Preserve current group_rescompat local evidence design.
model["local54_interaction_groups"] = ["electronic3", "sasa_hbond4", "flex_rot5"]
model["local54_interaction_hidden"] = 96
model["use_local54_interaction_kv"] = True
model["use_local54_residue_chem_compat"] = True
model["use_local54_edge_context_kv"] = False
model["use_local54_chem_adapter"] = False
model["use_local54_chem_reweight"] = False
model["use_ligand_context_prior"] = False
model["ligand_context_prior_sources"] = []

model["local_delta_detach_protein"] = False
model["local_delta_protein_grad_scale"] = 0.60
model["use_global_delta_gate"] = False
model["use_gradient_checkpointing"] = False
model["use_v12_cross_attention"] = False
model["use_v12_coattention_block"] = True

model["local54_interaction_kv_alpha_init"] = 0.035
model["local54_interaction_kv_alpha_max"] = 0.20
model["local54_compat_hidden"] = 128
model["local54_compat_gate_bias_init"] = 0.01
model["local54_compat_alpha_init"] = 0.02
model["local54_compat_alpha_max"] = 0.12
model["local54_compat_normalize_read"] = True
model["local54_compat_read_scale"] = 0.35
model["local54_compat_read_clip"] = 0.0

# V11m15 assignment objective.
model["use_assignment_embed_loss"] = True
model["assignment_embed_dim"] = 128
model["assignment_temperature"] = 0.10
model["assignment_embed_weight"] = 0.05
model["assignment_mask_same_pdb_inbatch"] = True
model["assignment_samevalid_temperature"] = 0.10
model["assignment_samevalid_neg_weight"] = 3.0
model["assignment_pair_beta"] = 10.0
model["assignment_pair_topk"] = 1
if "samevalid" in variant:
    model["assignment_samevalid_mode"] = "softmax"
    model["assignment_samevalid_weight"] = 0.02
elif "pairvalid" in variant:
    model["assignment_samevalid_mode"] = "pairwise"
    model["assignment_samevalid_weight"] = 0.02
else:
    model["assignment_samevalid_mode"] = "none"
    model["assignment_samevalid_weight"] = 0.0

train = cfg.setdefault("training", {})
train["init_from_checkpoint"] = init_checkpoint
train["freeze_v11_base"] = True
train["delta_trainable_tokens"] = [
    "assignment_",
    "cross_attention_layers.",
    "local_delta_",
    "local54_anchor_proj.",
    "local54_group_proj.",
    "local54_interaction_",
    "local54_compat_",
    "classifier",
]
train["learning_rate"] = 5.0e-5
train["min_lr"] = 1.0e-6
train["warmup_epochs"] = 2
train["num_epochs"] = 50
train["batch_size"] = 24
train["accumulation_steps"] = 1
train["early_stopping_patience"] = 50
train["num_workers"] = min(int(train.get("num_workers", 8)), 8)

out_path.parent.mkdir(parents=True, exist_ok=True)
with out_path.open("w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)

print(f"Wrote config: {out_path}")
print(f"variant={variant}")
print(f"assignment_samevalid_mode={model['assignment_samevalid_mode']}")
print(f"assignment_negatives_file={data.get('assignment_negatives_file')}")
PY

if [[ "${VARIANT}" == *samevalid* || "${VARIANT}" == *pairvalid* ]]; then
  SPLIT_FILE="$(python - "${CONFIG_FILE}" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(open(sys.argv[1]))
print(cfg.get("data", {}).get("split_file", "configs/splits/splits.txt"))
PY
)"
  echo "Precomputing same-valid assignment negatives:"
  echo "  split_file=${SPLIT_FILE}"
  echo "  summary=${ASSIGN_SUMMARY}"
  echo "  artifact=${ASSIGN_ARTIFACT}"
  python scripts/diagnostics/analyze_assignment_negatives.py \
    --config "${CONFIG_FILE}" \
    --splits "${SPLIT_FILE}" \
    --split train \
    --output "${ASSIGN_SUMMARY}" \
    --save-artifact "${ASSIGN_ARTIFACT}" \
    --jaccard-thresh 0.25 \
    --centroid-min-dist 7.0 \
    --overlap-count-thresh 4
fi

echo "Generated config:"
sed -n '1,260p' "${CONFIG_FILE}"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Training command: python scripts/train/train_v11m15.py --config ${CONFIG_FILE} --amp --accumulation-steps 1"
python scripts/train/train_v11m15.py \
  --config "${CONFIG_FILE}" \
  --amp \
  --accumulation-steps 1

RUN_DIR="$(ls -dt outputs/pocket_gvp_fusion_seed44_${VARIANT}_v11m15_* 2>/dev/null | head -1 || true)"
if [ -z "${RUN_DIR}" ]; then
  echo "ERROR: could not locate output dir for ${VARIANT}"
  exit 5
fi
CKPT="${RUN_DIR}/best_f1_c0_model.pt"
if [ ! -f "${CKPT}" ]; then
  echo "ERROR: best checkpoint not found after training: ${CKPT}"
  exit 6
fi

echo "Training output dir: ${RUN_DIR}"
echo "Best checkpoint: ${CKPT}"

if [ "${RUN_BENCHMARK}" = "1" ]; then
  if [ ! -f "${FEATURE_STATS}" ]; then
    echo "ERROR: feature stats not found: ${FEATURE_STATS}"
    exit 7
  fi
  BENCH_OUT="${PROJECT_DIR}/results/v11m15_assignment_benchmark_${VARIANT}_${SLURM_JOB_ID}_${TASK_ID}"
  mkdir -p "${BENCH_OUT}"
  echo "Benchmark output: ${BENCH_OUT}"
  python -u scripts/infer/infer_v11_benchmark.py \
    --checkpoint "${CKPT}" \
    --benchmark-root "${BENCH_ROOT}" \
    --feature-stats "${FEATURE_STATS}" \
    --datasets coach420,holo4k \
    --tag-mode per_ligand \
    --ligand-mode real \
    --output "${BENCH_OUT}" \
    --batch-size 2 \
    --num-workers "${SLURM_CPUS_PER_TASK}" \
    --max-seq-len 1800 \
    --max-atoms -1 \
    --amp
  echo "Benchmark summaries:"
  cat "${BENCH_OUT}/coach420_summary.json"
  cat "${BENCH_OUT}/holo4k_summary.json"
fi

echo "Finished at: $(date)"
