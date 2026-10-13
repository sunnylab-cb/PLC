#!/usr/bin/env bash
# Fixed best-F1 TRAIN/VAL generalization and teacher-confidence audit.
# One GPU job, sequential baseline + four completed LiBRe-transfer controls.
# Baseline: all 10317 TRAIN + 647 VAL; controls: common stratified 2048 TRAIN.
# Same crop/precision/eval mode on both sides; no retraining or TEST access.
# Cache baseline TRAIN real-ligand logits only for a possible self-distillation pilot.
# Frozen model source remains untouched. Slurm allocates the GPU; no login nvidia-smi.
#SBATCH --job-name=plc_train_val_gap
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00

set -euo pipefail
ROOT="${ROOT:?Set the isolated diagnostic root}"
BASE="/hpc2hdd/home/shuang886/PLC/experiments/protein_hybrid_gdn_20261006"
TRANSFER="/hpc2hdd/home/shuang886/PLC/experiments/libre_transfer_20261010"
PYTHON="/hpc2ssd/softwares/anaconda3/envs/af3/bin/python"
export PLC_FROZEN_CODE="${TRANSFER}/code_v1"
export PYTHONPATH="${PLC_FROZEN_CODE}:${TRANSFER}/deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${ROOT}/triton_${SLURM_JOB_ID}"
cd "${PLC_FROZEN_CODE}"
"${PYTHON}" scripts/diagnostics/prepare_libre_transfer.py --experiment "${TRANSFER}" --verify
"${PYTHON}" -u "${ROOT}/scripts/diagnostics/eval_train_val_gap.py" \
    --baseline-root "${BASE}" --transfer-root "${TRANSFER}" \
    --output "${ROOT}/results_${SLURM_JOB_ID}" --train-subset 2048 \
    --batch-size 8 --workers 4 --precision fp16 --save-teacher-train
