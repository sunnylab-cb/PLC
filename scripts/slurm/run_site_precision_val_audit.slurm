#!/usr/bin/env bash
# Full 647-row strict VAL audit before restarting site-head trials.
# Compare original BF16, BF16+FP32 GVP reduction, and a repeat of the latter.
# Explicit policy: v2 integer counts; v3 also tolerates probability drift <=0.01.
# v3 still requires identical repeat binary predictions and stable aggregate F1.
# A nonzero exit blocks the smoke/export/head dependency chain. No TEST access.
#SBATCH --job-name=site_precision_val
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:45:00

set -euo pipefail
ARCHIVE=/hpc2hdd/home/shuang886/PLC
EXPERIMENT=${EXPERIMENT:-${ARCHIVE}/experiments/strict_site_precision_20261005}
CODE=${CODE:-${EXPERIMENT}/code_v1}
PRIOR=${ARCHIVE}/experiments/strict_assignment_20260919
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export PYTHONPATH="${CODE}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${CODE}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
"${PYTHON}" -u scripts/diagnostics/audit_site_precision_val.py \
  --checkpoint "${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_scratch_seed44_12222454_1_v11m13_20260826_055836/best_f1_c0_model.pt" \
  --root "${ARCHIVE}/datasets/pdbbind_biolip_chain_si30_final" \
  --label-alignment "${PRIOR}/label_alignment_verified.json" \
  --reference-val "${PRIOR}/val_12804287/base44" \
  --output "${EXPERIMENT}/audit_${SLURM_JOB_ID}" \
  --acceptance-policy "${ACCEPTANCE_POLICY:-strict_fraction_v1}"
