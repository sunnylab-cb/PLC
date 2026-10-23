#!/usr/bin/env bash
# Strict VAL ONLY: frozen joint GFGF seed44 epoch25, full vs all-chain inference.
# Retain production TRAIN-normalized features, protein cap1800/ligand cap256.
# Same-axis global Top20; F1/MCC at0.5 and unchanged VAL gate; AP/AUROC.
# Stratify PDBbind/BioLiP, single/multichain, and cross-chain contact labels.
# No retraining, gate search, TEST/HOLO forwards, or automatic policy selection.
# CPU prepare -> one UE GPU inference -> CPU summary, linked by afterok.
#SBATCH --job-name=joint_val_chain
#SBATCH --partition=i64m512ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00

set -euo pipefail
: "${EVAL_ROOT:?}" "${PHASE:?}"
TRAIN=/hpc2hdd/home/shuang886/PLC/experiments/joint_global_si30_gfgf_20261017
CKPT="$TRAIN/outputs/pdbbind_biolip_joint_global_si30_GFGF_scratch_seed44_v11m13_20261018_233115/best_f1_c0_model.pt"
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export PYTHONPATH="${EVAL_ROOT}/code_v1:${TRAIN}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${EVAL_ROOT}/triton_${SLURM_JOB_ID}"
cd "${EVAL_ROOT}/code_v1"
exec 9>"${EVAL_ROOT}/${PHASE}.lock"
flock -n 9 || exit 2
if [[ "${PHASE}" != infer ]]; then export CUDA_VISIBLE_DEVICES=''; fi
"${PYTHON}" -u scripts/diagnostics/eval_joint_val_chain_context.py "${PHASE}" \
  --work "${EVAL_ROOT}/results" --training "$TRAIN" --checkpoint "$CKPT" --workers 4
