#!/usr/bin/env bash
# Diagnose 5ndf/4k7o reference drift on one scheduled GPU, without training.
# Ten unchanged-input repetitions each: FP32, BF16, BF16 with FP32 GVP scatter.
# Last condition is process-local; immutable model code and old outputs stay intact.
# Never changes the export acceptance thresholds or launches downstream training.
#SBATCH --job-name=site_prob_repeat
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:15:00

set -euo pipefail
ARCHIVE=/hpc2hdd/home/shuang886/PLC
PREVIOUS=${ARCHIVE}/experiments/strict_site_parallel_20260926
PRIOR=${ARCHIVE}/experiments/strict_assignment_20260919
DIAGNOSTIC=${ARCHIVE}/experiments/site_probability_repeat_20261004
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONPATH="${PREVIOUS}/code_pilots_v1"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
"${PYTHON}" -u "${DIAGNOSTIC}/diagnose_site_probability_repeat.py" \
  --code-root "${PREVIOUS}/code_pilots_v1" \
  --checkpoint "${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_scratch_seed44_12222454_1_v11m13_20260826_055836/best_f1_c0_model.pt" \
  --root "${ARCHIVE}/datasets/pdbbind_biolip_chain_si30_final" \
  --reference-val "${PRIOR}/val_12804287/base44" \
  --partial-cache "${PREVIOUS}/run_12869014_none/cache" \
  --output "${DIAGNOSTIC}/run_${SLURM_JOB_ID}" --ids 5ndf 4k7o --repeats 10
