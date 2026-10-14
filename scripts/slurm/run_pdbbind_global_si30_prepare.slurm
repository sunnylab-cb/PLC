#!/usr/bin/env bash
# PDBbind ONLY: reuse verified all-pairs cache -> global-SI connected split ->
# FRESH cross-set SI audit -> new TRAIN-only normalization -> exact GFGF config.
# local SI is diagnostic only. Native heavy-atom d_min <5A and >=3 positives.
# No COACH/HOLO constraints, no joint-cohort modifications, no a128 allocation.
# Training is a separate afterok job and additionally requires GPU smoke.
#SBATCH --job-name=pdb_gsi30_prep
#SBATCH --partition=i64m512ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set new immutable experiment root}"
SOURCE="${SOURCE:?Set completed fixed5 dual-cohort experiment root (read-only)}"
BASE="${BASE:?Set exact audited GFGF reference config}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
CODE="${EXPERIMENT}/code_v1"
export PYTHONPATH="${CODE}:${EXPERIMENT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${CODE}"
exec 9>"${EXPERIMENT}/preparation.lock"
flock -n 9 || { echo "Preparation already running" >&2; exit 2; }
"${PYTHON}" -u scripts/diagnostics/run_internal_pairwise_split.py \
    --pool-dir "${SOURCE}/pdbbind_only/pool" --output-dir "${EXPERIMENT}/split" \
    --edge-rule global_only --workers "${SLURM_CPUS_PER_TASK:-64}" --chunk-size 20000 --seed 42 \
    --reuse-all-pairs "${SOURCE}/pdbbind_only/split/all_pairs" \
    --reuse-source-script "${SOURCE}/code/scripts/diagnostics/run_internal_pairwise_split.py"
"${PYTHON}" -u scripts/diagnostics/preprocess_internal_cohort.py \
    --pool-dir "${SOURCE}/pdbbind_only/pool" --split-dir "${EXPERIMENT}/split" \
    --output-dir "${EXPERIMENT}/dataset"
"${PYTHON}" -u scripts/diagnostics/prepare_internal_gfgf.py \
    --experiment "${EXPERIMENT}" --pool-dir "${SOURCE}/pdbbind_only/pool" --base "${BASE}"
