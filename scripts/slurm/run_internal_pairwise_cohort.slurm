#!/usr/bin/env bash
# Two INDEPENDENT internal-only cohorts, each submitted as a separate CPU job.
# COHORT=pdbbind_only: PDBbind QC -> its own all-pairs SI graph/split/stats.
# COHORT=pdbbind_biolip_joint: joint QC -> a separately recomputed graph/split/stats.
# No COACH/HOLO exclusion, no old split assignment, no representative-only SI.
# Legacy ligand chemistry QC + actual >=3 native-coordinate contact tags remain.
# Biopython global + high-coverage local alignments, fixed scores, all unique pairs.
# Full graph and fresh cross-set audit must pass before TRAIN-only normalization.
# No GPU, model training, source overwrites, downloads, or a128 CPU allocation.
# Completed SI chunks can resume by resubmitting the SAME immutable experiment.
#SBATCH --job-name=internal_si30
#SBATCH --partition=i64m512u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set isolated experiment root}"
COHORT="${COHORT:?Set pdbbind_only or pdbbind_biolip_joint}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
PDBBIND="${PDBBIND:-/hpc2hdd/home/shuang886/jhspoolers/PLC/data}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip}"
CODE="${EXPERIMENT}/code"
OUT="${EXPERIMENT}/${COHORT}"
mkdir -p "${OUT}"
exec 9>"${OUT}/pipeline.lock"
flock -n 9 || { echo "Cohort already running: ${OUT}" >&2; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "${CODE}"
BIO_ARGS=()
case "${COHORT}" in
    pdbbind_only) ;;
    pdbbind_biolip_joint) BIO_ARGS=(--biolip-root "${BIOLIP}") ;;
    *) echo "Unknown COHORT: ${COHORT}" >&2; exit 2 ;;
esac
"${PYTHON}" -u scripts/diagnostics/prepare_internal_codataset_sequences.py \
    --pdbbind-root "${PDBBIND}" \
    --pdbbind-ids preparation/dataset_split/mmseqs_splits_strict.txt \
    "${BIO_ARGS[@]}" --output-dir "${OUT}/pool" --workers 8
"${PYTHON}" -u scripts/diagnostics/run_internal_pairwise_split.py \
    --pool-dir "${OUT}/pool" --output-dir "${OUT}/split" \
    --workers "${SLURM_CPUS_PER_TASK:-64}" --chunk-size 20000 --seed 42
"${PYTHON}" -u scripts/diagnostics/preprocess_internal_cohort.py \
    --pool-dir "${OUT}/pool" --split-dir "${OUT}/split" --output-dir "${OUT}/dataset"
echo "COHORT_PREPROCESSING_COMPLETE ${COHORT}; model training requires separate smoke approval"
