#!/usr/bin/env bash
# Expanded PDBbind + BioLiP, independently split from the PDBbind-only study.
# STAGE=si: exhaustive global-SI>=.30 graph, whole connected components, then
# fresh cross-set SI recalculation. Local SI is diagnostic, no external holdout.
# STAGE=prepare: new TRAIN-only stats, fixed5 labels, exact scratch GFGF config.
# Both stages fail closed; the GPU smoke/train jobs depend on their success.
# No feature regeneration, checkpoint loading, pool mutation or old stats reuse.
#SBATCH --job-name=joint_gsi30_prepare
#SBATCH --partition=i64m512ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH --time=7-00:00:00

set -euo pipefail
: "${EXPERIMENT:?}" "${POOL:?}" "${BASE:?}" "${STAGE:?}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
CODE="${EXPERIMENT}/code_v1"
export PYTHONPATH="${CODE}:${EXPERIMENT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${CODE}"
exec 9>"${EXPERIMENT}/preparation.lock"
flock -n 9 || { echo "Preparation already running" >&2; exit 2; }
case "${STAGE}" in
    si)
        "${PYTHON}" -u scripts/diagnostics/run_internal_pairwise_split.py \
            --pool-dir "${POOL}" --output-dir "${EXPERIMENT}/split" \
            --edge-rule global_only --workers "${SLURM_CPUS_PER_TASK:-64}" \
            --chunk-size 20000 --seed 42
        ;;
    prepare)
        "${PYTHON}" -u scripts/diagnostics/preprocess_internal_cohort.py \
            --pool-dir "${POOL}" --split-dir "${EXPERIMENT}/split" \
            --output-dir "${EXPERIMENT}/dataset"
        "${PYTHON}" -u scripts/diagnostics/prepare_internal_gfgf.py \
            --experiment "${EXPERIMENT}" --pool-dir "${POOL}" --base "${BASE}" \
            --cohort pdbbind_biolip_joint
        ;;
    *) echo "Unknown STAGE=${STAGE}" >&2; exit 2 ;;
esac
