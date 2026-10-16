#!/usr/bin/env bash
# Resume only unfinished BioLiP ligand features with 64 single-thread workers.
# Leave the predecessor running while this job queues. Once allocated, validate
# its identity and the rewired QC dependency, stop it, then wait for its lock.
# The original immutable extractor is reused; chemistry and outputs do not change.
# Submit held, set QC to afterok:<new_job>, then release QC and this job.
# Required exports: EXPERIMENT, TAKEOVER_CODE_ROOT, OLD_JOB_ID, QC_JOB_ID.
#SBATCH --job-name=bio_ligand_resume64
#SBATCH --partition=i64m512ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=448G
#SBATCH --time=2-00:00:00

set -euo pipefail
: "${EXPERIMENT:?}" "${TAKEOVER_CODE_ROOT:?}" "${OLD_JOB_ID:?}" "${QC_JOB_ID:?}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPENMM_CPU_THREADS=1 PYTHONUNBUFFERED=1
exec /hpc2ssd/softwares/anaconda3/envs/af3/bin/python -u \
    "${TAKEOVER_CODE_ROOT}/scripts/diagnostics/biolip_ligand_takeover.py"
