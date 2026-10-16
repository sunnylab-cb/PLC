#!/bin/bash
# Audit failed ESM2 chains directly in existing CIFs. No GPU, downloads, V4
# overwrite or training admission. Reject missing/ambiguous residue axes.
# Run once, then submit the GPU smoke/full job with afterok dependencies.
# Required: RECOVERY_ROOT, EXPANSION_ROOT, CODE_ROOT (immutable deployment).
#SBATCH --job-name=bio_esm_recover_plan
#SBATCH --partition=i64m512ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

set -euo pipefail
: "${RECOVERY_ROOT:?}" "${EXPANSION_ROOT:?}" "${CODE_ROOT:?}"
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$CODE_ROOT"
exec "$PYTHON" -u scripts/diagnostics/biolip_esm2_recovery.py \
    --stage plan --expansion "$EXPANSION_ROOT" --output "$RECOVERY_ROOT" \
    --biolip-root /hpc2hdd/home/shuang886/biolip \
    --workers "${SLURM_CPUS_PER_TASK:-4}"
