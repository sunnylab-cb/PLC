#!/bin/bash
# Recover ESM2 final-layer features from the sealed CIF-axis audit manifest.
# Reuse exact-sequence caches; long chains use the existing overlapping windows.
# Quarantined outputs never replace current V4-aligned caches or enter training.
# Set LIMIT=3 for smoke, unset for the full run; serialize smoke -> full.
#SBATCH --job-name=bio_esm_recover
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=1-00:00:00

set -euo pipefail
: "${RECOVERY_ROOT:?}" "${EXPANSION_ROOT:?}" "${CODE_ROOT:?}"
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$CODE_ROOT"
args=(--stage extract --expansion "$EXPANSION_ROOT" --output "$RECOVERY_ROOT"
      --biolip-root /hpc2hdd/home/shuang886/biolip --workers "${SLURM_CPUS_PER_TASK:-8}")
if [[ -n "${LIMIT:-}" ]]; then args+=(--limit "$LIMIT"); fi
exec "$PYTHON" -u scripts/diagnostics/biolip_esm2_recovery.py "${args[@]}"
