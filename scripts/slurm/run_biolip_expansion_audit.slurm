#!/usr/bin/env bash
# Read-only BioLiP expansion assessment: stream metadata, refresh file presence,
# count type/size/known-chain/multisite ceilings. No tensor loading, extraction,
# training, new split, or homology clearance. All outputs live in an isolated root.
#SBATCH --job-name=biolip_expand_audit
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set isolated output root}"
PYTHON="/hpc2ssd/softwares/anaconda3/envs/af3/bin/python"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
echo "job=${SLURM_JOB_ID} node=${SLURMD_NODENAME} read-only metadata audit"
"${PYTHON}" -u "${EXPERIMENT}/audit_biolip_expansion.py" \
  --biolip-root /hpc2hdd/home/shuang886/biolip \
  --final-split-root /hpc2hdd/home/shuang886/biolip/pdbbind_chain_si30_split_11283646_final \
  --output "${EXPERIMENT}/summary.json"
