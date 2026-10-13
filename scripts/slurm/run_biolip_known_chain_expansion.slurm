#!/usr/bin/env bash
# Complete only the 1,390 missing ligands whose chain sequences match current
# strict TRAIN. All outputs are isolated, unnormalized candidate features.
# This does NOT add samples to TRAIN, weaken homology rules or start a model.
#SBATCH --job-name=biolip_known_expand
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set isolated output root}"
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONPATH="${EXPERIMENT}:${PYTHONPATH:-}"
"${PYTHON}" -u "${EXPERIMENT}/scripts/diagnostics/extract_biolip_known_chain_expansion.py" \
  --audit-dir /hpc2hdd/home/shuang886/PLC/experiments/training_coverage_audit_20261013/biolip_tensor_qc \
  --biolip-root /hpc2hdd/home/shuang886/biolip \
  --output-dir "${EXPERIMENT}/prepared" --workers 8
