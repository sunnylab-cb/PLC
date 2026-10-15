#!/usr/bin/env bash
# Expand the FULL legacy-eligible BioLiP pool, not merely cached features.
# STAGE=plan / ligand_ready / ligand_remaining / protein / collect / qc.
# Missing features are preparation tasks, not final sample exclusions.
# Isolated raw outputs; no training, split change or production overwrite.
#SBATCH --job-name=biolip_expand_cpu
#SBATCH --partition=i64m512ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set isolated experiment root}"
STAGE="${STAGE:?Set expansion stage}"
BIO_ROOT="${BIO_ROOT:-/hpc2hdd/home/shuang886/biolip}"
QC_POOL="${QC_POOL:-/hpc2hdd/home/shuang886/PLC/experiments/internal_si30_dual_fixed5_20261013/pdbbind_biolip_joint/pool}"
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export PATH="/hpc2ssd/softwares/anaconda3/envs/af3/bin:${PATH}"
export PYTHONPATH="${EXPERIMENT}/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export OPENMM_CPU_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/plc_mpl_${SLURM_JOB_ID}"
ARGS=(--stage "${STAGE}" --experiment "${EXPERIMENT}" --biolip-root "${BIO_ROOT}"
      --qc-pool "${QC_POOL}" --workers "${SLURM_CPUS_PER_TASK:-8}")
if [[ -n "${SMOKE_LIMIT:-}" ]]; then ARGS+=(--limit "${SMOKE_LIMIT}"); fi
"${PYTHON}" -u "${EXPERIMENT}/code/scripts/diagnostics/biolip_feature_expansion.py" "${ARGS[@]}"
