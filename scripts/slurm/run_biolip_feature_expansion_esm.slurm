#!/usr/bin/env bash
# Fill ESM2-650M embeddings for exact V4 residue axes, deduplicating sequences.
# esm_ready uses existing V4; esm_new follows missing-protein extraction.
# Local checkpoint only, no download; one allocated GPU, no residue truncation.
#SBATCH --job-name=biolip_expand_esm
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set isolated experiment root}"
STAGE="${STAGE:?Set esm_ready or esm_new}"
BIO_ROOT="${BIO_ROOT:-/hpc2hdd/home/shuang886/biolip}"
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export PATH="/hpc2ssd/softwares/anaconda3/envs/af3/bin:${PATH}"
export PYTHONPATH="${EXPERIMENT}/code:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
ARGS=(--stage "${STAGE}" --experiment "${EXPERIMENT}" --biolip-root "${BIO_ROOT}"
      --workers "${SLURM_CPUS_PER_TASK:-8}")
if [[ -n "${SMOKE_LIMIT:-}" ]]; then ARGS+=(--limit "${SMOKE_LIMIT}"); fi
"${PYTHON}" -u "${EXPERIMENT}/code/scripts/diagnostics/biolip_feature_expansion.py" "${ARGS[@]}"
