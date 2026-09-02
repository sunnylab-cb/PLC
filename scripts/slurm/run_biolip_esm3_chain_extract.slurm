#!/usr/bin/env bash
# Extract exact-chain ESM3 features for the 5,157 audited BioLiP training pairs.
# Work is deduplicated by (PDB, chain), normalized with PDBbind ESM3 statistics,
# and linked back to every per-ligand sample only after residue-key QC passes.

#SBATCH --job-name=biolip_e3
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --array=0-3%4
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_e3_%A_%a.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/biolip_e3_%A_%a.err

set -euo pipefail
PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
PYTHON="${PYTHON:-/hpc2hdd/home/shuang886/.conda/envs/AIAA/bin/python}"
ESM3_SITE="${ESM3_SITE:-/hpc2hdd/home/shuang886/python/esm3_3.2.1}"
BIOLIP="${BIOLIP:-/hpc2hdd/home/shuang886/biolip/v11_allclean_route5_20260811}"
SPLIT_ROOT="${SPLIT_ROOT:-/hpc2hdd/home/shuang886/biolip/pdbbind_chain_si30_split_11283646_final}"
OUTPUT="${OUTPUT:-/hpc2hdd/home/shuang886/biolip/esm3_chain_si30}"
STATS="${STATS:-${PROJECT}/tmp/esm3_stats_output.pt}"
HF_CACHE="${HF_CACHE:-/hpc2hdd/home/shuang886/.cache/huggingface/hub}"

mkdir -p "${PROJECT}/logs" "${OUTPUT}"
cd "${PROJECT}"
export PYTHONPATH="${ESM3_SITE}:${PROJECT}/preparation/protein:${PROJECT}:${PYTHONPATH:-}"
export HF_HUB_CACHE="${HF_CACHE}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

test -f "${SPLIT_ROOT}/biolip_train_ids.txt"
test -f "${BIOLIP}/pilot_manifest.tsv"
test -f "${STATS}"
test -d "${HF_CACHE}/models--EvolutionaryScale--esm3-sm-open-v1"
"${PYTHON}" -c "import esm.sdk.api, torch; print('ESM3 extraction environment:', torch.__version__)"

"${PYTHON}" preparation/protein/esm3_extract_biolip_chains.py \
  --biolip-root "${BIOLIP}" \
  --ids "${SPLIT_ROOT}/biolip_train_ids.txt" \
  --stats "${STATS}" \
  --output-root "${OUTPUT}" \
  --device cuda:0 \
  --shard-id "${SLURM_ARRAY_TASK_ID}" \
  --num-shards 4
