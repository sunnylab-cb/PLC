#!/usr/bin/env bash
# Extract PDBbind ESM-C 600M block-12/24/36 residue features on star.
#
# Modes:
#   smoke-short  Validate model/cache/output on one short pending sequence.
#   smoke-long   Validate the longest (5400-residue) sequence and peak VRAM.
#   shard0/1     Run one of two deterministic full-manifest shards.

set -euo pipefail

MODE="${1:?usage: $0 smoke-short|smoke-long|shard0|shard1}"
GPU="${GPU:-0}"

PYTHON="${PYTHON:-/home/husrcf/anaconda3/envs/ESMC/bin/python}"
EXTRACTOR="${EXTRACTOR:-/data/husrcf/ProtBind/PPC/benchmark/models/esmc/scripts/extract_esmc_unique.py}"
CACHE_DIR="${CACHE_DIR:-/data/husrcf/ProtBind/PPC/benchmark/models/esmc/hf_cache}"
FEATURE_ROOT="${FEATURE_ROOT:-/data/husrcf/ProtBind/PLC/features/esmc_600m_2024_12_mlc_l12_l24_l36}"
MANIFEST="${MANIFEST:-${FEATURE_ROOT}/manifest/unique_sequences.tsv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${FEATURE_ROOT}/unique}"
LOG_DIR="${LOG_DIR:-/data/husrcf/ProtBind/PLC/logs}"

mkdir -p "${OUTPUT_ROOT}" "${LOG_DIR}"

EXTRA_ARGS=()
TAG="${MODE}"
case "${MODE}" in
  smoke-short)
    EXTRA_ARGS+=(--limit 1)
    ;;
  smoke-long)
    EXTRA_ARGS+=(--min-length 5400 --max-length 5400 --limit 1)
    ;;
  shard0)
    EXTRA_ARGS+=(--shard-count 2 --shard-index 0)
    ;;
  shard1)
    EXTRA_ARGS+=(--shard-count 2 --shard-index 1)
    ;;
  *)
    echo "invalid mode: ${MODE}" >&2
    exit 2
    ;;
esac

echo "mode=${MODE} gpu=${GPU} manifest=${MANIFEST} output=${OUTPUT_ROOT}"
echo "start=$(date --iso-8601=seconds) host=$(hostname)"

export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONUNBUFFERED=1

"${PYTHON}" "${EXTRACTOR}" \
  --manifest "${MANIFEST}" \
  --output-root "${OUTPUT_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --repo-id biohub/esmc-600m-2024-12 \
  --revision e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c \
  --device cuda:0 \
  --dtype bfloat16 \
  --output-dtype float32 \
  --layers 12,24,36 \
  --hidden-normalization final_layernorm \
  --local-files-only \
  --no-use-flash-attn \
  --tag "${TAG}" \
  "${EXTRA_ARGS[@]}"

echo "end=$(date --iso-8601=seconds)"
