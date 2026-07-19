#!/usr/bin/env bash
# Mirror the minimum PLC training stack from HPC HDD storage to Star.
#
# This script is intended to run on an HPC login host. It deliberately skips
# historical outputs, raw/duplicate feature caches, and the ESMC cache already
# present on Star. Re-running it resumes partial files and reconciles metadata.

set -euo pipefail

SOURCE_PROJECT="${SOURCE_PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
SOURCE_ARCHIVE="${SOURCE_ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
STAR_HOST="${STAR_HOST:-husrcf@10.123.4.78}"
STAR_ROOT="${STAR_ROOT:-/data1/PLC}"
STAR_KEY="${STAR_KEY:-${HOME}/.ssh/id_ed25519_star}"

BASELINE_NAME="pocket_gvp_fusion_seed44_group_rescompat_lr2e4_v11m13_20260611_205114"
BASELINE_DIR="${SOURCE_ARCHIVE}/outputs/${BASELINE_NAME}"

SSH_ARGS=(
  -i "${STAR_KEY}"
  -o IdentitiesOnly=yes
  -o ServerAliveInterval=60
  -o ServerAliveCountMax=10
)
RSYNC_ARGS=(
  -aH
  --partial
  --human-readable
  --info=progress2,stats1
  -e "ssh ${SSH_ARGS[*]}"
)

if [[ ! -f "${STAR_KEY}" ]]; then
  echo "ERROR: missing Star SSH key: ${STAR_KEY}" >&2
  exit 2
fi

ssh "${SSH_ARGS[@]}" "${STAR_HOST}" test -d "${STAR_ROOT}/data"
ssh "${SSH_ARGS[@]}" "${STAR_HOST}" test -d "${STAR_ROOT}/repo"
ssh "${SSH_ARGS[@]}" "${STAR_HOST}" mkdir -p \
  "${STAR_ROOT}/outputs/${BASELINE_NAME}" \
  "${STAR_ROOT}/logs" \
  "${STAR_ROOT}/tmp"

echo "[1/4] Syncing PLC code without generated data or history"
rsync "${RSYNC_ARGS[@]}" \
  --exclude=.git/ \
  --exclude=data/ \
  --exclude=outputs/ \
  --exclude=results/ \
  --exclude=logs/ \
  --exclude=tmp/ \
  "${SOURCE_PROJECT}/" \
  "${STAR_HOST}:${STAR_ROOT}/repo/"

echo "[2/4] Syncing the promoted V11m13 warm-start checkpoint"
rsync "${RSYNC_ARGS[@]}" \
  "${BASELINE_DIR}/best_f1_c0_model.pt" \
  "${BASELINE_DIR}/config.yaml" \
  "${BASELINE_DIR}/splits.txt" \
  "${STAR_HOST}:${STAR_ROOT}/outputs/${BASELINE_NAME}/"

echo "[3/4] Syncing core protein/ligand training features"
rsync "${RSYNC_ARGS[@]}" \
  "${SOURCE_PROJECT}/data/complete_features_v4_nma" \
  "${SOURCE_PROJECT}/data/esm2_latents" \
  "${SOURCE_PROJECT}/data/ligand_features_v5" \
  "${SOURCE_PROJECT}/data/unimol_embeddings" \
  "${SOURCE_PROJECT}/data/residue_ligand_distances_dcenter.pt" \
  "${SOURCE_PROJECT}/data/stats_ligand_esm2_unimol.pt" \
  "${SOURCE_PROJECT}/data/stats_output.pt" \
  "${STAR_HOST}:${STAR_ROOT}/data/"

echo "[4/4] Syncing normalized ESM3 features used by the optional prior arm"
rsync "${RSYNC_ARGS[@]}" \
  "${SOURCE_PROJECT}/data/esm3_latents_norm" \
  "${STAR_HOST}:${STAR_ROOT}/data/"

ssh "${SSH_ARGS[@]}" "${STAR_HOST}" ln -sfnT \
  "${STAR_ROOT}/data" "${STAR_ROOT}/repo/data"
ssh "${SSH_ARGS[@]}" "${STAR_HOST}" ln -sfnT \
  "${STAR_ROOT}/outputs" "${STAR_ROOT}/repo/outputs"
ssh "${SSH_ARGS[@]}" "${STAR_HOST}" ln -sfnT \
  "${STAR_ROOT}/logs" "${STAR_ROOT}/repo/logs"
ssh "${SSH_ARGS[@]}" "${STAR_HOST}" ln -sfnT \
  "${STAR_ROOT}/tmp" "${STAR_ROOT}/repo/tmp"

echo "PLC mirror complete: ${STAR_HOST}:${STAR_ROOT}"
