#!/usr/bin/env bash
# Diagnose failed job13402625: repeated legacy TRAIN vs batched statistics.
# Same scratch initialization, TRAIN batch, optimizer and ten updates per replay.
# Keep original speed-gate tolerances; no speed/accuracy claim or full training.
# One GPU, eight CPUs on -u. Existing training jobs/snapshots are untouched.
#SBATCH --job-name=localscan_parity
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=00:30:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set new audit root}"
REFERENCE="/hpc2hdd/home/shuang886/PLC/experiments/local_spatial_scan_ampfix_u_20261024"
FAILED="/hpc2hdd/home/shuang886/PLC/experiments/local_scan_speed_u_20261029/speed_profile.json"
SOURCE_ROOT="/hpc2hdd/home/shuang886/PLC/experiments/joint_global_si30_gfgf_20261017"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
CODE="${EXPERIMENT}/code_v1"
export PYTHONPATH="${CODE}:${SOURCE_ROOT}/deps:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd "${CODE}"
exec 9>"${EXPERIMENT}/audit.lock"
flock -n 9 || { echo "Duplicate parity audit" >&2; exit 2; }
"${PYTHON}" -u scripts/diagnostics/audit_local_scan_parity.py \
    --reference "${REFERENCE}" --failed-report "${FAILED}" \
    --output "${EXPERIMENT}/parity_audit.json"
