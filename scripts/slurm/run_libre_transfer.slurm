#!/usr/bin/env bash
# LiBRe-inspired transfer on strict merged PDBbind+BioLiP, seed44 SCRATCH.
# 0 product local-read channels; 1 same-parameter additive control;
# 2 CNC-off only; 3 sampled residue-class positive mass (lambda=.02).
# GFGF/V11m13, local54, co-attn, preprocessing, best-F1 selection unchanged.
# No new weights/downloads, no shared P-L coordinates, no compile/resume/freeze.
# First STAGE=smoke validates real B16 TRAIN data and the production loss.
# Submit training array 0-3%2 only after successful smoke, with afterok dependency.
# One GPU PER TASK, max two simultaneous tasks; all outputs on HDD.
# Slurm owns GPU allocation. No nvidia-smi check on login nodes.
#SBATCH --job-name=plc_libre_transfer
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --time=36:00:00

set -euo pipefail
EXPERIMENT="${EXPERIMENT:?Set isolated experiment root}"
CODE="${EXPERIMENT}/code_v1"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
export PYTHONPATH="${CODE}:${EXPERIMENT}/deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${EXPERIMENT}/triton_cache_${SLURM_JOB_ID:-manual}"
cd "${CODE}"
echo "Slurm job=${SLURM_JOB_ID:-none} nodes=${SLURM_JOB_NODELIST:-none} stage=${STAGE:-smoke} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"${PYTHON}" scripts/diagnostics/prepare_libre_transfer.py --experiment "${EXPERIMENT}" --verify
if [[ "${STAGE:-smoke}" == smoke ]]; then
    timeout 45m "${PYTHON}" -u scripts/diagnostics/smoke_libre_transfer.py --experiment "${EXPERIMENT}"
elif [[ "${STAGE}" == train ]]; then
    "${PYTHON}" -c 'import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); s=json.loads((p/"smoke_pass.json").read_text()); assert s["passed"] is True; assert s["provenance_sha256"] == hashlib.sha256((p/"prepared/provenance.json").read_bytes()).hexdigest()' "${EXPERIMENT}"
    VARIANTS=(plc_product_readout plc_additive_readout_control plc_cnc_off plc_residue_classmass)
    ID="${SLURM_ARRAY_TASK_ID:?Expected array 0-3}"
    [[ "${ID}" =~ ^[0-3]$ ]] || exit 2
    "${PYTHON}" -u scripts/train/train_v11m13.py \
        --config "${EXPERIMENT}/prepared/${VARIANTS[$ID]}.yaml" --amp --accumulation-steps 1
else
    echo "Unknown STAGE: ${STAGE}" >&2
    exit 2
fi
