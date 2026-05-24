#!/bin/bash
#SBATCH --job-name=v11m4_delta
#SBATCH --partition=i64m1tga800u
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/v11m4_delta_%j.out
#SBATCH --error=logs/v11m4_delta_%j.err

# ============================================================
# Pocket-GVP-Fusion V11m4 local pair-delta training
# ============================================================
# Submit:
#   sbatch scripts/slurm/run_v11m4_pair_delta.slurm
#
# Override config/resume if needed:
#   sbatch --export=ALL,CONFIG_FILE=configs/pocket_gvp_v11m4_hpc.yaml scripts/slurm/run_v11m4_pair_delta.slurm
#   sbatch --export=ALL,RESUME_CHECKPOINT=outputs/.../checkpoint_epoch_010.pt scripts/slurm/run_v11m4_pair_delta.slurm
#
# Matching interactive allocation:
#   srun -p i64m1tga800u -n 1 --mem=64G -c 8 --gres=gpu:1 --time=24:00:00 --pty bash
#
# Notes:
#   - Project directory is fixed for the HPC spooler checkout.
#   - Uses conda environment "af3" as requested.
#   - Runs V11m4 scratch training via configs/pocket_gvp_v11m4_hpc.yaml.
# ============================================================

PROJECT_DIR="/hpc2hdd/home/shuang886/jhspoolers/PLC"
CONFIG_FILE="${CONFIG_FILE:-configs/pocket_gvp_v11m4_hpc.yaml}"
EXPERIMENT_NOTE="${EXPERIMENT_NOTE:-v11m4_local_pair_delta}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "Project: ${PROJECT_DIR}"
echo "Config: ${CONFIG_FILE}"
echo "Experiment note: ${EXPERIMENT_NOTE}"
echo "============================================"

mkdir -p logs

# Keep module usage minimal; cluster images differ. If your site requires
# explicit modules, load them before conda activation in the submission shell.
if command -v module &> /dev/null; then
    module purge || true
fi

# Activate conda environment.
if command -v conda &> /dev/null; then
    CONDA_BASE="$(conda info --base)"
    source "${CONDA_BASE}/etc/profile.d/conda.sh"
else
    echo "ERROR: conda command not found in PATH"
    exit 1
fi
conda activate af3

cd "${PROJECT_DIR}" || exit 1

echo "Python: $(which python)"
python --version

echo "GPU devices:"
if command -v rocm-smi &> /dev/null; then
    rocm-smi
elif command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "No rocm-smi/nvidia-smi found"
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

CMD=(python scripts/train/train_v11m4.py --config "${CONFIG_FILE}" --amp)
if [ -n "${RESUME_CHECKPOINT}" ]; then
    CMD+=(--resume "${RESUME_CHECKPOINT}")
fi

echo "Command: ${CMD[*]}"
echo "============================================"

"${CMD[@]}"
EXIT_CODE=$?

echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"
echo "============================================"

exit ${EXIT_CODE}
