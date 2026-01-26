#!/bin/bash
#SBATCH --job-name=gvp_train
#SBATCH --partition=i64m1tga40ue           # Adjust to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # For data loading parallelism
#SBATCH --gres=gpu:1              # Request 1 GPU
#SBATCH --mem=40G                 # RAM
#SBATCH --time=48:00:00           # Adjust based on dataset size
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# ============================================================
# Training Script for Pocket-GVP-Fusion v2
# ============================================================

# Configuration - MODIFY THESE PATHS
PROJECT_DIR="/hpc2hdd/home/jzhao908/protein-ligand-PoC"
CONFIG_FILE="configs/pocket_gvp_v2.yaml"
# CONDA_ENV="protein_ligand"  # Your conda environment name

# Options
# RESUME_CHECKPOINT="outputs/pocket_gvp_fusion_v2_20260124_205624/checkpoints/latest.pt"  # Set to path like "outputs/experiment/checkpoints/latest.pt" to resume
RESUME_CHECKPOINT=""  # Set to path like "outputs/experiment/checkpoints/latest.pt" to resume

# ============================================================
# Setup
# ============================================================

echo "============================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Start time: $(date)"
echo "============================================"

# Create logs directory if needed
mkdir -p logs

# Load modules (adjust for your cluster)
module purge
module load cuda/11.8  # Or your CUDA version
module load anaconda3  # Or your python module

# Activate conda environment
# Try standard conda init location
# if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
#     source $(conda info --base)/etc/profile.d/conda.sh
# else
#     # Fallback to direct activation if source script not found
#     echo "Warning: conda.sh not found, trying explicit source..."
# fi
# conda activate ${CONDA_ENV} || echo "Conda activate failed, assuming environment already active"
source venv/bin/activate

# Verify GPU is available
echo "GPU devices:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "nvidia-smi not found"
fi
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Change to project directory
cd ${PROJECT_DIR}

# ============================================================
# Run Training
# ============================================================

echo "Starting training..."
echo "  Config: ${CONFIG_FILE}"
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "  Resuming from: ${RESUME_CHECKPOINT}"
fi
echo ""

# Build command
CMD="python scripts/train.py --config ${CONFIG_FILE} --device cuda --debug"

# Add resume flag if set
if [ -n "${RESUME_CHECKPOINT}" ]; then
    CMD="${CMD} --resume ${RESUME_CHECKPOINT}"
fi

echo "Command: ${CMD}"
echo ""

# Execute
${CMD}

EXIT_CODE=$?

# ============================================================
# Summary
# ============================================================

echo ""
echo "============================================"
echo "End time: $(date)"
echo "Exit code: ${EXIT_CODE}"

if [ ${EXIT_CODE} -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code ${EXIT_CODE}"
fi

echo "============================================"

exit ${EXIT_CODE}
