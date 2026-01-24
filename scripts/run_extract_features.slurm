#!/bin/bash
#SBATCH --job-name=extract_features
#SBATCH --partition=gpu           # Adjust to your cluster's GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8         # For data loading parallelism
#SBATCH --gres=gpu:1              # Request 1 GPU for ESM inference
#SBATCH --mem=64G                 # ESM models need significant memory
#SBATCH --time=24:00:00           # Adjust based on dataset size
#SBATCH --output=logs/extract_%j.out
#SBATCH --error=logs/extract_%j.err

# ============================================================
# Batch Feature Extraction for PDBbind using pypropel
# ============================================================

# Configuration - MODIFY THESE PATHS
PROJECT_DIR="/path/to/protein_ligand"
DATA_DIR="/path/to/pdbbind"
OUTPUT_DIR="${DATA_DIR}/features"
CONDA_ENV="protein_ligand"  # Your conda environment name

# ESM model settings
ESM_MODEL="esm2_t33_650M_UR50D"
BATCH_SIZE=4  # Reduce if OOM errors occur

# Feature options
USE_DSSP=true  # Set to false if DSSP not installed

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
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}

# Verify GPU is available
echo "GPU devices:"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo ""

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

# Change to project directory
cd ${PROJECT_DIR}

# ============================================================
# Run Feature Extraction
# ============================================================

echo "Starting feature extraction..."
echo "  Data dir: ${DATA_DIR}"
echo "  Output dir: ${OUTPUT_DIR}"
echo "  ESM model: ${ESM_MODEL}"
echo ""

# Build command
CMD="python scripts/extract_features.py \
    --data_dir ${DATA_DIR} \
    --output ${OUTPUT_DIR} \
    --esm_model ${ESM_MODEL} \
    --batch_size ${BATCH_SIZE} \
    --device cuda \
    --resume"

# Add --no_dssp if disabled
if [ "${USE_DSSP}" = false ]; then
    CMD="${CMD} --no_dssp"
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
    echo "Feature extraction completed successfully!"
    echo "Output saved to: ${OUTPUT_DIR}"
    
    # Count output files
    NUM_FILES=$(ls -1 ${OUTPUT_DIR}/*.npz 2>/dev/null | wc -l)
    echo "Total feature files: ${NUM_FILES}"
else
    echo "Feature extraction failed with exit code ${EXIT_CODE}"
fi

echo "============================================"

exit ${EXIT_CODE}
