#!/bin/bash
#SBATCH --job-name=ablation_${VARIANT}
#SBATCH --output=outputs/ablation_%j.out
#SBATCH --error=outputs/ablation_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# ================================================================
# Ablation Study Training Script
# ================================================================
# Usage:
#   sbatch --export=VARIANT=no_gvp,SPLIT_FILE=path/to/splits.txt scripts/run_train_ablation.slurm
#
# Variants:
#   - no_gvp: Disable GVP encoder
#   - no_esm: Disable ESM2 embeddings
#   - no_physchem_early: Disable early physchem injection
#   - no_physchem_late: Disable late physchem injection
#
# To run all ablations in parallel:
#   for v in no_gvp no_esm no_physchem_early no_physchem_late; do
#     sbatch --export=VARIANT=$v,SPLIT_FILE=outputs/baseline/splits.txt scripts/run_train_ablation.slurm
#   done
# ================================================================

# Set defaults
VARIANT=${VARIANT:-no_gvp}
SPLIT_FILE=${SPLIT_FILE:-}

echo "======================================"
echo "Ablation Study: ${VARIANT}"
echo "Split file: ${SPLIT_FILE:-auto}"
echo "======================================"

# Navigate to project directory
cd /Users/zhaoj/Project/protein_ligand

# Activate environment (adjust as needed)
source venv/bin/activate

# Set config path
CONFIG_FILE="configs/ablation_${VARIANT}.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# If split file provided, create temp config with split_file set
if [ -n "$SPLIT_FILE" ]; then
    TEMP_CONFIG="/tmp/ablation_${VARIANT}_${SLURM_JOB_ID}.yaml"
    # Use yq or sed to inject split_file
    sed "s|split_file: null|split_file: \"${SPLIT_FILE}\"|g" "$CONFIG_FILE" > "$TEMP_CONFIG"
    CONFIG_FILE="$TEMP_CONFIG"
    echo "Using split file: $SPLIT_FILE"
fi

# Run training
python scripts/train.py --config "$CONFIG_FILE"

echo "======================================"
echo "Ablation ${VARIANT} complete"
echo "======================================"
