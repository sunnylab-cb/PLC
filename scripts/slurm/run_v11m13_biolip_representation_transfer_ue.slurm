#!/usr/bin/env bash
# Expedited launcher for the BioLip representation-transfer matrix.
# The original i64m1tga800u job is intentionally retained as a fallback.

#SBATCH --job-name=v11m13_biolip_repr_ue
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=36:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_biolip_repr_ue_%A_%a.out
#SBATCH --error=/hpc2hdd/home/shuang886/jhspoolers/PLC/logs/v11m13_biolip_repr_ue_%A_%a.err

set -euo pipefail

PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
exec bash "${PROJECT}/scripts/slurm/run_v11m13_biolip_representation_transfer.slurm"
