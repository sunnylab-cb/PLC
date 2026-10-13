#!/usr/bin/env bash
# LiBRe official checkpoint inference smoke on the first 8 released test rows.
# Offline ESM2 layer33 -> official 40D ligand graph -> strict-loaded LiBRe.
# Checks finite/repeatable output, upstream CLI parity and retained EOS effects.
# This is not a full COACH420/HOLO4K benchmark and does not train any model.
#SBATCH --job-name=libre_smoke
#SBATCH --partition=emergency_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
ROOT=/hpc2hdd/home/shuang886/PLC/external_baselines/libre
REVISION=e9151a85bab2d05b67b953bc55a74c780aa2bf24
PYTHON=/hpc2ssd/softwares/anaconda3/envs/af3/bin/python
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/hpc2hdd/home/shuang886/.cache/torch
cd "${ROOT}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
"${PYTHON}" -u "${ROOT}/runners/smoke_libre_inference.py" \
  --root "${ROOT}" --revision "${REVISION}" \
  --output "${ROOT}/results/smoke_${SLURM_JOB_ID}"
