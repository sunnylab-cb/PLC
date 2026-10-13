#!/usr/bin/env bash
# Full LiBRe external comparison, NOT training or the eight-example smoke test.
# Array 0: COACH420, 1: HOLO4K; one GPU each, official frozen checkpoint.
# Fresh raw ESM2 per chain, official 40D SMILES graph, fixed threshold 0.6.
# Match frozen PLC sample/residue axes, verify 5A labels and report F1/AP/AUROC
# plus protein-balanced and conditional BP / centroid PDR-LAA diagnostics.
# No oracle binding-chain selection, no atom cap, no silent 1500 truncation.
# Long chains use overlapping windows; results are a documented adaptation.
#SBATCH --job-name=libre_bench
#SBATCH --partition=i64m1tga800ue
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00

set -euo pipefail
ROOT=${LIBRE_ROOT:-/hpc2hdd/home/shuang886/PLC/external_baselines/libre}
REVISION=e9151a85bab2d05b67b953bc55a74c780aa2bf24
PYTHON=${LIBRE_PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}
BENCH=${LIBRE_BENCHMARK_ROOT:-/hpc2hdd/home/shuang886/jhspoolers/PLC/benchmark}
REFERENCE=${LIBRE_REFERENCE_ROOT:-/hpc2hdd/home/shuang886/PLC/experiments/protein_hybrid_gdn_20261006/benchmark_bestf1_12956268/GFGF}
DATASETS=(coach420 holo4k)
DATASET=${DATASETS[${SLURM_ARRAY_TASK_ID}]}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/hpc2hdd/home/shuang886/.cache/torch
cd "${ROOT}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
"${PYTHON}" -u "${ROOT}/runners/benchmark_v2/run_libre_benchmark.py" \
  --root "${ROOT}" --revision "${REVISION}" --dataset "${DATASET}" \
  --benchmark-root "${BENCH}" \
  --reference-csv "${REFERENCE}/${DATASET}/val_gate/${DATASET}_all_results.csv" \
  --output "${ROOT}/results/benchmark_${SLURM_ARRAY_JOB_ID}/${DATASET}"
