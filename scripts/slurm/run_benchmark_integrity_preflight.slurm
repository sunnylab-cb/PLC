#!/usr/bin/env bash
# P0 CPU audit: keep all original ligand rows, reconstruct the label residue
# axis from the original atom+H PDB, key-align ESM2/ESM3, and record any explicit
# SASA-zero repairs. No checkpoint inference and no source cache modifications.
# The dependent GPU job must only run after this full-input audit succeeds.
#SBATCH --job-name=bench_input_qc
#SBATCH --partition=emergency_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=08:00:00

set -euo pipefail
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
EXPERIMENT="${EXPERIMENT:-${ARCHIVE}/experiments/benchmark_integrity_20260919}"
CODE="${CODE:-${EXPERIMENT}/code}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
RUN="${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_esm3gp_scratch_seed44_12620756_1_v11m13e3ft_20260903_011256"
OUT="${EXPERIMENT}/preflight_${SLURM_JOB_ID:-manual}"
cd "${CODE}"
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
"${PYTHON}" -u scripts/infer/infer_v11_benchmark.py \
  --checkpoint "${RUN}/best_f1_c0_model.pt" --config "${RUN}/config.yaml" \
  --benchmark-root "${PROJECT}/benchmark" --feature-stats "${PROJECT}/data/stats_output.pt" \
  --datasets coach420,holo4k --tag-mode per_ligand --ligand-mode real \
  --esm3-root esm3_latents_norm --max-atoms -1 --max-seq-len 1800 \
  --residue-alignment keys --residue-coverage protein --nonfinite-policy sasa_zero \
  --tag-protein-root p2rank-datasets-fixed_atom_h \
  --preflight-only --cpu --batch-size 2 --num-workers "${SLURM_CPUS_PER_TASK}" \
  --output "${OUT}"
echo "PREFLIGHT_PASSED=${OUT}"
