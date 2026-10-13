#!/usr/bin/env bash
# P0 paired controls on unchanged baseline44 and ESM3-global-prior44 weights.
# One GPU, sequential evaluations, no training or test-set threshold tuning:
#   legacy_bf16: positional baseline, explicit SASA repair, BF16;
#   keys_legacy_budget: same residue counts, fix feature alignment/missing masks;
#   keys_protein: restore the full canonical protein axis up to 1800 residues.
# All arms keep the same ligand-row keys. Only the final arm exports probability
# maps for the later P1 candidate-pocket diagnostic; no oracle selection here.
# Submit with --dependency=afterok:<full CPU preflight job>.
#SBATCH --job-name=bench_fix_s44
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=12:00:00

set -euo pipefail
ARCHIVE="${ARCHIVE:-/hpc2hdd/home/shuang886/PLC}"
EXPERIMENT="${EXPERIMENT:-${ARCHIVE}/experiments/benchmark_integrity_20260919}"
CODE="${CODE:-${EXPERIMENT}/code}"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
PROJECT="${PROJECT:-/hpc2hdd/home/shuang886/jhspoolers/PLC}"
OUT="${EXPERIMENT}/eval_${SLURM_JOB_ID:-manual}"
RUNS=(
  "${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_scratch_seed44_12222454_1_v11m13_20260826_055836"
  "${ARCHIVE}/outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_esm3gp_scratch_seed44_12620756_1_v11m13e3ft_20260903_011256"
)
NAMES=(base44 esm3gp44)
cd "${CODE}"
export PYTHONPATH="${CODE}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
for i in 0 1; do
  RUN="${RUNS[$i]}"
  ESM_ARGS=()
  if [[ "$i" == 1 ]]; then ESM_ARGS=(--esm3-root esm3_latents_norm); fi
  for ARM in legacy_bf16 keys_legacy_budget keys_protein; do
    ALIGN=keys
    COVERAGE=legacy_budget
    EXTRA=()
    if [[ "${ARM}" == legacy_bf16 ]]; then ALIGN=legacy; fi
    if [[ "${ARM}" == keys_protein ]]; then COVERAGE=protein; EXTRA=(--save-predictions); fi
    "${PYTHON}" -u scripts/infer/infer_v11_benchmark.py \
      --checkpoint "${RUN}/best_f1_c0_model.pt" --config "${RUN}/config.yaml" \
      --benchmark-root "${PROJECT}/benchmark" --feature-stats "${PROJECT}/data/stats_output.pt" \
      --datasets coach420,holo4k --tag-mode per_ligand --ligand-mode real \
      --use-checkpoint-gate --precision bf16 --max-atoms -1 --max-seq-len 1800 \
      --residue-alignment "${ALIGN}" --residue-coverage "${COVERAGE}" \
      --nonfinite-policy sasa_zero --tag-protein-root p2rank-datasets-fixed_atom_h \
      --batch-size 2 --num-workers "${SLURM_CPUS_PER_TASK}" \
      --output "${OUT}/${NAMES[$i]}/${ARM}" "${ESM_ARGS[@]}" "${EXTRA[@]}"
  done
done
echo "EVALUATION_COMPLETE=${OUT}"
