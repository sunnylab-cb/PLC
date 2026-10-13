#!/usr/bin/env bash
# External per-ligand COACH420/HOLO4K evaluation of FFFF vs GFGF, seed44.
# Uses the exact training snapshot and best VAL-F1 checkpoints already tested.
# Tasks: 0=FFFF/COACH, 1=GFGF/COACH, 2=FFFF/HOLO, 3=GFGF/HOLO.
# Each task runs argmax then its fixed VAL gate; never tune on benchmark labels.
# Key-aligned full protein axis (cap1800), uncapped atoms, TRAIN normalization.
# Existing audited SASA-only repair is explicit; other non-finite inputs fail.
# Reports sample/protein-balanced F1, LAA/PDR/BP and their coverage, plus QC.
# No training, checkpoint reselection, shared code changes or torch.compile.
#SBATCH --job-name=prot_gdn_bench
#SBATCH --partition=emergency_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-3%2

set -euo pipefail
ROOT="${ROOT:-/hpc2hdd/home/shuang886/PLC/experiments/protein_hybrid_gdn_20261006}"
CODE="${ROOT}/code_v1"
PROJECT=/hpc2hdd/home/shuang886/jhspoolers/PLC
BENCH_ROOT="${PROJECT}/benchmark"
FEATURE_STATS="${PROJECT}/data/stats_output.pt"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
PATTERNS=(FFFF GFGF)
DATASETS=(coach420 holo4k)
RUNS=(
  protein_hybrid_FFFF_scratch_seed44_v11m13_20261008_202728
  protein_hybrid_GFGF_scratch_seed44_v11m13_20261008_202709
)
ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
[[ "${ID}" =~ ^[0-3]$ ]] || exit 2
ARM=$((ID % 2))
PATTERN="${PATTERNS[$ARM]}"
DATASET="${DATASETS[$((ID / 2))]}"
RUN="${ROOT}/outputs/${RUNS[$ARM]}"
CKPT="${RUN}/best_f1_c0_model.pt"
OUT="${ROOT}/benchmark_bestf1_${SLURM_ARRAY_JOB_ID:-preflight}/${PATTERN}/${DATASET}"
mkdir -p "${OUT}"
export PYTHONPATH="${CODE}:${ROOT}/deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${ROOT}/triton_benchmark_${SLURM_JOB_ID:-preflight}"
cd "${CODE}"

"${PYTHON}" - "${ROOT}" "${RUN}" "${OUT}" "${PATTERN}" "${DATASET}" "${BENCH_ROOT}" "${FEATURE_STATS}" <<'PY'
import hashlib
import json
from pathlib import Path
import sys
import torch
import yaml

root, run, out = map(Path, sys.argv[1:4])
pattern, dataset = sys.argv[4:6]
bench, stats = map(Path, sys.argv[6:8])
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

prior = json.loads((root / 'test_bestf1_12954704' / pattern / 'complete.json').read_text())
assert prior['complete'] and prior['n_test'] == 647
p = prior['provenance']
assert sha(run / 'best_f1_c0_model.pt') == p['checkpoint_sha256']
assert sha(stats) == 'a6fff4c69ef936d9b9f76b704d98fb908952743883338062f2d9ab712bcc5117'
runner = root / 'code_v1/scripts/infer/infer_v11_benchmark.py'
assert sha(runner) == '1d382a6af82a1d76792226c83bc08f49d5d5bf7a2a19fcb3e32062189dca3ab2'
cfg = yaml.safe_load((run / 'config.yaml').read_text())
assert cfg['model']['protein_hybrid_pattern'] == pattern
assert not cfg['model'].get('use_esm3', False)
cfg['model']['use_compile'] = False
cfg['model']['use_gradient_checkpointing'] = False
cfg['data']['max_atoms'] = -1
tag = bench / 'residue_ligand_tags/atom_h' / f'{dataset}_per_ligand_dmin.pt'
tags = torch.load(tag, map_location='cpu', weights_only=False)
assert len(tags) > 0
for directory in ('features_fixed_conservative/protein_v4', 'esm2',
                  'ligand_features_v5', 'p2rank-datasets-fixed_atom_h'):
    assert (bench / directory / dataset).is_dir(), directory
(out / 'eval_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
provenance = dict(p, dataset=dataset, benchmark_root=str(bench),
                  tag_sha256=sha(tag), expected_rows=len(tags),
                  feature_stats_sha256=sha(stats), runner_sha256=sha(runner),
                  source_config_sha256=sha(run / 'config.yaml'),
                  precision='fp16', residue_alignment='keys', residue_coverage='protein',
                  nonfinite_policy='sasa_zero', ligand_mode='real')
(out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(json.dumps(provenance, indent=2), flush=True)
PY

COMMON=(
  --checkpoint "${CKPT}" --config "${OUT}/eval_config.yaml"
  --benchmark-root "${BENCH_ROOT}" --feature-stats "${FEATURE_STATS}"
  --datasets "${DATASET}" --tag-mode per_ligand --ligand-mode real
  --max-atoms -1 --max-seq-len 1800 --atom-truncation-policy alternating_ends
  --residue-alignment keys --residue-coverage protein
  --tag-protein-root p2rank-datasets-fixed_atom_h --nonfinite-policy sasa_zero
  --precision fp16 --batch-size 2 --num-workers 4
)
if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then
  "${PYTHON}" -u scripts/infer/infer_v11_benchmark.py "${COMMON[@]}" \
    --cpu --preflight-only --max-samples 1 --output "${OUT}/input_smoke"
  exit 0
fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
for DECISION in argmax val_gate; do
  EXTRA=()
  if [[ "${DECISION}" == val_gate ]]; then
    EXTRA=(--use-checkpoint-gate --save-predictions)
  fi
  "${PYTHON}" -u scripts/infer/infer_v11_benchmark.py "${COMMON[@]}" \
    --output "${OUT}/${DECISION}" "${EXTRA[@]}"
done

"${PYTHON}" - "${OUT}" "${BENCH_ROOT}" "${DATASET}" <<'PY'
import csv
import json
from pathlib import Path
import sys
import torch

out, bench = map(Path, sys.argv[1:3])
dataset = sys.argv[3]
p = json.loads((out / 'provenance.json').read_text())
tag = bench / 'residue_ligand_tags/atom_h' / f'{dataset}_per_ligand_dmin.pt'
expected = set(torch.load(tag, map_location='cpu', weights_only=False))
reference = None
for decision in ('argmax', 'val_gate'):
    result = out / decision
    meta = json.loads((result / 'benchmark_summary_all.json').read_text())
    assert meta['checkpoint_sha256'] == p['checkpoint_sha256']
    assert meta['feature_stats_sha256'] == p['feature_stats_sha256']
    assert meta['contact_threshold'] == (p['val_gate'] if decision == 'val_gate' else None)
    s = meta['summaries'][dataset]
    qc = s['input_integrity']
    assert qc['n_requested'] == qc['n_evaluated'] == len(expected) == p['expected_rows']
    assert qc['n_skipped'] == 0 and qc['all_evaluated_logits_finite']
    with (result / f'{dataset}_all_results.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(expected) and {r['pdb_id'] for r in rows} == expected
    axes = {r['pdb_id']: tuple(r[k] for k in ('protein_key', 'ligand_key',
            'n_residues', 'n_atoms', 'n_true_contact')) for r in rows}
    assert all(int(r['n_atoms']) == int(r['ligand_atoms_raw']) for r in rows)
    if reference is not None:
        assert axes == reference, 'Input/label axes differ between decisions'
    reference = axes
    print(decision, json.dumps(s), flush=True)
(out / 'complete.json').write_text(json.dumps(dict(complete=True, provenance=p), indent=2) + '\n')
PY
