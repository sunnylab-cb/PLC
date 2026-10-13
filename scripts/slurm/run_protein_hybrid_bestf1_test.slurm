#!/usr/bin/env bash
# Strict TEST of the completed FFFF/GFGF seed44 scratch runs, best F1 only.
# Task0=FFFF, task1=GFGF; one GPU/task, sequential real/shuffle/zero modes.
# Report argmax and the checkpoint's fixed VAL gate, NEVER tune on TEST.
# Test ligand atoms are not truncated (max_atoms=-1); protein cap stays1800.
# Reuse the exact training snapshot and isolated FLA+einops dependencies.
# No retraining, resume, early-stop change, benchmark expansion or ckpt search.
#SBATCH --job-name=prot_gdn_test
#SBATCH --partition=i64m1tga800ue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-1%2

set -euo pipefail
ROOT="${ROOT:-/hpc2hdd/home/shuang886/PLC/experiments/protein_hybrid_gdn_20261006}"
CODE="${ROOT}/code_v1"
PYTHON="${PYTHON:-/hpc2ssd/softwares/anaconda3/envs/af3/bin/python}"
PATTERNS=(FFFF GFGF)
RUNS=(
  protein_hybrid_FFFF_scratch_seed44_v11m13_20261008_202728
  protein_hybrid_GFGF_scratch_seed44_v11m13_20261008_202709
)
ID="${SLURM_ARRAY_TASK_ID:-${TASK_ID:-0}}"
[[ "${ID}" == 0 || "${ID}" == 1 ]] || exit 2
PATTERN="${PATTERNS[$ID]}"
RUN="${ROOT}/outputs/${RUNS[$ID]}"
CKPT="${RUN}/best_f1_c0_model.pt"
OUT="${ROOT}/test_bestf1_${SLURM_ARRAY_JOB_ID:-preflight}/${PATTERN}"
mkdir -p "${OUT}"
export PYTHONPATH="${CODE}:${ROOT}/deps:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="${ROOT}/triton_test_${SLURM_JOB_ID:-preflight}"
cd "${CODE}"

"${PYTHON}" - "${RUN}" "${OUT}" "${PATTERN}" <<'PY'
import hashlib
import json
import math
from pathlib import Path
import sys
import torch
import yaml
from scripts.diagnostics.strict_assignment_common import read_strict_splits

run, out = map(Path, sys.argv[1:3])
pattern = sys.argv[3]
checkpoint = run / 'best_f1_c0_model.pt'
ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
cfg = yaml.safe_load((run / 'config.yaml').read_text())
assert cfg['model'] == ckpt['config']['model'], 'Checkpoint/config architecture mismatch'
assert cfg['model']['protein_hybrid_pattern'] == pattern
assert cfg['training'].get('init_from_checkpoint') is None
assert cfg['training']['freeze_v11_base'] is False
gate = float(ckpt['val_contact_gate'])
assert math.isfinite(gate) and 0 <= gate <= 1
split = Path(cfg['data']['split_file'])
split_hash = hashlib.sha256(split.read_bytes()).hexdigest()
assert split_hash == '170a4e6364c19662ad922ace7957e1733409dfe146ebdb74b15c95130bb622bc'
groups = read_strict_splits(split)
assert len(groups['test']) == 647
for key in ('data_root', 'distances_file'):
    assert Path(cfg['data'][key]).exists(), key
cfg['data']['max_atoms'] = -1
cfg['model']['use_compile'] = False
cfg['model']['use_gradient_checkpointing'] = False
(out / 'eval_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
sha = hashlib.sha256()
with checkpoint.open('rb') as handle:
    for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
        sha.update(block)
provenance = dict(checkpoint=str(checkpoint), checkpoint_sha256=sha.hexdigest(),
                  selection='best_f1_c0_model.pt selected on VAL, not TEST',
                  epoch_one_based=int(ckpt['epoch']) + 1,
                  best_val_f1=float(ckpt['val_metrics']['f1_c0']), val_gate=gate,
                  split_sha256=split_hash, expected_test_n=647,
                  shuffle_seed=42, max_atoms=-1, max_seq_len=cfg['data']['max_seq_len'],
                  precision='FP16 autocast: same existing internal eval and training-validation path',
                  pattern=pattern)
(out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
print(json.dumps(provenance, indent=2), flush=True)
PY

if [[ "${PREFLIGHT_ONLY:-0}" == 1 ]]; then exit 0; fi
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
for DECISION in argmax val_gate; do
    GATE_ARGS=()
    if [[ "${DECISION}" == val_gate ]]; then GATE_ARGS=(--use_checkpoint_gate); fi
    "${PYTHON}" -u scripts/infer/infer_v11_internal_ligdep.py \
        --checkpoint "${CKPT}" --config "${OUT}/eval_config.yaml" \
        --split test --ligand_mode all --shuffle_seed 42 \
        --batch_size 4 --num_workers 4 --max_samples 0 --num_save_rows 0 --amp \
        --output "${OUT}/${DECISION}" "${GATE_ARGS[@]}"
done

"${PYTHON}" - "${OUT}" <<'PY'
import csv
import json
from pathlib import Path
import sys
from scripts.diagnostics.strict_assignment_common import read_strict_splits
import yaml

out = Path(sys.argv[1])
cfg = yaml.safe_load((out / 'eval_config.yaml').read_text())
expected = set(read_strict_splits(cfg['data']['split_file'])['test'])
provenance = json.loads((out / 'provenance.json').read_text())
donors = None
for decision in ('argmax', 'val_gate'):
    summary = json.loads((out / decision / 'ligand_dependency_summary.json').read_text())
    assert summary['n_samples'] == len(expected) == 647
    assert summary['modes'] == ['real', 'shuffle', 'zero']
    for mode in summary['modes']:
        rows = list(csv.DictReader((out / decision / f'per_sample_{mode}.csv').open()))
        assert len(rows) == len(expected) and {r['pdb_id'] for r in rows} == expected
        actual_gate = summary['summaries'][mode]['contact_threshold']
        assert actual_gate == (provenance['val_gate'] if decision == 'val_gate' else None)
        if mode == 'shuffle':
            mapping = {r['pdb_id']: r['ligand_source_pdb_id'] for r in rows}
            assert all(k != v for k, v in mapping.items())
            if donors is not None:
                assert donors == mapping, 'Different query donors across decisions'
            donors = mapping
    print(decision, json.dumps(summary['summaries']), json.dumps(summary['deltas_from_real']))
(out / 'complete.json').write_text(json.dumps(dict(complete=True, n_test=647, provenance=provenance), indent=2) + '\n')
PY
