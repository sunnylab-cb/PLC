#!/usr/bin/env python3
"""Audit all strict VAL rows before approving FP32 GVP accumulation exports.

Compare unchanged legacy BF16 and two mixed-precision repeats at the original
checkpoint gate. Collect all numerical outliers; nonzero exit blocks training.
Labels supply metrics only, never candidate generation or gate selection.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.export_site_quality_cache import prepared_dataset, verify_independent_states
from scripts.diagnostics.export_strict_val_predictions import (
    create_v11_model, model_needs_unimol, validate_axis, pocket_collate_fn_v10, prepare_model_inputs_v11)
from scripts.diagnostics.prepare_internal_label_alignment import load_overlay, aligned_values
from scripts.diagnostics.strict_assignment_common import read_strict_splits, write_json, write_csv, quantiles
from scripts.diagnostics.diagnose_site_probability_repeat import probability_difference
from scripts.diagnostics.site_inference_precision import (
    STABLE, LEGACY, LIMITS, inference_precision, input_fingerprint, source_hashes, audit_acceptance,
    ACCEPTANCE_V1, ACCEPTANCE_POLICIES, acceptance_limits, binary_disagreement_exceeded,
    repeat_disagreement_exceeded)
from scripts.infer.benchmark_integrity import safe_tensor_load, sha256_file, require_finite


def aggregate(rows, prefix):
    groups = {}
    for r in rows:
        groups.setdefault(r['protein_key'], []).append(r[prefix + 'f1'])
    tp = sum(r[prefix + 'tp'] for r in rows)
    size = sum(r[prefix + 'size'] for r in rows)
    true = sum(r['n_true'] for r in rows)
    return dict(sample_f1=float(np.mean([r[prefix + 'f1'] for r in rows])),
                protein_f1=float(np.mean([np.mean(x) for x in groups.values()])),
                micro_f1=2*tp/max(size+true, 1), micro_precision=tp/max(size, 1),
                micro_recall=tp/max(true, 1), predicted_counts=quantiles(r[prefix+'size'] for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--label-alignment', type=Path, required=True)
    ap.add_argument('--reference-val', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--acceptance-policy', choices=ACCEPTANCE_POLICIES, default=ACCEPTANCE_V1)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Use a scheduled GPU allocation')
    torch.set_num_threads(1)
    torch.manual_seed(44)
    np.random.seed(44)
    split = args.root / 'splits.txt'
    splits = read_strict_splits(split)
    if {k:len(v) for k,v in splits.items()} != dict(train=10317, val=647, test=647):
        raise ValueError('Unexpected strict population')
    ids = splits['val']
    overlay = load_overlay(args.label_alignment, split, ids)
    ckpt = safe_tensor_load(args.checkpoint)
    config = copy.deepcopy(ckpt['config'])
    d = config['data']
    reference_path = args.reference_val / 'export_summary.json'
    reference = json.loads(reference_path.read_text())
    gate = float(ckpt['val_contact_gate'])
    provenance = dict(checkpoint_sha256=sha256_file(args.checkpoint), split_sha256=sha256_file(split),
                      label_overlay_sha256=sha256_file(args.label_alignment), gate=gate)
    if (any(reference[k] != v for k,v in provenance.items())
            or sha256_file(d['split_file']) != provenance['split_sha256']):
        raise ValueError('Reference/checkpoint/split/label provenance mismatch')
    model, version = create_v11_model(ckpt, config, torch.device('cuda'))
    model.requires_grad_(False)
    del ckpt
    if version != 'v11m13' or model_needs_unimol(model) or any(m.training for m in model.modules()):
        raise ValueError('Expected frozen audited V11m13')
    dataset = prepared_dataset(args.root, ids, d)
    if dataset.pdb_ids != ids:
        raise ValueError('Dataset silently filtered or reordered VAL')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / 'predictions').mkdir()
    rows, manifest, independence = [], [], None
    started = time.time()
    report = dict(complete=False, passed=False, split='val', smoke=False, inference_protocol=STABLE,
                  reference_summary_sha256=sha256_file(reference_path), **provenance,
                  acceptance_policy=args.acceptance_policy,
                  limits=acceptance_limits(args.acceptance_policy), source_hashes=source_hashes(ROOT),
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), matmul_precision=torch.get_float32_matmul_precision(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  deterministic=torch.are_deterministic_algorithms_enabled())
    try:
        for i, sid in enumerate(ids):
            protein = safe_tensor_load(dataset._find_base_protein_file(sid))
            esm = safe_tensor_load(dataset._find_primary_esm_file(sid))
            distance = aligned_values(sid, protein, dataset.distances_dict[sid], overlay)
            keys, ca, distance = validate_axis(protein, esm, distance)
            target = distance.numpy() < d['label_contact_thresh']
            sample = dataset[i]
            n = sample['n_residues']
            if n != min(len(ca), len(esm['embeddings']), d['max_seq_len']) or not torch.equal(sample['gvp_coords'], ca[:n].float()):
                raise ValueError(f'Input axis mismatch: {sid}')
            for name, value in sample.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    require_finite(value, sid + '/' + name)
            inputs = prepare_model_inputs_v11(pocket_collate_fn_v10([sample]), torch.device('cuda'))
            fingerprint = input_fingerprint(inputs)
            ref_path = args.reference_val / 'predictions' / f'{sid}.npz'
            with np.load(ref_path, allow_pickle=False) as a:
                ref = {k:a[k] for k in a.files}
            observed = np.arange(len(ca)) < n
            if (not np.array_equal(target, ref['labels'] == 0)
                    or not np.array_equal(ca.numpy(), ref['ca_coords'])
                    or not np.array_equal(observed, ref['observed'])
                    or not np.array_equal(ref['residue_keys'], np.asarray(keys, dtype=str))
                    or not np.array_equal(ref['contact_prediction'], observed & (ref['contact_probability'] >= gate))):
                raise ValueError(f'Archived axes/labels/gate mismatch: {sid}')
            def forward(protocol, check_independence=False):
                nonlocal independence
                with inference_precision(protocol), torch.inference_mode():
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        result = model(**inputs)
                    for key in ('logits', 'h_prot_global', 'h_ligand_context'):
                        require_finite(result[key], sid + '/' + key)
                    if check_independence:
                        independence = verify_independent_states(model, inputs, result, args.output / 'independence.json')
                    p = np.zeros(len(ca), np.float32)
                    p[:n] = torch.softmax(result['logits'].float(), -1)[0, :n, 0].cpu().numpy()
                if input_fingerprint(inputs) != fingerprint:
                    raise ValueError(f'Model mutated inputs: {sid}')
                return p
            legacy = forward(LEGACY)
            stable = forward(STABLE, independence is None)
            repeat = forward(STABLE)
            diff = probability_difference(stable, ref['contact_probability'], observed, target, gate)
            legacy_diff = probability_difference(legacy, ref['contact_probability'], observed, target, gate)
            paired = probability_difference(stable, legacy, observed, target, gate)
            rep = probability_difference(stable, repeat, observed, target, gate)
            row = dict(sample_id=sid, protein_key=str(ref['protein_key']), n_full=len(ca), n_observed=n,
                       n_true=int(target.sum()), **diff, repeat_max_abs=rep['max_abs'], repeat_flips=rep['flips'],
                       legacy_reference_max_abs=legacy_diff['max_abs'],
                       legacy_reference_disagreement=legacy_diff['flip_fraction_full'],
                       paired_max_abs=paired['max_abs'], paired_flips=paired['flips'], paired_f1_delta=paired['f1_delta'])
            for prefix, p in (('stable_', stable), ('legacy_', legacy), ('reference_', ref['contact_probability'])):
                mask = observed & (p >= gate)
                tp, size = int((mask & target).sum()), int(mask.sum())
                row.update({prefix+'tp':tp, prefix+'size':size, prefix+'f1':2*tp/max(size+int(target.sum()), 1)})
            rows.append(row)
            path = args.output / 'predictions' / f'{sid}.npz'
            np.savez_compressed(path, probability=stable, legacy_probability=legacy, repeat_probability=repeat,
                                prediction=observed & (stable >= gate), observed=observed, target=target,
                                inference_protocol=np.array(STABLE), input_sha256=np.array(fingerprint))
            manifest.append(dict(sample_id=sid, sha256=sha256_file(path), input_sha256=fingerprint,
                                 reference_sha256=sha256_file(ref_path)))
            if (i+1) % 25 == 0:
                write_csv(args.output / 'comparisons.csv', rows)
                print(f'VAL_PRECISION_AUDIT {i+1}/{len(ids)} elapsed={time.time()-started:.1f}s', flush=True)
        failures = audit_acceptance(rows, ids, args.acceptance_policy)
        # Verify that the archived aggregate itself still reproduces from its rows.
        if abs(aggregate(rows, 'reference_')['sample_f1'] - reference['full_axis_f1']) > 1e-7:
            failures.append('archived aggregate mismatch')
        report.update(complete=True, passed=not failures, failures=failures, n=len(rows),
                      policy_comparison={p: audit_acceptance(rows, ids, p) for p in ACCEPTANCE_POLICIES},
                      independence=independence, elapsed_seconds=time.time()-started,
                      stable=aggregate(rows, 'stable_'), legacy=aggregate(rows, 'legacy_'),
                      reference=aggregate(rows, 'reference_'),
                      mean_reference_disagreement=float(np.mean([r['flip_fraction_full'] for r in rows])),
                      mean_reference_f1_delta=float(np.mean([r['f1_delta'] for r in rows])),
                      mean_paired_f1_delta=float(np.mean([r['paired_f1_delta'] for r in rows])),
                      numerical_outlier_rows=[r['sample_id'] for r in rows
                          if r['max_abs'] > LIMITS['max_probability_error']
                          or binary_disagreement_exceeded(r['flips'], r['n_full'], args.acceptance_policy)
                          or repeat_disagreement_exceeded(r['repeat_max_abs'], r['repeat_flips'], args.acceptance_policy)],
                      tolerated_repeat_drift_rows=[r['sample_id'] for r in rows
                          if r['repeat_max_abs'] > LIMITS['max_repeat_probability_error']
                          and not repeat_disagreement_exceeded(r['repeat_max_abs'], r['repeat_flips'], args.acceptance_policy)])
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        write_csv(args.output / 'comparisons.csv', rows)
        write_json(args.output / 'comparisons.json', rows)
        write_json(args.output / 'manifest.json', manifest)
        report['manifest_sha256'] = sha256_file(args.output / 'manifest.json')
        report['comparisons_sha256'] = sha256_file(args.output / 'comparisons.json')
        write_json(args.output / 'summary.json', report)
    print('VAL_PRECISION_AUDIT_COMPLETE', json.dumps(report), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
