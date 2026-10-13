"""Versioned inference-only GVP reduction and explicit acceptance policies.

The context replaces a module-global callable and is for single-threaded offline
export processes only. It never changes checkpoint parameters or model files.
"""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

LEGACY = 'legacy_bf16'
STABLE = 'bf16_gvp_fp32_v1'
PROTOCOLS = (LEGACY, STABLE)
ACCEPTANCE_V1 = 'strict_fraction_v1'
ACCEPTANCE_V2 = 'integer_count_v2'
ACCEPTANCE_V3 = 'prediction_stable_v3'
ACCEPTANCE_POLICIES = (ACCEPTANCE_V1, ACCEPTANCE_V2, ACCEPTANCE_V3)
LIMITS = dict(max_probability_error=.05, max_row_disagreement=.01,
              max_mean_disagreement=.001, max_mean_f1_delta=.001,
              max_repeat_probability_error=1e-5)


def acceptance_limits(policy):
    if policy not in ACCEPTANCE_POLICIES:
        raise ValueError(f'Unknown acceptance policy: {policy}')
    result = dict(LIMITS)
    if policy in (ACCEPTANCE_V2, ACCEPTANCE_V3):
        result.update(row_binary_rule='flips <= max(1, n_full // 100)', min_row_flip_budget=1)
    if policy == ACCEPTANCE_V3:
        result['max_repeat_probability_error'] = .01
    return result


def binary_disagreement_exceeded(flips, n_full, policy=ACCEPTANCE_V1):
    acceptance_limits(policy)
    if (not isinstance(n_full, (int, np.integer)) or n_full <= 0
            or not isinstance(flips, (int, np.integer)) or not 0 <= flips <= n_full):
        raise ValueError('Invalid residue/flip counts')
    if policy in (ACCEPTANCE_V2, ACCEPTANCE_V3):
        # Integer arithmetic avoids floating-point floor errors at multiples of 100.
        return flips > max(1, n_full // 100)
    return flips / n_full > LIMITS['max_row_disagreement']


def repeat_disagreement_exceeded(max_abs, flips, policy=ACCEPTANCE_V1):
    limit = acceptance_limits(policy)['max_repeat_probability_error']
    return not np.isfinite(max_abs) or max_abs > limit or flips != 0


@contextmanager
def inference_precision(protocol):
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unknown inference precision: {protocol}')
    if protocol == LEGACY:
        yield
        return
    import src.models.gvp_layers as layers
    original = layers.scatter
    def stable_scatter(src, *args, **kwargs):
        if torch.is_grad_enabled():
            raise RuntimeError('FP32 GVP reduction option is inference-only')
        if src.dtype in (torch.bfloat16, torch.float16):
            return original(src.float(), *args, **kwargs).to(src.dtype)
        return original(src, *args, **kwargs)
    layers.scatter = stable_scatter
    try:
        yield
    finally:
        layers.scatter = original


def input_fingerprint(inputs):
    h = hashlib.sha256()
    for name, value in sorted(inputs.items()):
        if torch.is_tensor(value):
            x = value.detach().cpu().contiguous()
            h.update(json.dumps([name, list(x.shape), str(x.dtype)]).encode())
            h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
        else:
            h.update(json.dumps([name, value], sort_keys=True).encode())
    return h.hexdigest()


def source_hashes(root):
    from scripts.infer.benchmark_integrity import sha256_file
    paths = set()
    for directory in ('src/models', 'src/data', 'src/utils'):
        paths.update((root / directory).rglob('*.py'))
    paths.update(root / p for p in (
        'scripts/infer/infer_v11_internal_ligdep.py',
        'scripts/infer/benchmark_integrity.py',
        'scripts/diagnostics/export_strict_val_predictions.py',
        'scripts/diagnostics/export_site_quality_cache.py',
        'scripts/diagnostics/audit_site_precision_val.py',
        'scripts/diagnostics/prepare_internal_label_alignment.py',
        'scripts/diagnostics/site_inference_precision.py',
        'scripts/diagnostics/diagnose_site_probability_repeat.py',
        'scripts/diagnostics/strict_assignment_common.py'))
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(paths)}


def audit_acceptance(rows, expected_ids, policy=ACCEPTANCE_V1):
    """Collect all outliers under the explicitly selected acceptance version."""
    acceptance_limits(policy)
    ids = [r['sample_id'] for r in rows]
    failures = []
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids) or not rows:
        failures.append('population mismatch')
    fields = ('max_abs', 'flip_fraction_full', 'f1_delta', 'repeat_max_abs', 'repeat_flips')
    for row in rows:
        if not all(np.isfinite(row[k]) for k in fields):
            failures.append(f"{row['sample_id']}: nonfinite comparison")
            continue
        try:
            binary_bad = binary_disagreement_exceeded(row['flips'], row['n_full'], policy)
            if not np.isclose(row['flip_fraction_full'], row['flips'] / row['n_full'], rtol=0, atol=1e-12):
                raise ValueError('Fraction/count mismatch')
        except (KeyError, ValueError):
            failures.append(f"{row['sample_id']}: invalid binary counts")
            continue
        bad = []
        if row['max_abs'] > LIMITS['max_probability_error']:
            bad.append('probability')
        if binary_bad:
            bad.append('binary disagreement')
        if repeat_disagreement_exceeded(row['repeat_max_abs'], row['repeat_flips'], policy):
            bad.append('repeatability')
        if bad:
            failures.append(f"{row['sample_id']}: {', '.join(bad)}")
    if rows:
        if np.mean([r['flip_fraction_full'] for r in rows]) > LIMITS['max_mean_disagreement']:
            failures.append('mean disagreement')
        if abs(np.mean([r['f1_delta'] for r in rows])) > LIMITS['max_mean_f1_delta']:
            failures.append('mean F1 reproduction')
    return failures


def load_precision_approval(path, *, root, checkpoint_sha256, split_sha256,
                            label_overlay_sha256, reference_summary_sha256, gate, expected_ids,
                            acceptance_policy=ACCEPTANCE_V1):
    from scripts.infer.benchmark_integrity import sha256_file
    path = Path(path)
    meta = json.loads(path.read_text())
    if meta.get('acceptance_policy', ACCEPTANCE_V1) != acceptance_policy:
        raise ValueError('Precision audit mismatch: acceptance_policy')
    expected = dict(complete=True, passed=True, split='val', smoke=False,
                    inference_protocol=STABLE, checkpoint_sha256=checkpoint_sha256,
                    split_sha256=split_sha256, label_overlay_sha256=label_overlay_sha256,
                    reference_summary_sha256=reference_summary_sha256, gate=gate,
                    limits=acceptance_limits(acceptance_policy), source_hashes=source_hashes(root))
    for key, value in expected.items():
        if meta.get(key) != value:
            raise ValueError(f'Precision audit mismatch: {key}')
    manifest_path = path.parent / 'manifest.json'
    if sha256_file(manifest_path) != meta['manifest_sha256']:
        raise ValueError('Changed precision audit manifest')
    manifest = json.loads(manifest_path.read_text())
    rows = json.loads((path.parent / 'comparisons.json').read_text())
    if sha256_file(path.parent / 'comparisons.json') != meta['comparisons_sha256']:
        raise ValueError('Changed precision audit comparisons')
    if audit_acceptance(rows, expected_ids, acceptance_policy):
        raise ValueError('Precision audit numerical gates failed')
    ids = [r['sample_id'] for r in manifest]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError('Precision audit manifest population mismatch')
    result = {}
    for item in manifest:
        sid = item['sample_id']
        if Path(sid).name != sid:
            raise ValueError('Unsafe audit sample ID')
        pred = path.parent / 'predictions' / f'{sid}.npz'
        if sha256_file(pred) != item['sha256']:
            raise ValueError(f'Changed audited prediction: {sid}')
        result[sid] = dict(item, path=pred)
    return result
