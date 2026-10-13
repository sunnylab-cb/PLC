import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnostics.site_inference_precision import (
    STABLE, LEGACY, LIMITS, inference_precision, input_fingerprint, audit_acceptance,
    load_precision_approval, ACCEPTANCE_V1, ACCEPTANCE_V2, ACCEPTANCE_V3, acceptance_limits,
    binary_disagreement_exceeded, repeat_disagreement_exceeded,
)
from scripts.infer.benchmark_integrity import sha256_file
from scripts.train.train_site_quality_probe import read_cache
from test_site_quality_probe import create_cache


def test_fp32_scatter_only_promotes_reduction_and_restores(monkeypatch):
    import src.models.gvp_layers as layers
    dtypes = []
    def scatter(src, *args, **kwargs):
        dtypes.append(src.dtype)
        return src.sum(0)
    monkeypatch.setattr(layers, 'scatter', scatter)
    with inference_precision(STABLE), torch.inference_mode():
        out = layers.scatter(torch.ones(3, 2, dtype=torch.bfloat16))
        assert out.dtype == torch.bfloat16
        layers.scatter(torch.ones(3, 2))
    assert dtypes == [torch.float32, torch.float32]
    assert layers.scatter is scatter
    with inference_precision(LEGACY):
        assert layers.scatter is scatter
    with pytest.raises(RuntimeError, match='inference-only'):
        with inference_precision(STABLE):
            layers.scatter(torch.ones(3, 2))
    assert layers.scatter is scatter
    with pytest.raises(ValueError):
        with inference_precision('unknown'):
            pass


def row(sid='a', **kwargs):
    values = dict(max_abs=.01, flip_fraction_full=0., f1_delta=0.,
                  repeat_max_abs=0., repeat_flips=0, n_full=1000)
    values.update(kwargs)
    values.setdefault('flips', round(values['flip_fraction_full'] * values['n_full']))
    return dict(sample_id=sid, **values)


def test_original_limits_all_outliers_and_aggregate_gates():
    assert LIMITS['max_probability_error'] == .05
    assert audit_acceptance([row()], ['a']) == []
    bad = audit_acceptance([row('a', max_abs=.051), row('b', max_abs=.06)], ['a', 'b'])
    assert len(bad) == 2 and all('probability' in x for x in bad)
    assert audit_acceptance([row(f1_delta=.002)], ['a']) == ['mean F1 reproduction']
    assert audit_acceptance([row(flip_fraction_full=.002)], ['a']) == ['mean disagreement']
    assert 'population mismatch' in audit_acceptance([row()], ['a', 'b'])
    assert 'population mismatch' in audit_acceptance([row(), row()], ['a'])
    assert audit_acceptance([row(repeat_max_abs=2e-5)], ['a'])
    assert audit_acceptance([row(repeat_flips=1)], ['a'])
    assert audit_acceptance([row(max_abs=float('nan'))], ['a'])


@pytest.mark.parametrize('n,budget', [(1, 1), (92, 1), (99, 1), (100, 1), (199, 1), (200, 2), (3426, 34)])
def test_v2_integer_budget_boundaries(n, budget):
    assert not binary_disagreement_exceeded(budget, n, ACCEPTANCE_V2)
    if budget < n:
        assert binary_disagreement_exceeded(budget + 1, n, ACCEPTANCE_V2)
    assert binary_disagreement_exceeded(1, n, ACCEPTANCE_V1) == (n < 100)


def test_v2_policy_is_uniform_and_keeps_other_limits():
    assert all(acceptance_limits(ACCEPTANCE_V2)[k] == v for k, v in LIMITS.items())
    rows = [row(f'sample_{i}') for i in range(20)]
    rows[0] = row('sample_0', n_full=92, flips=1, flip_fraction_full=1/92)
    ids = [r['sample_id'] for r in rows]
    assert audit_acceptance(rows, ids, ACCEPTANCE_V1) == ['sample_0: binary disagreement']
    assert audit_acceptance(rows, ids, ACCEPTANCE_V2) == []
    for changed, reason in [(dict(max_abs=.051), 'probability'),
                            (dict(repeat_max_abs=2e-5), 'repeatability'),
                            (dict(f1_delta=.1), 'mean F1 reproduction')]:
        altered = [dict(rows[0], **changed)] + rows[1:]
        assert any(reason in f for f in audit_acceptance(altered, ids, ACCEPTANCE_V2))
    assert 'mean disagreement' in audit_acceptance([rows[0]], ['sample_0'], ACCEPTANCE_V2)
    assert audit_acceptance([row(flips=1)], ['a'], ACCEPTANCE_V2) == ['a: invalid binary counts']
    with pytest.raises(ValueError):
        binary_disagreement_exceeded(1, 0, ACCEPTANCE_V2)
    with pytest.raises(ValueError):
        binary_disagreement_exceeded(-1, 92, ACCEPTANCE_V2)


def test_input_fingerprint_values_shape_dtype_and_order():
    x = dict(a=torch.tensor(1.), b=torch.ones(2, dtype=torch.bfloat16), c=None)
    assert input_fingerprint(x) == input_fingerprint(dict(reversed(list(x.items()))))
    assert input_fingerprint(x) != input_fingerprint(dict(x, a=torch.tensor(2.)))
    assert input_fingerprint(x) != input_fingerprint(dict(x, b=torch.ones(2)))
    assert input_fingerprint(x) != input_fingerprint(dict(x, b=torch.ones(1, 2, dtype=torch.bfloat16)))


def test_v3_practical_repeat_tolerance_is_shared_and_explicit():
    limits = acceptance_limits(ACCEPTANCE_V3)
    assert limits == dict(acceptance_limits(ACCEPTANCE_V2), max_repeat_probability_error=.01)
    for p in (ACCEPTANCE_V1, ACCEPTANCE_V2):
        assert repeat_disagreement_exceeded(.00481236, 0, p)
        assert audit_acceptance([row(repeat_max_abs=.00481236)], ['a'], p)
    for error in (0., .00481236, .01):
        assert not repeat_disagreement_exceeded(error, 0, ACCEPTANCE_V3)
        assert audit_acceptance([row(repeat_max_abs=error)], ['a'], ACCEPTANCE_V3) == []
    for error, flips in ((.010001, 0), (0., 1), (float('nan'), 0), (float('inf'), 0)):
        assert repeat_disagreement_exceeded(error, flips, ACCEPTANCE_V3)
        assert audit_acceptance([row(repeat_max_abs=error, repeat_flips=flips)], ['a'], ACCEPTANCE_V3)
    for changes in (dict(max_abs=.051), dict(f1_delta=.00101), dict(flip_fraction_full=.002)):
        assert audit_acceptance([row(**changes)], ['a'], ACCEPTANCE_V3)
    assert not binary_disagreement_exceeded(1, 92, ACCEPTANCE_V3)
    assert binary_disagreement_exceeded(2, 92, ACCEPTANCE_V3)


def test_cache_rejects_missing_or_mixed_row_protocol(tmp_path):
    root, split = create_cache(tmp_path)
    meta = json.loads((root/'summary.json').read_text())
    meta['inference_protocol'] = STABLE
    (root/'summary.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='Mixed inference'):
        read_cache(root, split, True)
    manifest = json.loads((root/'manifest.json').read_text())
    for item in manifest:
        path = root/item['subset']/f"{item['sample_id']}.npz"
        with np.load(path) as z:
            values = {k:z[k] for k in z.files}
        np.savez_compressed(path, **values, inference_protocol=np.array(STABLE))
        item['sha256'] = sha256_file(path)
    (root/'manifest.json').write_text(json.dumps(manifest))
    meta['manifest_sha256'] = sha256_file(root/'manifest.json')
    (root/'summary.json').write_text(json.dumps(meta))
    assert len(read_cache(root, split, True)[0]['train']) == 4
    meta['acceptance_policy'] = ACCEPTANCE_V2
    (root/'summary.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='acceptance_policy'):
        read_cache(root, split, True)


def test_approval_rejects_failed_smoke_or_changed_sources(tmp_path, monkeypatch):
    import scripts.diagnostics.site_inference_precision as module
    monkeypatch.setattr(module, 'source_hashes', lambda root: {'model.py':'same'})
    args = dict(root=tmp_path, checkpoint_sha256='ckpt', split_sha256='split',
                label_overlay_sha256='overlay', reference_summary_sha256='reference', gate=.4,
                expected_ids=['a'])
    meta = dict(complete=True, passed=True, split='val', smoke=False, inference_protocol=STABLE,
                checkpoint_sha256='ckpt', split_sha256='split', label_overlay_sha256='overlay',
                reference_summary_sha256='reference', gate=.4, limits=LIMITS, source_hashes={'model.py':'same'})
    (tmp_path/'predictions').mkdir()
    pred = tmp_path/'predictions/a.npz'
    np.savez_compressed(pred, probability=np.array([.2]))
    manifest = [dict(sample_id='a', sha256=sha256_file(pred), input_sha256='input')]
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'comparisons.json').write_text(json.dumps([row()]))
    meta.update(manifest_sha256=sha256_file(tmp_path/'manifest.json'),
                comparisons_sha256=sha256_file(tmp_path/'comparisons.json'))
    path = tmp_path/'summary.json'
    path.write_text(json.dumps(meta))
    assert load_precision_approval(path, **args)['a']['path'] == pred
    for changed in [dict(passed=False), dict(smoke=True), dict(source_hashes={'model.py':'changed'}),
                    dict(acceptance_policy=ACCEPTANCE_V2)]:
        path.write_text(json.dumps(dict(meta, **changed)))
        with pytest.raises(ValueError, match='Precision audit mismatch'):
            load_precision_approval(path, **args)
    v2 = dict(meta, acceptance_policy=ACCEPTANCE_V2, limits=acceptance_limits(ACCEPTANCE_V2))
    path.write_text(json.dumps(v2))
    assert load_precision_approval(path, **args, acceptance_policy=ACCEPTANCE_V2)['a']['path'] == pred
    v3 = dict(meta, acceptance_policy=ACCEPTANCE_V3, limits=acceptance_limits(ACCEPTANCE_V3))
    path.write_text(json.dumps(v3))
    assert load_precision_approval(path, **args, acceptance_policy=ACCEPTANCE_V3)['a']['path'] == pred
    with pytest.raises(ValueError, match='acceptance_policy'):
        load_precision_approval(path, **args, acceptance_policy=ACCEPTANCE_V2)
    path.write_text(json.dumps(meta))
    np.savez_compressed(pred, probability=np.array([.3]))
    with pytest.raises(ValueError, match='Changed audited prediction'):
        load_precision_approval(path, **args)
