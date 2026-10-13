import json
from contextlib import nullcontext
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnostics.site_quality_features import candidate_stats, pack_candidates, STAT_NAMES
from scripts.diagnostics.audit_train_multisite_union import complete_groups, geometry_matches, audit_groups
from scripts.diagnostics.strict_assignment_common import write_json
from scripts.infer.benchmark_integrity import sha256_file
from scripts.train.train_site_parallel import main, load_tokens, forward_row, render_refined
from scripts.train.train_site_quality_probe import read_cache
from src.models.site_parallel_heads import PairedQualityHead, TokenQualityHead, BoundaryResidualHead, gain_loss
from test_site_quality_probe import create_cache
from scripts.diagnostics.export_site_quality_cache import verify_independent_states


def token_cache(tmp_path):
    root, split = create_cache(tmp_path)
    rng = np.random.default_rng(2)
    manifest = json.loads((root/'manifest.json').read_text())
    ca = np.zeros((8, 3), np.float32)
    ca[:, 0] = np.arange(8)*2
    p = np.full(8, .1, np.float32)
    p[[0, 1, 4, 5]] = .7
    observed = np.ones(8, bool)
    masks = [np.array([0, 1, 4, 5]), np.arange(4), np.array([5, 6, 7])]
    idx, off = pack_candidates(masks)
    for item in manifest:
        path = root/item['subset']/f"{item['sample_id']}.npz"
        with np.load(path) as z:
            data = {k: z[k] for k in z.files}
        data.update(residue_tokens=rng.normal(size=(8, 16)).astype(np.float16),
                    atom_tokens=rng.normal(size=(4, 16)).astype(np.float16), ca=ca,
                    probability=p, observed=observed, contact_target=np.arange(8)<4,
                    candidate_indices=idx, candidate_offsets=off,
                    stats=candidate_stats(ca, p, p>=.4, observed, masks))
        np.savez_compressed(path, **data)
        item['sha256'] = sha256_file(path)
    write_json(root/'manifest.json', manifest)
    meta = json.loads((root/'summary.json').read_text())
    meta.update(token_cache=True, gate=.4, manifest_sha256=sha256_file(root/'manifest.json'))
    write_json(root/'summary.json', meta)
    return root, split


def test_gain_controls_equal_capacity_and_ties():
    mean, std = np.zeros(19), np.ones(19)
    a, b = [PairedQualityHead(16, mean, std, gain=v) for v in (False, True)]
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())
    score = b(torch.randn(2, 3, 16), torch.randn(2, 16), torch.randn(2, 3, 19))
    assert torch.equal(score[:, 0], torch.zeros(2))
    loss = gain_loss(score, torch.tensor([[.5, .5, .9], [.2, .7, .2]]), torch.ones(2, 3, dtype=torch.bool), torch.ones(2))
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in b.parameters())


def test_token_matching_permutation_invariance_and_blind_control():
    torch.manual_seed(44)
    p, q, st = torch.randn(3, 16), torch.randn(16), torch.randn(3, 19)
    r, a = torch.randn(8, 16), torch.randn(5, 16)
    m = torch.rand(3, 8)>.5
    m[0] = False
    counts = []
    for mode in ('tokens', 'mean', 'blind'):
        h = TokenQualityHead(16, np.zeros(19), np.ones(19), mode).eval()
        counts.append(sum(x.numel() for x in h.parameters()))
        before = h(p, q, st, r, a, m)
        after = h(p, q, st, r.flip(0), a.flip(0), m.flip(1))
        torch.testing.assert_close(before, after)
        changed = h(p, q+2, st, r, a+3, m)
        if mode == 'blind':
            torch.testing.assert_close(before, changed, rtol=0, atol=0)
        before.sum().backward()
        assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in h.parameters())
    assert len(set(counts)) == 1


def test_refiner_zero_init_bounds_and_outside_support():
    h = BoundaryResidualHead(16)
    r, q, p, odds = torch.randn(8, 16), torch.randn(16), torch.randn(3, 16), torch.randn(8)
    m = torch.rand(3, 8)>.5
    support = m.clone()
    support[0] = False
    out = h(r, q, p, odds, m, support)
    torch.testing.assert_close(out, odds[None].expand_as(out))
    torch.nn.init.normal_(h.readout[-1].weight)
    out = h(r, q, p, odds, m, support)
    delta = out-odds
    assert delta.abs().max() <= .500001 and torch.equal(delta[~support], torch.zeros_like(delta[~support]))


def test_render_fallback_and_no_target_in_prediction(tmp_path):
    root, split = token_cache(tmp_path)
    rows, _ = read_cache(root, split, True)
    row = rows['val'][0]
    row['token_path'] = str(root/'val'/f"{row['sample_id']}.npz")
    args = SimpleNamespace(device='cpu', gate=.4)
    h = BoundaryResidualHead(16).eval()
    result = render_refined(h, [row], args)[0]
    np.testing.assert_allclose(result['quality_f1'], row['quality_f1'][0])
    model = TokenQualityHead(16, np.zeros(19), np.ones(19)).eval()
    before, _ = forward_row(model, row, 'cpu')
    row['quality_f1'] = 1-row['quality_f1']
    after, _ = forward_row(model, row, 'cpu')
    torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.parametrize('branch', ['gain', 'tokens', 'boundary'])
def test_parallel_training_synthetic_end_to_end(tmp_path, monkeypatch, branch):
    root, split = token_cache(tmp_path)
    out = tmp_path/branch
    monkeypatch.setattr(sys, 'argv', ['train_site_parallel.py', '--cache', str(root), '--splits', str(split),
        '--output', str(out), '--branch', branch, '--smoke', '--epochs', '2', '--batch-size', '2', '--device', 'cpu'])
    main()
    result = json.loads((out/'summary.json').read_text())
    assert result['complete'] and result['split'] == 'val'
    assert len(result['results']) == (9 if branch=='tokens' else 6)
    assert all(np.isfinite(r['sample_f1']) for r in result['results'])
    load_tokens.cache_clear()


def test_complete_link_does_not_chain_geometry_or_sites():
    objects = [{'sample_id': str(i), 'value': i} for i in range(3)]
    groups = complete_groups(objects, lambda a,b: abs(a['value']-b['value']) <= 1)
    assert [len(g) for g in groups] == [2, 1]
    ca = np.array([[0, 0, 0], [3, 0, 0], [15, 0, 0], [18, 0, 0]], float)
    a = dict(sample_id='a', axis_key='same', geometry_key='a', ca=ca,
             mask=np.array([1, 1, 0, 0], bool), smiles='X', source='biolip')
    b = dict(a, sample_id='b', geometry_key='b', ca=ca+[5, 3, -2], mask=~a['mask'], smiles='Y')
    assert geometry_matches(a, b)
    rows, artifacts = audit_groups([a, b])
    assert len(rows) == 1 and rows[0]['n_sites_heuristic'] == 2
    assert rows[0]['union_gain_over_largest_observation'] == 2
    assert artifacts[0]['union_positive_indices'] == [0, 1, 2, 3]


@pytest.mark.parametrize('dependency', [False, True])
def test_precision_audit_rejects_real_cross_dependency(tmp_path, monkeypatch, dependency):
    class Toy(torch.nn.Module):
        def forward(self, esm_embeddings, ligand_x):
            protein = esm_embeddings.clone()
            if dependency:
                protein = protein + ligand_x.mean()*.1
            return dict(h_prot_global=protein, h_ligand_context=ligand_x.clone())
    model = Toy().eval()
    inputs = dict(esm_embeddings=torch.ones(1, 4, 3), ligand_x=torch.ones(1, 5, 3))
    monkeypatch.setattr(torch, 'autocast', lambda *a, **kw: nullcontext())
    path = tmp_path/'audit.json'
    if dependency:
        with pytest.raises(ValueError, match='independence audit failed'):
            verify_independent_states(model, inputs, model(**inputs), path)
    else:
        result = verify_independent_states(model, inputs, model(**inputs), path)
        assert result['passed'] and result['fp32_repeat_max']['h_prot_global'] == 0
    assert json.loads(path.read_text())['passed'] is not dependency
