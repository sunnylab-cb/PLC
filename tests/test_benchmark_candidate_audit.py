import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.diagnostics.benchmark_candidate_audit import (
    binary_f1, candidate_metrics, contact_components, score_component, summarize, process_protein,
)


def test_candidates_do_not_depend_on_targets_but_oracle_does():
    ca = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [20, 0, 0], [21, 0, 0], [22, 0, 0]], dtype=float)
    active = np.ones(6, dtype=bool)
    p = np.array([.95, .9, .8, .7, .6, .6])
    target = np.array([False] * 3 + [True] * 3)
    row, chosen, masks = candidate_metrics(ca, p, active, target)
    other, chosen2, masks2 = candidate_metrics(ca, p, active, ~target)
    assert len(masks) == 2 and np.array_equal(chosen, ~target)
    assert row['oracle_f1'] == 1 and row['selected_f1'] == 0
    assert row['candidate_hit_dcc4'] == 1 and row['selected_hit_dcc4'] == 0
    assert np.array_equal(chosen, chosen2) and all(np.array_equal(a, b) for a, b in zip(masks, masks2))
    assert other['selected_f1'] == 1


def test_no_candidates_and_zero_gt_have_explicit_denominators():
    ca = np.zeros((2, 3))
    row, chosen, masks = candidate_metrics(ca, np.ones(2), np.ones(2, bool), np.zeros(2, bool))
    assert row['n_candidates'] == 0 and not chosen.any() and not masks
    assert row['candidate_hit_dcc4'] is None and row['selected_f1'] == 0
    assert row['n_discarded_small_component_residues'] == 2


def test_components_are_transitively_connected_and_score_not_raw_sum():
    ca = np.array([[0, 0, 0], [7, 0, 0], [14, 0, 0], [100, 0, 0]])
    groups = contact_components(ca, np.ones(4, bool), 8., 3)
    assert len(groups) == 1 and groups[0].tolist() == [0, 1, 2]
    p = np.array([.9, .1, .1, .8, .8, .8])
    assert score_component(p, np.arange(3)) > score_component(p, np.arange(3, 6))


def test_protein_balancing_does_not_repeat_large_groups():
    rows = [{'protein_key': 'A', 'baseline_f1': 1.}] * 3 + [{'protein_key': 'B', 'baseline_f1': 0.}]
    result = summarize(rows, ('baseline_f1',))
    assert result['sample_mean']['baseline_f1']['mean'] == .75
    assert result['protein_mean']['baseline_f1']['mean'] == .5
    assert binary_f1(np.zeros(3, bool), np.zeros(3, bool)) == 0


def test_full_protein_prediction_and_chemical_pair_audit(tmp_path):
    from rdkit import Chem

    dataset, protein = 'demo', '1abc'
    predictions = tmp_path / 'run' / 'predictions' / dataset
    sdfs = tmp_path / 'sdfs' / dataset
    predictions.mkdir(parents=True)
    sdfs.mkdir(parents=True)
    ca = np.array([[0,0,0], [1,0,0], [2,0,0], [20,0,0], [21,0,0], [22,0,0]], float)
    p = np.array([.95, .9, .8, .7, .6, .6])
    rows = []
    for i, chemical in enumerate(['C', 'N']):
        key = f'{protein}_{chemical}_A{i}'
        target = np.arange(6) < 3 if i == 0 else np.arange(6) >= 3
        np.savez(predictions / f'{key}.npz', probs=np.column_stack((p, (1-p)/2, (1-p)/2)),
                 preds=np.zeros(6, dtype=int), labels=np.where(target, 0, 2), ca_coords=ca,
                 residue_keys=np.array([['A', str(j), ''] for j in range(6)]),
                 protein_key=np.asarray(protein), ligand_key=np.asarray(key))
        mol = Chem.MolFromSmiles(chemical)
        with Chem.SDWriter(str(sdfs / f'{key}.sdf')) as writer:
            writer.write(mol)
        rows.append({'pdb_id': key, 'ligand_key': key, 'f1_c0': str(2/3)})
    results, pairs = process_protein((str(tmp_path/'run'), dataset, protein, rows, str(tmp_path/'sdfs'), [8.], 3))
    assert len(results) == 2 and all(r['distinct_chem_distinct_pocket'] for r in results)
    assert len(pairs) == 1 and pairs[0]['relation'] == 'different_chemistry'
    assert pairs[0]['symmetric_swap_f1_gap'] == 0
