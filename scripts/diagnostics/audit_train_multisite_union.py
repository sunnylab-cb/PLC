#!/usr/bin/env python3
"""TRAIN-only union coverage audit, not supervision approval or new training.

Geometry clusters are complete-link: every pair must pass max aligned CA error
<=0.05 A. Site groups are conservative complete-link high-overlap observations.
Unknown residues remain unknown, never certified nonbinding negatives.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.audit_strict_train_negatives import initialize_worker, load_sample, rigid_errors
from scripts.diagnostics.prepare_internal_label_alignment import load_overlay
from scripts.diagnostics.strict_assignment_common import read_strict_splits, write_json, write_csv, quantiles
from scripts.infer.benchmark_integrity import safe_tensor_load, sha256_file


def complete_groups(items, compatible):
    groups = []
    for item in sorted(items, key=lambda s: s['sample_id']):
        for group in groups:
            if all(compatible(item, other) for other in group):
                group.append(item)
                break
        else:
            groups.append([item])
    return groups


def geometry_matches(a, b):
    return a['geometry_key'] == b['geometry_key'] or rigid_errors(a['ca'], b['ca'])[1] <= .05


def equivalent_site(a, b):
    union = a['mask'] | b['mask']
    if not a['mask'].any() or not b['mask'].any():
        return False
    if (a['mask'] & b['mask']).sum()/union.sum() < .5:
        return False
    return all(np.linalg.norm(c[a['mask']].mean(0)-c[b['mask']].mean(0)) < 8.
               for c in (a['ca'], b['ca']))


def audit_groups(samples):
    axes = defaultdict(list)
    for sample in samples:
        axes[sample['axis_key']].append(sample)
    rows, artifacts = [], []
    for axis in sorted(axes):
        for group in complete_groups(axes[axis], geometry_matches):
            sites = complete_groups([s for s in group if s['mask'].any()], equivalent_site)
            union = np.logical_or.reduce([s['mask'] for s in group])
            chemistry = sorted({s['smiles'] for s in group if s['smiles']})
            gid = group[0]['sample_id']
            rows.append(dict(group_id=gid, axis_key=axis, rows=len(group), n_sites_heuristic=len(sites),
                known_chemical_identities=len(chemistry), chemistry_unknown=sum(not s['smiles'] for s in group),
                known_positive_residues=int(union.sum()), unknown_residues=int((~union).sum()),
                union_gain_over_largest_observation=int(union.sum()-max(s['mask'].sum() for s in group)),
                pdbbind_rows=sum(s['source']=='pdbbind' for s in group), biolip_rows=sum(s['source']=='biolip' for s in group)))
            artifacts.append(dict(group_id=gid, axis_key=axis, members=[s['sample_id'] for s in group],
                reference_geometry_key=group[0]['geometry_key'], union_positive_indices=np.flatnonzero(union).tolist(),
                heuristic_sites=[dict(members=[s['sample_id'] for s in site],
                                     positive_indices=np.flatnonzero(np.logical_or.reduce([s['mask'] for s in site])).tolist(),
                                     known_chemistry=sorted({s['smiles'] for s in site if s['smiles']})) for site in sites]))
    return rows, artifacts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--chemistry-manifest', type=Path, required=True)
    ap.add_argument('--label-alignment', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    split = args.root/'splits.txt'
    splits = read_strict_splits(split)
    if {k: len(v) for k, v in splits.items()} != {'train': 10317, 'val': 647, 'test': 647}:
        raise ValueError('Unexpected split population')
    train = splits['train']
    manifest = json.loads(args.chemistry_manifest.read_text())
    if manifest['split'] != 'train' or manifest['split_sha256'] != sha256_file(split) or set(manifest['rows'])-set(train):
        raise ValueError('Chemistry provenance mismatch')
    overlay = load_overlay(args.label_alignment, split, train)
    source = {}
    for line in (args.root/'provenance.jsonl').read_text().splitlines():
        row = json.loads(line)
        source[row['sample_id']] = row['source']
    all_distances = safe_tensor_load(args.root/'residue_ligand_distances_dcenter.pt')
    distances = {sid: all_distances[sid] for sid in train}
    del all_distances
    args.output.mkdir(parents=True, exist_ok=False)
    samples = []
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker,
        initargs=(str(args.root), distances, manifest['rows'], source, overlay)) as pool:
        for i, sample in enumerate(pool.map(load_sample, train, chunksize=8)):
            samples.append(sample)
            if (i+1)%250 == 0:
                print('UNION_TRAIN_QC', i+1, '/', len(train), flush=True)
    errors = [dict(sample_id=s['sample_id'], error=s['error']) for s in samples if s['error']]
    write_json(args.output/'qc_errors.json', errors)
    if errors:
        raise ValueError('Failed TRAIN source QC; no union produced')
    rows, artifacts = audit_groups(samples)
    write_csv(args.output/'groups.csv', rows)
    write_json(args.output/'train_unions_diagnostic_only.json', artifacts)
    write_json(args.output/'summary.json', dict(complete=True, split='train', n_rows=len(train),
        split_sha256=sha256_file(split), chemistry_sha256=sha256_file(args.chemistry_manifest),
        label_overlay_sha256=sha256_file(args.label_alignment), code_sha256=sha256_file(__file__),
        geometry_groups=len(rows), multisite_groups=sum(r['n_sites_heuristic']>=2 for r in rows),
        multisite_rows=sum(r['rows'] for r in rows if r['n_sites_heuristic']>=2),
        informative_union_groups=sum(r['union_gain_over_largest_observation']>0 for r in rows),
        union_gain=quantiles(r['union_gain_over_largest_observation'] for r in rows),
        site_counts=quantiles(r['n_sites_heuristic'] for r in rows),
        limitations=['Complete-link geometry avoids transitive coordinate drift.',
            'Site grouping is heuristic, not biologically certified distinct pockets.',
            'Residues outside the union are unknown, not negatives.',
            'Only strict TRAIN annotations are read for union construction.',
            'No training dataset, split, or model was modified.']))
    print('TRAIN_MULTISITE_UNION_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
