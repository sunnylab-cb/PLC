#!/usr/bin/env python3
"""Prepare matched scratch FFFF/GFGF configs; do not modify shared datasets.

Uses only existing source-verified TRAIN/VAL label-axis corrections. Both arms
receive the same private label file. TEST labels are not inspected or changed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.diagnostics.prepare_internal_label_alignment import aligned_values, load_overlay
from scripts.diagnostics.strict_assignment_common import find_feature, read_strict_splits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-config', type=Path, required=True)
    ap.add_argument('--data-root', type=Path, required=True)
    ap.add_argument('--label-alignment', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    split = args.data_root / 'splits.txt'
    groups = read_strict_splits(split)
    if sha(split) != '170a4e6364c19662ad922ace7957e1733409dfe146ebdb74b15c95130bb622bc':
        raise ValueError('Unexpected strict split')
    labels_path = args.output / 'train_val_aligned_distances.pt'
    if labels_path.exists():
        raise FileExistsError('Use a fresh preparation directory; do not overwrite provenance')
    original_labels = args.data_root / 'residue_ligand_distances_dcenter.pt'
    labels = torch.load(original_labels, map_location='cpu', weights_only=False)
    overlay = load_overlay(args.label_alignment, split, groups['train'] + groups['val'])
    for sid in overlay:
        protein = torch.load(find_feature(args.data_root, 'complete_features_v4_nma', sid, True),
                             map_location='cpu', weights_only=False)
        labels[sid] = aligned_values(sid, protein, labels[sid], overlay)
    torch.save(labels, labels_path)
    base = yaml.safe_load(args.base_config.read_text())
    for pattern in ('FFFF', 'GFGF'):
        cfg = copy.deepcopy(base)
        cfg['experiment'].update(name=f'protein_hybrid_{pattern}_scratch_seed44', seed=44,
                                 version_suffix='v11m13', output_dir=str(args.output.parent / 'outputs'))
        cfg['data'].update(data_root=str(args.data_root), split_file=str(split),
                           distances_file=str(labels_path))
        cfg['model'].update(protein_hybrid_pattern=pattern, protein_gdn_backend='fla', use_compile=False)
        train = cfg['training']
        train.update(init_from_checkpoint=None, init_from_v11m3=None, freeze_v11_base=False,
                     freeze_v11m3_base=False, use_amp=True, use_bf16=True,
                     save_every=120, num_epochs=120)
        for key in ('resume', 'resume_from_checkpoint', 'checkpoint_path', 'delta_trainable_tokens'):
            train.pop(key, None)
        if cfg['model']['n_transformer_layers'] != 4 or not cfg['model']['use_v12_coattention_block']:
            raise ValueError('Wrong base architecture')
        (args.output / f'{pattern}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    record = dict(split_sha256=sha(split), split_counts={k: len(v) for k, v in groups.items()},
                  base_config=str(args.base_config), base_config_sha256=sha(args.base_config),
                  source_label_sha256=sha(original_labels), aligned_label_sha256=sha(labels_path),
                  overlay_sha256=sha(args.label_alignment), corrected_ids=sorted(overlay),
                  patterns=['FFFF', 'GFGF'], seed=44, resume=False,
                  note='Known source-verified label fixes in BOTH arms; not exact reproduction of historical label prefixing.')
    (args.output / 'provenance.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2), flush=True)


if __name__ == '__main__':
    main()
