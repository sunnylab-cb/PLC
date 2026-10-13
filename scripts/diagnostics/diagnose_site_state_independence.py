#!/usr/bin/env python3
"""Scheduled, read-only repeated-forward diagnosis of frozen pre-cross states."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.export_site_quality_cache import prepared_dataset
from scripts.diagnostics.export_strict_val_predictions import create_v11_model, pocket_collate_fn_v10, prepare_model_inputs_v11
from scripts.diagnostics.strict_assignment_common import read_strict_splits, write_json
from scripts.infer.benchmark_integrity import safe_tensor_load


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(44)
    ckpt = safe_tensor_load(args.checkpoint)
    config = copy.deepcopy(ckpt['config'])
    model, _ = create_v11_model(ckpt, config, torch.device('cuda'))
    model.requires_grad_(False)
    ids = read_strict_splits(args.root / 'splits.txt')['train'][:1]
    data = prepared_dataset(args.root, ids, config['data'])
    inputs = prepare_model_inputs_v11(pocket_collate_fn_v10([data[0]]), torch.device('cuda'))
    originals = {k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)}
    captured = {}
    def hook(name):
        def capture(module, args, output):
            tensors = output if isinstance(output, tuple) else (output,)
            captured[name] = [t.detach().float().clone() for t in tensors if torch.is_tensor(t)]
        return capture
    handles = [getattr(model, name).register_forward_hook(hook(name))
               for name in ('gvp_encoder', 'protein_context_encoder', 'ligand_self_attn')
               if isinstance(getattr(model, name, None), torch.nn.Module)]
    def run(amp, change=None):
        altered = dict(inputs)
        if change:
            altered[change] = torch.zeros_like(inputs[change])
        captured.clear()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
            out = model(**altered)
        result = {k: [out[k].detach().float().clone()] for k in ('h_prot_global', 'h_ligand_context')}
        result.update(captured)
        if any(not torch.equal(inputs[k], v) for k, v in originals.items()):
            raise ValueError('Model mutated input tensors')
        return result
    report = {'sample_id': ids[0], 'training_modules': [n for n, m in model.named_modules() if m.training], 'trials': []}
    for amp in (True, False):
        ref = run(amp)
        for change in (None, None, 'ligand_x', 'esm_embeddings'):
            cur = run(amp, change)
            errors = {k: max(float((a-b).abs().max()) for a, b in zip(ref[k], cur[k])) for k in ref}
            row = {'bf16': amp, 'intervention': change or 'repeat_identical', 'max_abs': errors}
            report['trials'].append(row)
            print(json.dumps(row), flush=True)
    for handle in handles:
        handle.remove()
    write_json(args.output, report)
    print('INDEPENDENCE_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
