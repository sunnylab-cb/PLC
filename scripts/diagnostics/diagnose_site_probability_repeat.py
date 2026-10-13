#!/usr/bin/env python3
"""Read-only repeatability probe using an explicitly selected frozen code tree.

No training, tolerance changes, new calibration or acceptance of partial caches.
The FP32 scatter condition is a process-local diagnostic, not a model patch.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def probability_difference(p, reference, observed, target, gate):
    p, reference = np.asarray(p), np.asarray(reference)
    observed, target = np.asarray(observed, bool), np.asarray(target, bool)
    if not (p.shape == reference.shape == observed.shape == target.shape) or p.ndim != 1:
        raise ValueError("Probability/label axes differ")
    if not observed.any() or not np.isfinite(p).all() or not np.isfinite(reference).all():
        raise ValueError("Empty observed axis or nonfinite probabilities")
    a, b = observed & (p >= gate), observed & (reference >= gate)
    def f1(mask):
        return float(2 * (mask & target).sum() / max(int(mask.sum() + target.sum()), 1))
    error = np.abs(p - reference)
    ix = int(error.argmax())
    return dict(max_abs=float(error[ix]), mean_abs_full=float(error.mean()),
                mean_abs_observed=float(error[observed].mean()), max_index=ix,
                new_probability=float(p[ix]), reference_probability=float(reference[ix]),
                flips=int((a != b).sum()), flip_fraction_full=float((a != b).mean()),
                f1=f1(a), reference_f1=f1(b), f1_delta=f1(a) - f1(b))


def repeat_summary(probabilities, observed, target, gate):
    if len(probabilities) < 2:
        raise ValueError("At least two repetitions required")
    rows = [probability_difference(a, b, observed, target, gate)
            for i, a in enumerate(probabilities) for b in probabilities[:i]]
    return dict(repetitions=len(probabilities), max_pairwise_abs=max(r['max_abs'] for r in rows),
                max_pairwise_flips=max(r['flips'] for r in rows),
                max_pairwise_f1_delta=max(abs(r['f1_delta']) for r in rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code-root', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--reference-val', type=Path, required=True)
    ap.add_argument('--partial-cache', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--ids', nargs='+', default=['5ndf', '4k7o'])
    ap.add_argument('--repeats', type=int, default=10)
    args = ap.parse_args()
    if args.repeats < 2:
        ap.error('--repeats must be >=2')
    code = args.code_root.resolve()
    sys.path.insert(0, str(code))
    import torch
    import torch_geometric
    from scripts.diagnostics.export_site_quality_cache import prepared_dataset
    from scripts.diagnostics.export_strict_val_predictions import (
        create_v11_model, pocket_collate_fn_v10, prepare_model_inputs_v11)
    from scripts.diagnostics.strict_assignment_common import read_strict_splits, write_json
    from scripts.infer.benchmark_integrity import safe_tensor_load, sha256_file
    import src.models.gvp_layers as gvp_layers

    if not torch.cuda.is_available():
        raise RuntimeError('Use a scheduled GPU allocation')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(44)
    np.random.seed(44)
    split = args.root / 'splits.txt'
    if not set(args.ids).issubset(read_strict_splits(split)['val']):
        raise ValueError('This diagnostic accepts strict VAL IDs only')
    reference_summary = json.loads((args.reference_val / 'export_summary.json').read_text())
    ckpt = safe_tensor_load(args.checkpoint)
    config = copy.deepcopy(ckpt['config'])
    gate = float(ckpt['val_contact_gate'])
    checkpoint_hash = sha256_file(args.checkpoint)
    if (reference_summary['checkpoint_sha256'] != checkpoint_hash
            or reference_summary['split_sha256'] != sha256_file(split)
            or sha256_file(config['data']['split_file']) != sha256_file(split)
            or gate != reference_summary['gate']):
        raise ValueError('Reference/checkpoint/split provenance mismatch')
    model, version = create_v11_model(ckpt, config, torch.device('cuda'))
    del ckpt
    model.requires_grad_(False)
    if version != 'v11m13' or any(m.training for m in model.modules()):
        raise ValueError('Expected the frozen V11m13 eval model')
    dataset = prepared_dataset(args.root, args.ids, config['data'])
    if dataset.pdb_ids != args.ids:
        raise ValueError('Dataset silently filtered or reordered requested IDs')
    def tensor_hash(t):
        x = t.detach().cpu().contiguous()
        return dict(shape=list(x.shape), dtype=str(x.dtype),
                    sha256=hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    def input_hashes(inputs):
        return {k: tensor_hash(v) for k, v in inputs.items() if torch.is_tensor(v)}
    def loaded_sources():
        result = {}
        for module in list(sys.modules.values()):
            filename = getattr(module, '__file__', None)
            if not filename:
                continue
            path = Path(filename).resolve()
            if path.is_relative_to(code) and path.is_file():
                result[str(path.relative_to(code))] = sha256_file(path)
        return result
    report = dict(complete=False, code_root=str(code), checkpoint_sha256=checkpoint_hash,
                  split_sha256=sha256_file(split), gate=gate, version=version,
                  diagnostic_only=True, torch=str(torch.__version__), cuda=torch.version.cuda,
                  pyg=str(torch_geometric.__version__), device=torch.cuda.get_device_name(),
                  matmul_precision=torch.get_float32_matmul_precision(),
                  tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                  tf32_cudnn=torch.backends.cudnn.allow_tf32,
                  deterministic=torch.are_deterministic_algorithms_enabled(), samples={})
    original_scatter = gvp_layers.scatter
    try:
        for index, sid in enumerate(args.ids):
            inputs = prepare_model_inputs_v11(pocket_collate_fn_v10([dataset[index]]), torch.device('cuda'))
            fingerprints = input_hashes(inputs)
            reload_inputs = prepare_model_inputs_v11(pocket_collate_fn_v10([dataset[index]]), torch.device('cuda'))
            if fingerprints != input_hashes(reload_inputs):
                raise ValueError(f'Loader is not repeatable: {sid}')
            del reload_inputs
            with np.load(args.reference_val / 'predictions' / f'{sid}.npz') as z:
                ref = {k: z[k] for k in z.files}
            with np.load(args.partial_cache / 'val' / f'{sid}.npz') as z:
                cached = {k: z[k] for k in ('probability', 'contact_target', 'ca', 'observed')}
            observed, target = ref['observed'].astype(bool), ref['labels'] == 0
            if (not np.array_equal(cached['contact_target'], target)
                    or not np.array_equal(cached['ca'], ref['ca_coords'])
                    or not np.array_equal(cached['observed'], observed)):
                raise ValueError(f'Archived label/coordinate/coverage mismatch: {sid}')
            sample = dataset[index]
            n = sample['n_residues']
            if (not np.array_equal(observed, np.arange(len(observed)) < n)
                    or not np.array_equal(sample['gvp_coords'].numpy(), ref['ca_coords'][:n])):
                raise ValueError(f'Current loader axis mismatch: {sid}')
            row = dict(n_full=len(observed), n_observed=n, inputs=fingerprints,
                       inputs_repeatable=True, archive_axes_match=True,
                       archive_difference=probability_difference(cached['probability'],
                           ref['contact_probability'], observed, target, gate), modes={})
            report['samples'][sid] = row
            stored, fp32_first = {}, None
            # FP32 first provides an unchanged high-precision reference for both BF16 conditions.
            for mode in ('fp32', 'bf16', 'bf16_gvp_scatter_fp32'):
                amp = mode != 'fp32'
                scatter_types = set()
                def scatter(src, *pos, **kw):
                    scatter_types.add((str(src.dtype), kw.get('reduce', 'sum')))
                    if mode == 'bf16_gvp_scatter_fp32' and src.dtype in (torch.bfloat16, torch.float16):
                        return original_scatter(src.float(), *pos, **kw).to(src.dtype)
                    return original_scatter(src, *pos, **kw)
                gvp_layers.scatter = scatter
                captured = {}
                def capture(module, pos, out):
                    tensors = out if isinstance(out, tuple) else (out,)
                    captured['gvp'] = [t.detach().float().cpu().clone() for t in tensors if torch.is_tensor(t)]
                handle = model.gvp_encoder.register_forward_hook(capture)
                trials, probs, first_states = [], [], None
                try:
                    for repeat in range(args.repeats):
                        cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=amp):
                            out = model(**inputs)
                        p = np.zeros(len(observed), np.float32)
                        p[:n] = torch.softmax(out['logits'].float(), -1)[0, :n, 0].cpu().numpy()
                        states = {k: [out[k].detach().float().cpu().clone()]
                                  for k in ('h_prot_global', 'h_ligand_context', 'logits')}
                        states.update(captured)
                        if any(not torch.isfinite(t).all() for ts in states.values() for t in ts):
                            raise ValueError('Nonfinite diagnostic states')
                        if first_states is None:
                            first_states = states
                        if mode == 'fp32' and repeat == 0:
                            fp32_first = p.copy()
                        item = dict(repeat=repeat,
                            reference=probability_difference(p, ref['contact_probability'], observed, target, gate),
                            partial=probability_difference(p, cached['probability'], observed, target, gate),
                            fp32=probability_difference(p, fp32_first, observed, target, gate),
                            state_error_vs_first={k: max(float((a-b).abs().max())
                                for a, b in zip(first_states[k], states[k])) for k in states},
                            rng_changed=not (torch.equal(cpu_rng, torch.get_rng_state())
                                            and torch.equal(gpu_rng, torch.cuda.get_rng_state())))
                        if fingerprints != input_hashes(inputs):
                            raise ValueError(f'Model mutated inputs: {sid}')
                        trials.append(item)
                        probs.append(p)
                        print(json.dumps(dict(sample=sid, mode=mode, **item)), flush=True)
                        del out, states
                finally:
                    handle.remove()
                    gvp_layers.scatter = original_scatter
                row['modes'][mode] = dict(trials=trials, summary=repeat_summary(probs, observed, target, gate),
                                         scatter_input_types=sorted(scatter_types))
                stored[mode] = np.stack(probs)
                write_json(args.output / 'report.json', report)
            np.savez_compressed(args.output / f'{sid}_probabilities.npz', **stored,
                                archived=ref['contact_probability'], partial=cached['probability'],
                                observed=observed, target=target)
        report['complete'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        gvp_layers.scatter = original_scatter
        report['loaded_code_hashes'] = loaded_sources()
        report['diagnostic_sha256'] = sha256_file(Path(__file__))
        write_json(args.output / 'report.json', report)
    print('PROBABILITY_REPEAT_DIAGNOSTIC_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
