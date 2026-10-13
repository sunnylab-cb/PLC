#!/usr/bin/env python3
"""GPU-only FLA equivalence, matched initialization and real-batch backward smoke."""
import argparse
import gc
import inspect
import json
import time
from importlib.metadata import version
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.models.protein_bidirectional_gdn import BidirectionalProteinGDN
from src.models.pocket_gvp_fusion_v11m13 import PocketGVPFusionV11m13
from src.data.pocket_dataset_v8 import PocketDatasetV8
from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11
from scripts.diagnostics.strict_assignment_common import read_strict_splits


def relative_rmse(a, b):
    return float((a.float() - b.float()).square().mean().sqrt() / b.float().square().mean().sqrt().clamp_min(1e-8))


def kernel_check():
    torch.manual_seed(44)
    fast = BidirectionalProteinGDN(128, 2, backend='fla').cuda()
    ref = BidirectionalProteinGDN(128, 2, backend='reference').cuda()
    ref.load_state_dict(fast.state_dict())
    x = torch.randn(2, 73, 128, device='cuda', requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    chains = torch.zeros(2, 73, device='cuda', dtype=torch.long)
    chains[0, 69:] = 1
    mask = torch.ones_like(chains, dtype=torch.bool)
    mask[1, 67:] = False
    with torch.autocast('cuda', dtype=torch.bfloat16):
        a = fast(x, mask=mask, chain_ids=chains)
        b = ref(xr, mask=mask, chain_ids=chains)
    probe = torch.randn_like(a)
    (a * probe).sum().backward()
    (b * probe).sum().backward()
    errors = {'output_rrmse': relative_rmse(a, b), 'input_grad_rrmse': relative_rmse(x.grad, xr.grad)}
    for name, p in fast.named_parameters():
        rp = dict(ref.named_parameters())[name]
        if p.grad is None or not torch.isfinite(p.grad).all():
            raise RuntimeError(f'Invalid FLA gradient: {name}')
        errors[f'grad/{name}'] = relative_rmse(p.grad, rp.grad)
    if errors['output_rrmse'] > .04 or errors['input_grad_rrmse'] > .08 or max(errors.values()) > .15:
        raise RuntimeError(f'FLA/reference mismatch: {errors}')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        before = fast(x, mask=mask, chain_ids=chains)
        changed = x.detach().clone()
        changed[0, 69:] += 100
        changed[1, 67:] = float('nan')
        after = fast(changed, mask=mask, chain_ids=chains)
        torch.testing.assert_close(before[0, :69], after[0, :69], rtol=.01, atol=.002)
        assert torch.isfinite(after).all()
        assert not torch.count_nonzero(after[1, 67:])
    return errors


def make_batch(cfg):
    d = cfg['data']
    ids = read_strict_splits(d['split_file'])['train'][:16]
    allowed = inspect.signature(PocketDatasetV8).parameters
    kwargs = {k: v for k, v in d.items() if k in allowed}
    kwargs.update(pdb_ids=ids, use_nma=False, split='train', verbose=False)
    dataset = PocketDatasetV8(**kwargs)
    if len(dataset) < 4:
        raise RuntimeError('Missing smoke TRAIN samples')
    return pocket_collate_fn_v10([dataset[i] for i in range(4)])


def full_model_check(configs):
    torch.manual_seed(44)
    control = PocketGVPFusionV11m13(configs[0]['model'])
    torch.manual_seed(44)
    hybrid = PocketGVPFusionV11m13(configs[1]['model'])
    same = 0
    hd = hybrid.state_dict()
    for key, value in control.state_dict().items():
        if key in hd:
            torch.testing.assert_close(value, hd[key], rtol=0, atol=0)
            same += 1
    del control, hybrid, hd
    batch = make_batch(configs[0])
    inputs = prepare_model_inputs_v11(batch, torch.device('cuda'))
    labels = batch['labels'].cuda()
    valid = inputs['protein_mask'].bool()
    rows = []
    for pattern, cfg in zip(('FFFF', 'GFGF'), configs):
        torch.manual_seed(44)
        model = PocketGVPFusionV11m13(cfg['model'])
        model.cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for step in range(3):
            torch.cuda.synchronize()
            start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(**inputs)
                # This is a connectivity smoke, not a changed production loss.
                loss = torch.nn.functional.cross_entropy(outputs['logits'][valid].float(), labels[valid])
            if not torch.isfinite(loss):
                raise RuntimeError(f'{pattern} non-finite loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            for name, parameter in model.named_parameters():
                if pattern == 'GFGF' and '.attn.decay_log_bias' in name and parameter.grad is None:
                    raise RuntimeError(f'Disconnected GDN: {name}')
            optimizer.step()
            torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)
            print(f'FULL_SMOKE {pattern} step={step} loss={loss.item():.6f} grad={grad.item():.6f}', flush=True)
        rows.append(dict(pattern=pattern, parameters=sum(p.numel() for p in model.parameters()),
                         peak_allocated_gb=torch.cuda.max_memory_allocated() / 2**30,
                         step_seconds=elapsed, batch_shape=list(inputs['protein_mask'].shape),
                         note='Four TRAIN samples; warmup excluded when comparing step time'))
        model.eval()
        # The existing validation loop uses the default FP16 autocast. Verify
        # this separately rather than assuming the BF16 training smoke covers it.
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            validation_logits = model(**inputs)['logits']
        if not torch.isfinite(validation_logits[valid]).all():
            raise RuntimeError(f'{pattern} non-finite validation logits')
        optimizer.zero_grad(set_to_none=True)
        del optimizer, outputs, loss, validation_logits, model
        gc.collect()
        torch.cuda.empty_cache()
    return dict(unchanged_tensors_identical=same, runs=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    configs = [yaml.safe_load((args.prepared / f'{p}.yaml').read_text()) for p in ('FFFF', 'GFGF')]
    torch.set_num_threads(4)
    report = dict(torch=torch.__version__, fla_core=version('fla-core'),
                  gpu=torch.cuda.get_device_name(), kernel=kernel_check())
    report['full_model'] = full_model_check(configs)
    report['passed'] = True
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print('HYBRID_SMOKE_PASSED', json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
