#!/usr/bin/env python3
"""Frozen-cache gain, token interaction, and boundary pilots; strict TRAIN/VAL only.

Each invocation runs ONE branch and its matched controls sequentially. There is
no resume, backbone update, test access, native ligand pose, or threshold sweep.
"""
import argparse
import copy
import json
from functools import lru_cache
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.train.train_site_quality_probe import read_cache, fit_stats, make_batch, summarize_selection
from scripts.diagnostics.site_quality_features import candidate_stats, candidate_targets, pack_candidates
from scripts.diagnostics.strict_assignment_common import write_csv, write_json
from scripts.infer.benchmark_integrity import sha256_file
from src.models.site_quality_head import SiteQualityHead, quality_loss, select_with_fallback
from src.models.site_parallel_heads import PairedQualityHead, TokenQualityHead, BoundaryResidualHead, gain_loss


@lru_cache(maxsize=24)
def load_tokens(path):
    with np.load(path, allow_pickle=False) as z:
        keys = ('residue_tokens', 'atom_tokens', 'candidate_indices', 'candidate_offsets',
                'ca', 'probability', 'contact_target', 'observed')
        data = {k: np.array(z[k]) for k in keys}
    n, d = data['residue_tokens'].shape
    a = data['atom_tokens']
    size = len(data['probability'])
    if not n or not len(a) or a.ndim != 2 or a.shape[1] != d or n > size:
        raise ValueError('Invalid token shape')
    if data['ca'].shape != (size, 3) or data['contact_target'].shape != (size,) or data['observed'].shape != (size,):
        raise ValueError('Token/target axis mismatch')
    if not np.array_equal(data['observed'], np.arange(size) < n):
        raise ValueError('Token observed axis must match prefix')
    idx, off = data['candidate_indices'], data['candidate_offsets']
    if off[0] != 0 or off[-1] != len(idx) or np.any(np.diff(off) < 0) or np.any((idx < 0) | (idx >= size)):
        raise ValueError('Invalid packed mask indices')
    if not all(np.isfinite(v).all() for v in data.values()):
        raise ValueError('Nonfinite token cache')
    return data


def unpack(data):
    idx, off = data['candidate_indices'], data['candidate_offsets']
    return [idx[off[i]:off[i+1]] for i in range(len(off)-1)]


def token_inputs(row, device, query=None):
    data = load_tokens(row['token_path'])
    residues = torch.as_tensor(data['residue_tokens'].astype(np.float32), device=device)
    atoms = torch.as_tensor(data['atom_tokens'].astype(np.float32), device=device)
    if query is not None:
        atoms = torch.as_tensor(query['atoms'], dtype=torch.float32, device=device)
    membership = torch.zeros(len(row['pocket']), len(residues), dtype=torch.bool, device=device)
    for i, indices in enumerate(unpack(data)):
        visible = indices[indices < len(residues)]
        membership[i, torch.as_tensor(visible, dtype=torch.long, device=device)] = True
    return residues, atoms, membership


def query_list(rows, mode):
    if mode == 'real':
        return [None] * len(rows)
    if mode == 'zero':
        return [{'ligand': np.zeros_like(r['ligand']),
                 'atoms': np.zeros_like(load_tokens(r['token_path'])['atom_tokens'])} for r in rows]
    if mode != 'permuted' or len(rows) < 2:
        raise ValueError('Query permutation requires >=2 rows')
    order = np.random.default_rng(44).permutation(len(rows))
    donor = np.empty(len(rows), int)
    donor[order] = np.roll(order, 1)
    return [{'ligand': rows[j]['ligand'], 'atoms': load_tokens(rows[j]['token_path'])['atom_tokens']} for j in donor]


def forward_row(model, row, device, query=None, blind=False):
    q = row['ligand'] if query is None else query['ligand']
    if blind:
        q = np.zeros_like(q)
    b = make_batch([row], device, queries=[q])
    if isinstance(model, TokenQualityHead):
        r, a, m = token_inputs(row, device, query)
        scores = model(b['pocket'][0], b['ligand'][0], b['stats'][0], r, a, m)[None]
    else:
        scores = model(b['pocket'], b['ligand'], b['stats'])
    return scores, b


def evaluate(model, rows, device, margin, queries=None, blind=False):
    model.eval()
    output = []
    queries = queries if queries is not None else [None] * len(rows)
    with torch.inference_mode():
        for row, query in zip(rows, queries):
            scores, b = forward_row(model, row, device, query, blind)
            j = int(select_with_fallback(scores, b['valid'], margin)[0])
            raw = int(select_with_fallback(scores, b['valid'], 0)[0])
            s, f1 = scores[0].cpu().numpy(), row['quality_f1']
            target = f1-f1[0] if getattr(model, 'gain', False) else f1
            output.append(dict(sample_id=row['sample_id'], protein_key=row['protein_key'], selected=j,
                baseline_f1=float(f1[0]), selected_f1=float(f1[j]), unguarded_f1=float(f1[raw]),
                oracle_f1=float(f1.max()), selected_size=int(row['sizes'][j]), original_size=int(row['sizes'][0]),
                true_positives=int(row['true_positives'][j]), n_true=row['n_true'],
                predicted_quality=float(s[j]), original_predicted_quality=float(s[0]),
                quality_mae=float(np.abs(s-target).mean()), within002_of_oracle=float(f1[j] >= f1.max()-.02)))
    metrics = summarize_selection(output)
    metrics['mean_regret'] = metrics['oracle_f1'] - metrics['sample_f1']
    return metrics, output


def train_ranker(model, train, val, weights, out, args, epochs, blind=False):
    out.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    rng = np.random.default_rng(args.seed)
    writer = SummaryWriter(str(out / 'tb'))
    history, best, best_epoch = [], -1., None
    start_time = time.time()
    for epoch in range(1, epochs+1):
        model.train()
        losses, norms = [], []
        order = rng.permutation(len(train))
        for start in range(0, len(order), args.batch_size):
            indices = order[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_loss = 0.
            for i in indices:
                score, b = forward_row(model, train[i], args.device, blind=blind)
                w = score.new_tensor([weights[i]])
                fn = gain_loss if getattr(model, 'gain', False) else quality_loss
                loss = fn(score, b['target'], b['valid'], w) / len(indices)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite rank loss')
                loss.backward()
                batch_loss += float(loss.detach())
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)))
            optimizer.step()
            losses.append(batch_loss)
        metrics, _ = evaluate(model, val, args.device, args.margin, blind=blind)
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), grad_norm=float(np.mean(norms)),
                      elapsed_s=time.time()-start_time, **metrics)
        if str(args.device).startswith('cuda'):
            record['peak_allocated_gib'] = torch.cuda.max_memory_allocated()/1024**3
        history.append(record)
        print(out.name, json.dumps(record), flush=True)
        write_csv(out / 'history.csv', history)
        for key, value in record.items():
            writer.add_scalar(key, value, epoch)
        writer.flush()
        if metrics['protein_f1'] > best:
            best, best_epoch = metrics['protein_f1'], epoch
            torch.save(dict(model_state_dict=model.state_dict(), epoch=epoch, metrics=metrics,
                            cache_summary_sha256=sha256_file(args.cache/'summary.json'),
                            args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}), out/'best_val_head.pt')
    writer.close()
    checkpoint = torch.load(out/'best_val_head.pt', map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    return best_epoch


def boundary_inputs(row, device, query=None):
    data = load_tokens(row['token_path'])
    residues, _, member = token_inputs(row, device)
    ca = torch.as_tensor(data['ca'][:len(residues)], dtype=torch.float32, device=device)
    # Protein-only neighborhood; no ligand coordinates or target masks.
    adjacent = torch.cdist(ca, ca) <= 4.
    support = (member.float() @ adjacent.float()) > 0
    support[0] = False
    p = torch.as_tensor(data['probability'][:len(residues)], dtype=torch.float32, device=device)
    logits = torch.logit(p.clamp(1e-6, 1-1e-6))
    q = row['ligand'] if query is None else query['ligand']
    ligand = torch.as_tensor(q, dtype=torch.float32, device=device)
    pocket = torch.as_tensor(row['pocket'], dtype=torch.float32, device=device)
    return residues, ligand, pocket, logits, member, support


def train_refiner(model, rows, weights, out, args, epochs):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    rng = np.random.default_rng(args.seed)
    writer = SummaryWriter(str(out/'refiner_tb'))
    history = []
    for epoch in range(1, epochs+1):
        losses, norms = [], []
        model.train()
        order = rng.permutation(len(rows))
        for start in range(0, len(order), args.batch_size):
            ids = order[start:start+args.batch_size]
            optimizer.zero_grad(set_to_none=True)
            for i in ids:
                r, q, p, logit, member, support = boundary_inputs(rows[i], args.device)
                indices = np.arange(1, len(p))
                if not len(indices):
                    continue
                # Uniform candidate subsampling is TRAIN-only; evaluation refines all.
                indices = rng.choice(indices, min(16, len(indices)), replace=False)
                idx = torch.as_tensor(indices, device=args.device)
                target = torch.as_tensor(load_tokens(rows[i]['token_path'])['contact_target'][:len(r)],
                                         dtype=torch.float32, device=args.device)
                prediction = model(r, q, p[idx], logit, member[idx], support[idx])
                loss = F.binary_cross_entropy_with_logits(prediction, target[None].expand_as(prediction))
                loss = loss * float(weights[i])/len(ids)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite refiner loss')
                loss.backward()
                losses.append(float(loss.detach())*len(ids))
            norms.append(float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)))
            optimizer.step()
        record = dict(epoch=epoch, train_loss=float(np.mean(losses)), grad_norm=float(np.mean(norms)))
        history.append(record)
        write_csv(out/'refiner_history.csv', history)
        for k, v in record.items():
            writer.add_scalar(k, v, epoch)
        writer.flush()
        print('REFINER', json.dumps(record), flush=True)
    writer.close()
    torch.save(dict(model_state_dict=model.state_dict(), epochs=epochs, selection='fixed TRAIN epoch, no VAL oracle'), out/'refiner_fixed.pt')
    model.eval().requires_grad_(False)


def render_refined(model, rows, args, queries=None, output=None):
    rendered = []
    model.eval()
    queries = queries if queries is not None else [None]*len(rows)
    with torch.inference_mode():
        for row, query in zip(rows, queries):
            data = load_tokens(row['token_path'])
            r, q, pocket, logit, member, support = boundary_inputs(row, args.device, query)
            masks, stats, pooled = [], [], []
            original = np.asarray(data['probability']) >= args.gate
            for start in range(0, len(pocket), 8):
                odds = model(r, q, pocket[start:start+8], logit, member[start:start+8], support[start:start+8])
                probs = odds.sigmoid().cpu().numpy()
                for j, prefix in enumerate(probs):
                    k = start+j
                    p = data['probability'].copy()
                    if k:
                        s = support[k].cpu().numpy()
                        visible = p[:len(r)]
                        visible[s] = prefix[s]
                    indices = np.flatnonzero(p >= args.gate).astype(np.int32)
                    masks.append(indices)
                    st = candidate_stats(data['ca'], p, original, data['observed'], [indices])[0]
                    st[-1] = float(k == 0)
                    stats.append(st)
                    visible = indices[indices < len(r)]
                    pooled.append(data['residue_tokens'][visible].astype(np.float32).mean(0) if len(visible) else np.zeros(r.shape[1], np.float32))
            f1, tp, sizes = candidate_targets(masks, data['contact_target'])
            if abs(float(f1[0])-float(row['quality_f1'][0])) > 1e-6:
                raise ValueError('Refiner changed original fallback')
            new = dict(row, pocket=np.asarray(pooled, np.float16), stats=np.asarray(stats, np.float32),
                       quality_f1=f1, true_positives=tp, sizes=sizes,
                       ligand=row['ligand'] if query is None else query['ligand'])
            rendered.append(new)
            if output is not None:
                output.mkdir(parents=True, exist_ok=True)
                idx, off = pack_candidates(masks)
                np.savez_compressed(output/f"{row['sample_id']}.npz", candidate_indices=idx, candidate_offsets=off,
                                    quality_f1=f1, true_positives=tp, sizes=sizes, sample_id=row['sample_id'])
    return rendered


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cache', type=Path, required=True)
    ap.add_argument('--splits', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--branch', choices=('gain', 'tokens', 'boundary'), required=True)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--margin', type=float, default=.02)
    ap.add_argument('--seed', type=int, default=44)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    if args.epochs < 2 or args.batch_size < 1 or args.lr <= 0 or args.margin < 0:
        raise ValueError('Invalid hyperparameters')
    torch.set_num_threads(2)
    rows, meta = read_cache(args.cache, args.splits, args.smoke)
    if not meta.get('token_cache'):
        raise ValueError('Need audited token cache for all branch controls')
    for subset in rows:
        for row in rows[subset]:
            row['token_path'] = str(args.cache/subset/f"{row['sample_id']}.npz")
    args.gate = float(meta['gate'])
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/'config.json', dict(args={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        cache=meta, source_sha256={p: sha256_file(ROOT/p) for p in
        ('scripts/train/train_site_parallel.py', 'src/models/site_parallel_heads.py', 'scripts/diagnostics/site_quality_features.py')},
        warning='Frozen-backbone in-sample stacking, not OOF. VAL only. Added-query ablations are not full ligand dependency.'))
    modes = {'gain': ('gain', 'paired_absolute'), 'tokens': ('tokens', 'mean', 'blind'),
             'boundary': ('ligand_refiner', 'protein_refiner')}[args.branch]
    results = []
    for mode in modes:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        train, val = rows['train'], rows['val']
        mean, std, weights = fit_stats(train)
        dim = train[0]['pocket'].shape[1]
        out = args.output/mode
        blind = mode == 'protein_refiner'
        refiner = None
        epochs = args.epochs
        if args.branch == 'boundary':
            out.mkdir()
            refiner = BoundaryResidualHead(dim, not blind).to(args.device)
            train_refiner(refiner, train, weights, out, args, args.epochs//2)
            train = render_refined(refiner, train, args, output=out/'masks_train')
            val = render_refined(refiner, val, args, output=out/'masks_val')
            mean, std, weights = fit_stats(train)
            model = SiteQualityHead('ligand_site', dim, mean, std).to(args.device)
            epochs -= args.epochs//2
            rank_out = out/'quality'
        else:
            rank_out = out
            model = (PairedQualityHead(dim, mean, std, mode == 'gain') if args.branch == 'gain'
                     else TokenQualityHead(dim, mean, std, mode)).to(args.device)
        best_epoch = train_ranker(model, train, val, weights, rank_out, args, epochs, blind)
        for query_mode in ('real', 'permuted', 'zero'):
            queries = query_list(rows['val'], query_mode)
            current_val = val
            if refiner is not None and query_mode != 'real':
                current_val = render_refined(refiner, rows['val'], args, queries, out/f'masks_val_{query_mode}')
                # The substituted query is already stored in each rendered row.
                queries = None
            metrics, detail = evaluate(model, current_val, args.device, args.margin, queries, blind)
            write_csv(out/f'val_{query_mode}.csv', detail)
            results.append(dict(mode=mode, query_mode=query_mode, best_epoch=best_epoch,
                                parameters=sum(p.numel() for p in model.parameters()), **metrics))
        write_json(args.output/'summary.json', dict(complete=False, split='val', results=results, smoke=args.smoke))
    write_csv(args.output/'summary.csv', results)
    write_json(args.output/'summary.json', dict(complete=True, split='val', results=results, smoke=args.smoke,
        warning='Exploratory model-selected VAL. Query-only controls hold real backbone/proposals fixed.'))
    print('SITE_PARALLEL_COMPLETE', args.output, flush=True)


if __name__ == '__main__':
    main()
