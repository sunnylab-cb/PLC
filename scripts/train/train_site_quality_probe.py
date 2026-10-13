#!/usr/bin/env python3
"""Train small quality heads from scratch on frozen TRAIN features, select on VAL.

No backbone updates, resume, native ligand pose, hard-negative assumption,
threshold tuning, or test/external benchmark access. Query permutation/zero is
head-only ablation with fixed real-pair proposals, NOT full ligand dependency.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.models.site_quality_head import SiteQualityHead, quality_loss, select_with_fallback
from scripts.diagnostics.site_quality_features import STAT_NAMES
from scripts.diagnostics.strict_assignment_common import read_strict_splits, write_csv, write_json
from scripts.infer.benchmark_integrity import sha256_file


def read_cache(root, split_path, smoke=False):
    meta = json.loads((root / "summary.json").read_text())
    splits = read_strict_splits(split_path)
    if not meta["complete"] or meta["smoke"] != smoke or meta["split_sha256"] != sha256_file(split_path):
        raise ValueError("Incomplete, wrong-split, or smoke/full cache mismatch")
    if meta["stats_names"] != list(STAT_NAMES) or meta["manifest_sha256"] != sha256_file(root / "manifest.json"):
        raise ValueError("Cache schema/manifest mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    rows = {"train": [], "val": []}
    seen = set()
    for item in manifest:
        subset, sid = item["subset"], item["sample_id"]
        if subset not in rows or sid not in splits[subset] or sid in seen or Path(sid).name != sid:
            raise ValueError("Cache manifest split overlap, duplicate, or wrong sample")
        seen.add(sid)
        path = root / subset / f"{sid}.npz"
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"Changed cache: {sid}")
        with np.load(path, allow_pickle=False) as a:
            if str(a["sample_id"]) != sid or str(a["subset"]) != subset:
                raise ValueError("Cache identity mismatch")
            for field in ('inference_protocol', 'acceptance_policy'):
                if field in meta and (field not in a or str(a[field]) != meta[field]):
                    raise ValueError(f"Mixed inference protocols/policies in cache: {sid} ({field})")
            row = {k: np.array(a[k]) for k in ("pocket", "ligand", "stats", "quality_f1", "sizes", "true_positives")}
            row.update(sample_id=sid, protein_key=str(a["protein_key"]), axis_key=str(a["axis_key"]), n_true=int(a["n_true"]))
        n, d = row["pocket"].shape
        if row["ligand"].shape != (d,) or row["stats"].shape != (n, len(STAT_NAMES)) or row["quality_f1"].shape != (n,):
            raise ValueError("Malformed cache shapes")
        if not all(np.isfinite(row[k]).all() for k in ("pocket", "ligand", "stats", "quality_f1")):
            raise ValueError("Nonfinite cache")
        if not n or row["stats"][0, -1] != 1 or np.any(row["stats"][1:, -1] != 0):
            raise ValueError("Original-prediction fallback is missing")
        if np.any((row["quality_f1"] < 0) | (row["quality_f1"] > 1)):
            raise ValueError("Invalid supervision")
        sizes, tp = row["sizes"], row["true_positives"]
        if sizes.shape != (n,) or tp.shape != (n,) or np.any(sizes < 0) or np.any(tp < 0) or np.any(tp > sizes) or np.any(tp > row["n_true"]):
            raise ValueError("Invalid contact counts")
        denominator = sizes + row["n_true"]
        expected = np.divide(2. * tp, denominator, out=np.zeros(n), where=denominator > 0)
        if not np.allclose(row["quality_f1"], expected, atol=1e-6):
            raise ValueError("Quality targets do not match GT contact counts")
        rows[subset].append(row)
    for subset in rows:
        if len(rows[subset]) != meta["counts"][subset]:
            raise ValueError("Cache row count mismatch")
        if not smoke and {r["sample_id"] for r in rows[subset]} != set(splits[subset]):
            raise ValueError("Full cache must include every requested row")
        rows[subset].sort(key=lambda r: r["sample_id"])
        if not rows[subset]:
            raise ValueError("Empty cache split")
    return rows, meta


def fit_stats(train):
    counts = Counter(r["axis_key"] for r in train)
    weights = np.array([1. / counts[r["axis_key"]] for r in train])
    weights /= weights.mean()
    means = np.array([r["stats"].mean(0, dtype=np.float64) for r in train])
    squares = np.array([np.square(r["stats"].astype(np.float64)).mean(0) for r in train])
    mean = np.average(means, axis=0, weights=weights)
    std = np.sqrt(np.maximum(0, np.average(squares, axis=0, weights=weights) - mean ** 2))
    return mean.astype(np.float32), np.maximum(std, .01).astype(np.float32), weights.astype(np.float32)


def make_batch(rows, device, queries=None, group_weights=None):
    b, max_k, dim = len(rows), max(len(r["pocket"]) for r in rows), rows[0]["pocket"].shape[1]
    batch = {"pocket": torch.zeros(b, max_k, dim), "stats": torch.zeros(b, max_k, len(STAT_NAMES)),
             "target": torch.zeros(b, max_k), "valid": torch.zeros(b, max_k, dtype=torch.bool),
             "ligand": torch.tensor(np.stack(queries if queries is not None else [r["ligand"] for r in rows]), dtype=torch.float32),
             "weights": torch.tensor(group_weights if group_weights is not None else np.ones(b), dtype=torch.float32)}
    for i, row in enumerate(rows):
        n = len(row["pocket"])
        for dst, src in (("pocket", "pocket"), ("stats", "stats"), ("target", "quality_f1")):
            batch[dst][i, :n] = torch.from_numpy(row[src].astype(np.float32))
        batch["valid"][i, :n] = True
    return {k: v.to(device) for k, v in batch.items()}


def evaluation(model, rows, device, batch_size, margin, query_mode="real"):
    model.eval()
    output = []
    queries = [r["ligand"] for r in rows]
    if query_mode == "permuted":
        # Random cyclic ordering gives a deterministic derangement, not chemistry matching.
        order = np.random.default_rng(44).permutation(len(rows))
        donor = np.empty(len(rows), dtype=int)
        donor[order] = np.roll(order, 1)
        queries = [rows[j]["ligand"] for j in donor]
    elif query_mode == "zero":
        queries = [np.zeros_like(q) for q in queries]
    elif query_mode != "real":
        raise ValueError(query_mode)
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            part = rows[start:start + batch_size]
            b = make_batch(part, device, queries[start:start + batch_size])
            scores = model(b["pocket"], b["ligand"], b["stats"])
            selected = select_with_fallback(scores, b["valid"], margin).cpu().tolist()
            raw = select_with_fallback(scores, b["valid"], 0.).cpu().tolist()
            scores = scores.cpu().numpy()
            for i, row in enumerate(part):
                j, f1, n = selected[i], row["quality_f1"], len(row["quality_f1"])
                output.append({"sample_id": row["sample_id"], "protein_key": row["protein_key"],
                               "selected": j, "baseline_f1": float(f1[0]), "selected_f1": float(f1[j]),
                               "unguarded_f1": float(f1[raw[i]]), "oracle_f1": float(f1.max()),
                               "selected_size": int(row["sizes"][j]), "original_size": int(row["sizes"][0]),
                               "true_positives": int(row["true_positives"][j]), "n_true": row["n_true"],
                               "predicted_quality": float(scores[i, j]), "original_predicted_quality": float(scores[i, 0]),
                               "quality_mae": float(np.abs(scores[i, :n] - f1).mean()),
                               "within002_of_oracle": float(f1[j] >= f1.max() - .02)})
    return summarize_selection(output), output


def summarize_selection(output):
    groups = defaultdict(list)
    for row in output:
        groups[row["protein_key"]].append(row)
    tp, size, true = [sum(r[k] for r in output) for k in ("true_positives", "selected_size", "n_true")]
    result = {"n": len(output), "n_proteins": len(groups),
              "sample_f1": float(np.mean([r["selected_f1"] for r in output])),
              "protein_f1": float(np.mean([np.mean([r["selected_f1"] for r in rs]) for rs in groups.values()])),
              "baseline_f1": float(np.mean([r["baseline_f1"] for r in output])),
              "unguarded_f1": float(np.mean([r["unguarded_f1"] for r in output])),
              "oracle_f1": float(np.mean([r["oracle_f1"] for r in output])),
              "micro_f1": 2 * tp / max(1, size + true), "micro_precision": tp / max(1, size), "micro_recall": tp / max(1, true),
              "changed": sum(r["selected"] != 0 for r in output),
              "improved": sum(r["selected_f1"] > r["baseline_f1"] for r in output),
              "worsened": sum(r["selected_f1"] < r["baseline_f1"] for r in output),
              "size_p50": float(np.median([r["selected_size"] for r in output])),
              "size_p90": float(np.percentile([r["selected_size"] for r in output], 90)),
              "quality_mae": float(np.mean([r["quality_mae"] for r in output])),
              "within002_of_oracle": float(np.mean([r["within002_of_oracle"] for r in output]))}
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--margin", type=float, default=.02)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.margin < 0:
        raise ValueError("Invalid training arguments")
    torch.set_num_threads(2)
    rows, meta = read_cache(args.cache, args.splits, args.smoke)
    train, val = rows["train"], rows["val"]
    mean, std, weights = fit_stats(train)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    write_json(args.output / "config.json", {**options, "cache": meta,
               "selector": "predicted quality(best) > predicted quality(original) + fixed margin",
               "checkpoint_selection": "highest VAL protein-balanced F1 at fixed margin; no threshold sweep",
               "stats_fit": "TRAIN-only, inverse exact-axis frequency, sample-balanced candidate moments",
               "training_code_sha256": sha256_file(__file__), "head_code_sha256": sha256_file(ROOT / "src/models/site_quality_head.py")})
    results = []
    for mode in ("stats_only", "protein_site", "ligand_site"):
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        out = args.output / mode
        out.mkdir()
        writer = SummaryWriter(str(out / "tb"))
        model = SiteQualityHead(mode, train[0]["pocket"].shape[1], mean, std).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
        best, best_epoch, history = -1., None, []
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses, norms = [], []
            order = rng.permutation(len(train))
            for start in range(0, len(train), args.batch_size):
                idx = order[start:start + args.batch_size]
                batch = make_batch([train[i] for i in idx], device, group_weights=weights[idx])
                optimizer.zero_grad(set_to_none=True)
                scores = model(batch["pocket"], batch["ligand"], batch["stats"])
                loss = quality_loss(scores, batch["target"], batch["valid"], batch["weights"])
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
                norms.append(float(norm))
            metrics, _ = evaluation(model, val, device, args.batch_size, args.margin)
            record = {"epoch": epoch, "train_loss": float(np.mean(losses)), "grad_norm": float(np.mean(norms)), **metrics}
            history.append(record)
            for key, value in record.items():
                writer.add_scalar(key, value, epoch)
            writer.flush()
            print(mode, json.dumps(record), flush=True)
            write_csv(out / "history.csv", history)
            if metrics["protein_f1"] > best:
                best, best_epoch = metrics["protein_f1"], epoch
                torch.save({"model_state_dict": model.state_dict(), "mode": mode, "epoch": epoch,
                            "input_dim": train[0]["pocket"].shape[1], "margin": args.margin,
                            "cache_summary_sha256": sha256_file(args.cache / "summary.json"),
                            "stats_mean": torch.tensor(mean), "stats_std": torch.tensor(std), "metrics": metrics}, out / "best_val_head.pt")
        writer.close()
        checkpoint = torch.load(out / "best_val_head.pt", map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        for query_mode in (("real", "permuted", "zero") if mode == "ligand_site" else ("real",)):
            metrics, detail = evaluation(model, val, device, args.batch_size, args.margin, query_mode)
            write_csv(out / f"val_{query_mode}.csv", detail)
            groups = defaultdict(list)
            for row in detail:
                groups[row["protein_key"]].append(row["selected_f1"] - row["baseline_f1"])
            delta = np.array([np.mean(groups[k]) for k in sorted(groups)])
            boot_rng = np.random.default_rng(44)
            samples = [boot_rng.choice(delta, len(delta), replace=True).mean() for _ in range(5000)]
            results.append({"mode": mode, "query_mode": query_mode, "best_epoch": best_epoch,
                            "n_parameters": sum(p.numel() for p in model.parameters()),
                            "protein_delta": float(delta.mean()), "delta_ci025": float(np.quantile(samples, .025)),
                            "delta_ci975": float(np.quantile(samples, .975)), **metrics})
    write_csv(args.output / "summary.csv", results)
    write_json(args.output / "summary.json", {"results": results, "smoke": args.smoke,
               "split": "val", "warning": "Model-selected exploratory VAL; bootstrap is descriptive, not across seeds/clusters. Query ablations alter only head queries with real-pair candidates fixed, not full-model ligand dependency."})
    print("SITE_QUALITY_PROBE_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
