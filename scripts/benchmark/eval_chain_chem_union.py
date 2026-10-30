#!/usr/bin/env python3
"""Custom HOLO4K: one receptor chain x exact chemical ligand, observed-copy union.

Not a replacement for published HOLO4K/per-copy scores. Infer ALL chain queries;
>=3-contact and zero-contact strata are reported separately, never used to
select inference chains. No atom cap, no benchmark gate/checkpoint tuning.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.data.chain_chem_union import (
    UnionBuilder, chemical_identity, geometry_digest, normalized_keys, union_sample, TASK_ID,
)
from scripts.benchmark.eval_holo_chain_context import chain_indices
from scripts.diagnostics.recompute_joint_topk_reference import Aggregate, score_row, metrics, sha, write_json
from scripts.infer.infer_v11_benchmark import BenchmarkV11Dataset, _load_cohort_norm, benchmark_collate
from scripts.infer.infer_v11_internal_ligdep import create_v11_model, model_needs_unimol


def make_parent(root, stats):
    ligand_norm, spatial_norm = _load_cohort_norm(stats)
    return BenchmarkV11Dataset(benchmark_root=root, dataset="holo4k",
        tag_file=root / "residue_ligand_tags/atom_h/holo4k_per_ligand_dmin.pt",
        protein_feature_root=root / "features_fixed_conservative/protein_v4",
        esm_root=root / "esm2", esm3_root=None, ligand_feature_root=root / "ligand_features_v5",
        contact_thresh=5., near_thresh=8., max_seq_len=2**31 - 1, max_atoms=-1,
        tag_mode="per_ligand", ligand_mode="real", ligand_norm=ligand_norm, spatial_norm=spatial_norm,
        spatial_clamp_dims=[62, 76, 78, 79], spatial_clamp_range=[-5., 5.],
        atom_truncation_policy="alternating_ends", residue_alignment="keys", residue_coverage="protein",
        tag_protein_root=root / "p2rank-datasets-fixed_atom_h", nonfinite_policy="sasa_zero")


def prepare(args):
    args.work.mkdir(parents=True, exist_ok=False)
    base = make_parent(args.benchmark, args.stats)
    builder, errors, identities = UnionBuilder(), {}, Counter()
    for i, sid in enumerate(base.keys):
        try:
            sample = base[i]
            keys = normalized_keys(sample["residue_keys"])
            if len(keys) != sample["n_residues"]:
                raise ValueError("Truncated/unaligned parent axis")
            sdf = args.benchmark / "ligands_sdf/holo4k" / f"{sid}.sdf"
            try:
                chemistry = chemical_identity(sdf)
            except ValueError as exc:
                chemistry = dict(status="unresolved_singleton", reason=str(exc))
            identities[chemistry["status"]] += 1
            for chain, idx in chain_indices(keys).items():
                part_keys = [keys[j] for j in idx]
                builder.add(dict(sample_id=sid, source="holo4k", split="external", component_id="external",
                    pdb_id=sample["protein_key"], chain=chain, indices=idx.tolist(), residue_keys=part_keys,
                    distances=sample["raw_distances"][idx], parent_n=len(keys), chemistry=chemistry,
                    geometry_sha256=geometry_digest(part_keys, sample["gvp_coords"][idx])))
        except Exception as exc:
            errors[sid] = f"{type(exc).__name__}: {exc}"
        if (i + 1) % 100 == 0:
            print("HOLO_UNION_PREP", i + 1, len(base), "failures", len(errors), flush=True)
    manifest = builder.finish()
    torch.save(manifest, args.work / "union.pt")
    assets = {str(path): sha(path) for path in (
        args.stats, args.benchmark / "residue_ligand_tags/atom_h/holo4k_per_ligand_dmin.pt",
        args.work / "union.pt")}
    report = dict(complete=not errors, task_id=TASK_ID, n_parent_pairs=len(base), failed=errors,
        n_chain_queries=len(manifest["rows"]), n_merged=sum(r["n_members"] > 1 for r in manifest["rows"]),
        n_eligible=sum(r["eligible"] for r in manifest["rows"]),
        n_zero=sum(r["n_contact"] == 0 for r in manifest["rows"]), chemistry=dict(identities),
        benchmark=str(args.benchmark), stats=str(args.stats), assets=assets, policy=manifest["policy"],
        no_gt_chain_selection=True, max_atoms=-1, external_si30_certified=False,
        note="Any failed parent blocks inference; inspect exclusions explicitly, never silently drop.")
    write_json(args.work / "preparation.json", report)
    print("HOLO_UNION_PREPARED", json.dumps({k: v for k, v in report.items() if k != "assets"}), flush=True)
    if errors:
        raise RuntimeError("HOLO union QC has failed parents; see preparation.json")


class BenchmarkUnionView(torch.utils.data.Dataset):
    def __init__(self, parent, manifest):
        self.parent, self.manifest = parent, manifest
        self.parents = {sid: i for i, sid in enumerate(parent.keys)}
        self.rows = manifest["rows"]
        if manifest["task_id"] != TASK_ID:
            raise ValueError("Wrong benchmark target version")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return union_sample(self.parent[self.parents[row["representative"]]], row, self.manifest["cap"])


def identity_collate(samples):
    return samples


def summarize(records, gate):
    from sklearn.metrics import average_precision_score, roc_auc_score
    aggregates, scores, targets, proteins = defaultdict(Aggregate), defaultdict(list), defaultdict(list), defaultdict(list)
    for row, prob, y in records:
        groups = ["all_queries"]
        groups.append("eligible_ge3" if row["eligible"] else ("zero_contact" if row["n_contact"] == 0 else "one_two_contact"))
        rules = score_row(prob, y, gate)
        for group in groups:
            scores[group].append(prob)
            targets[group].append(y)
            for rule, counts in rules.items():
                aggregates[(group, rule)].add(counts)
                proteins[(group, rule, row["pdb_id"])].append(
                    metrics(counts["n"], counts["positive"], counts["predicted"], counts["tp"]))
    result = {}
    for group in scores:
        p, y = np.concatenate(scores[group]), np.concatenate(targets[group])
        rules = {}
        for (g, rule), a in aggregates.items():
            if g != group:
                continue
            entry = a.finish()
            per_protein = [v for (gg, rr, _), v in proteins.items() if gg == group and rr == rule]
            entry["protein_mean_of_ligand_mean"] = {
                m: float(np.mean([np.mean([r[m] for r in rows]) for rows in per_protein]))
                for m in ("f1", "mcc", "precision", "recall")}
            entry["protein_mean_of_ligand_max"] = {
                m: float(np.mean([max(r[m] for r in rows) for rows in per_protein]))
                for m in ("f1", "mcc", "precision", "recall")}
            rules[rule] = entry
        result[group] = dict(rules=rules, auprc=float(average_precision_score(y, p)) if y.any() else None,
                            auroc=float(roc_auc_score(y, p)) if y.any() and not y.all() else None,
                            contact_count_quantiles=np.quantile([int(t.sum()) for t in targets[group]], [0, .5, .9, 1]).tolist())
    return result


@torch.inference_mode()
def infer(args):
    from src.data.collate import prepare_model_inputs_v11
    prep = json.loads((args.work / "preparation.json").read_text())
    if not prep["complete"] or prep["failed"]:
        raise ValueError("Incomplete benchmark preparation")
    for path, digest in prep["assets"].items():
        if sha(path) != digest:
            raise ValueError(f"Changed benchmark input: {path}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    if cfg["model"].get("use_esm3") or cfg["model"].get("use_compile"):
        raise ValueError("Initial custom protocol supports non-ESM3, non-compiled checkpoints")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Slurm BF16 GPU allocation required")
    gate = float(ckpt["val_contact_gate"])
    manifest = torch.load(args.work / "union.pt", map_location="cpu", weights_only=False)
    parent = make_parent(Path(prep["benchmark"]), Path(prep["stats"]))
    dataset = BenchmarkUnionView(parent, manifest)
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=args.workers,
                                         collate_fn=identity_collate)
    model, version = create_v11_model(ckpt, cfg, torch.device("cuda"))
    model.eval()
    if model_needs_unimol(model):
        raise ValueError("Unexpected UniMol input")
    args.output.mkdir(parents=True, exist_ok=False)
    records, hashes = [], {}
    for samples in loader:
        batch = benchmark_collate(samples)
        inputs = prepare_model_inputs_v11(batch, torch.device("cuda"))
        for key in ("esm_valid_mask", "esm3_valid_mask"):
            if key in batch:
                inputs[key] = batch[key].cuda()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs)["logits"]
        if not torch.isfinite(logits[inputs["protein_mask"].bool()]).all():
            raise ValueError("Nonfinite custom-benchmark logits")
        p = logits.float().softmax(-1)[..., 0].cpu().numpy()
        for i, sample in enumerate(samples):
            row = dataset.rows[len(records)]
            if sample["pdb_id"] != row["sample_id"]:
                raise ValueError("Unexpected query order")
            prob = p[i, :sample["n_residues"]]
            y = sample["labels"].numpy() == 0
            records.append((row, prob, y))
            path = args.output / f"{row['sample_id']}.npz"
            np.savez_compressed(path, probability=prob, contact=y, residue_keys=np.asarray(sample["residue_keys"], dtype=str))
            hashes[row["sample_id"]] = sha(path)
    if len(records) != len(dataset):
        raise ValueError("Incomplete custom benchmark")
    write_json(args.output / "summary.json", dict(complete=True, task_id=TASK_ID,
        model=version, checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        preparation_sha256=sha(args.work / "preparation.json"), val_gate=gate, metrics=summarize(records, gate),
        selection="reference-only; saved VAL gate, no HOLO tuning", native_per_copy_comparable=False,
        external_si30_certified=False, query_definition=manifest["policy"],
        bp_laa_not_reported="Legacy single-copy-center metrics are not valid after multi-site union",
        prediction_hashes=hashes))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("prepare", "infer"))
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--benchmark", type=Path)
    p.add_argument("--stats", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(4)
    if args.phase == "prepare":
        if not args.benchmark or not args.stats:
            p.error("prepare requires --benchmark and --stats")
        prepare(args)
    else:
        if not args.checkpoint or not args.output:
            p.error("infer requires --checkpoint and --output")
        infer(args)
