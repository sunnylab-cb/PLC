#!/usr/bin/env python3
"""Offline, chainwise LiBRe on the frozen PLC per-ligand benchmark population.

This is an external-protocol adaptation, not a reproduction of paper scores.
All chains are predicted without an oracle binding-chain choice. Long chains use
overlapping 1500-residue windows and mean probability merging. Metrics use the
same (at most 1800-residue) axis as the frozen PLC reference CSV. Native geometry
is used only for label QC and metric computation, never for model inputs.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def windows(length, width=1500, overlap=250):
    require(length > 0 and 0 <= overlap < width, "Invalid window parameters")
    if length <= width:
        return [(0, length)]
    starts = list(range(0, length - width + 1, width - overlap))
    if starts[-1] != length - width:
        starts.append(length - width)
    return [(start, start + width) for start in starts]


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def finite_mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def contact_metrics(labels, probabilities, threshold=0.6):
    from sklearn.metrics import average_precision_score, roc_auc_score

    labels = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(probabilities)
    require(labels.shape == probabilities.shape and labels.size > 0, "Metric shape mismatch")
    require(np.isfinite(probabilities).all(), "Non-finite probabilities")
    pred = probabilities > threshold
    tp = int((pred & labels).sum())
    fp = int((pred & ~labels).sum())
    fn = int((~pred & labels).sum())
    tn = int((~pred & ~labels).sum())
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "f1_c0": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "precision_c0": tp / (tp + fp) if tp + fp else 0.0,
        "recall_c0": tp / (tp + fn) if tp + fn else 0.0,
        "iou_c0": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
        "mcc_c0": (tp * tn - fp * fn) / denom if denom else 0.0,
        "ap_c0": float(average_precision_score(labels, probabilities)) if labels.any() else None,
        "auroc_c0": float(roc_auc_score(labels, probabilities)) if 0 < labels.sum() < labels.size else None,
        "n_pred_contact": int(pred.sum()), "n_true_contact": int(labels.sum()),
    }


def centers(row, probabilities, labels, coords):
    mask = probabilities > 0.6
    pred = coords[mask].mean(0) if mask.any() else coords[int(probabilities.argmax())]
    true = coords[labels].mean(0) if labels.any() else None
    for prefix, point in [("pred_contact_center", pred), ("true_contact_center", true)]:
        for axis, value in zip("xyz", point if point is not None else [None] * 3):
            row[f"{prefix}_{axis}"] = float(value) if value is not None else None
    distance = float(np.linalg.norm(pred - true)) if true is not None else None
    row["dcc_contact_center"] = distance
    row["dcc_contact_success"] = float(distance < 4) if distance is not None else None
    row["bp_f1_c0"] = row["f1_c0"] if distance is not None and distance < 4 else None


def enrich(rows):
    """Match PLC's single-centroid PDR/LAA and conditional BP definitions."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["protein_key"]].append(row)
    for group in groups.values():
        truth = [np.array([r[f"true_contact_center_{a}"] for a in "xyz"])
                 for r in group if r["true_contact_center_x"] is not None]
        for row in group:
            row.update(pdr_union_success=None, laa_success=None, laa_rank=None)
            if not truth:
                continue
            pred = np.array([row[f"pred_contact_center_{a}"] for a in "xyz"])
            distances = np.linalg.norm(np.stack(truth) - pred, axis=1)
            row["pdr_union_success"] = float(distances.min() < 4)
            if len(truth) >= 2 and row["true_contact_center_x"] is not None:
                current = np.array([row[f"true_contact_center_{a}"] for a in "xyz"])
                distance = float(np.linalg.norm(current - pred))
                rank = int((distances < distance - 1e-8).sum() + 1)
                row["laa_rank"] = rank
                row["laa_success"] = float(rank == 1 and distance < 4)


def summarize(rows):
    keys = ["f1_c0", "precision_c0", "recall_c0", "mcc_c0", "ap_c0", "auroc_c0",
            "bp_f1_c0", "pdr_union_success", "laa_success", "n_pred_contact", "n_true_contact"]
    groups = defaultdict(list)
    for row in rows:
        groups[row["protein_key"]].append(row)
    result = {"n_samples": len(rows), "n_proteins": len(groups)}
    result["sample_mean"] = {k: finite_mean([r.get(k) for r in rows]) for k in keys}
    result["protein_mean_of_ligand_mean"] = {
        k: finite_mean([finite_mean([r.get(k) for r in group]) for group in groups.values()]) for k in keys}
    result["protein_mean_of_ligand_max"] = {
        k: finite_mean([max([float(r[k]) for r in group if r.get(k) is not None and
                            math.isfinite(float(r[k]))], default=float("nan"))
                        for group in groups.values()]) for k in keys}
    result["metric_n_samples"] = {
        k: sum(r.get(k) is not None and math.isfinite(float(r[k])) for r in rows) for k in keys}
    result["contact_count_quantiles"] = {
        k: dict(zip(["min", "p50", "p90", "max"], np.quantile([r[k] for r in rows], [0, .5, .9, 1]).tolist()))
        for k in ["n_true_contact", "n_pred_contact"]} if rows else {}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--dataset", choices=["coach420", "holo4k"], required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0, help="Smoke only; zero means all reference rows")
    args = parser.parse_args()
    started = time.time()
    import torch
    import esm
    from rdkit import Chem
    from torch_geometric.data import Batch

    # Deployed beside this runner; in the repository it lives under scripts/infer.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "infer"))
    from benchmark_integrity import (safe_tensor_load, sha256_file, residue_keys,
                                     legacy_tag_axis, verify_tag_distances, align_distances)

    require(torch.cuda.is_available(), "GPU allocation required")
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    cache = args.output / "predictions"
    cache.mkdir(exist_ok=True)
    feature_root = args.root / "features/chainwise_fp32_l33_v1"
    feature_root.mkdir(parents=True, exist_ok=True)
    manifest = args.root / "manifests" / (args.revision + ".json")
    assets = json.loads(manifest.read_text())
    require(assets["commit"] == args.revision, "Manifest revision mismatch")
    for record in assets["assets"] + assets["esm2_reused"]:
        path = record.get("local_path", record.get("path"))
        require(sha256_file(path) == record["sha256"], f"Asset hash mismatch: {path}")
    tag_path = args.benchmark_root / f"residue_ligand_tags/atom_h/{args.dataset}_per_ligand_dmin.pt"
    protocol = {
        "dataset": args.dataset, "revision": args.revision, "limit": args.limit,
        "manifest_sha256": sha256_file(manifest), "runner_sha256": sha256_file(__file__),
        "integrity_sha256": sha256_file(sys.modules["benchmark_integrity"].__file__),
        "reference_csv": str(args.reference_csv), "reference_sha256": sha256_file(args.reference_csv),
        "tags_sha256": sha256_file(tag_path), "threshold": 0.6, "contact_angstrom": 5.0,
        "input": "raw ESM2 L33 FP32, residue-only tokens, fixed 1500 padding; official SMILES 40D graph",
        "chain_policy": "all chains, no oracle chain selection; overlap250 windows1500; probability mean",
        "axis": "same first min(protein_length,1800) residues as frozen PLC; not a new 1500 cap",
        "ligand_geometry": "SDF chemistry only; native coordinates used solely for label QC/metrics",
        "metric_note": "contact<5A binary, not PLC contact+near<8A binary; PDR/LAA centroid proxies; BP conditional",
        "claim": "matched-population external adaptation, NOT original-paper protocol reproduction",
    }
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = args.output / "protocol.json"
    if protocol_path.exists():
        require(json.loads(protocol_path.read_text()) == protocol, "Resume protocol changed")
    else:
        write_json(protocol_path, protocol)
    source = args.root / "source" / args.revision
    sys.path.insert(0, str(source / "src"))
    from models.libre import LiBRe
    import utils as upstream
    model = LiBRe(use_cnn_lstm=True, use_ligand=True, residue_input_dim=1280)
    checkpoint = args.root / "checkpoints" / args.revision / "model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().to("cuda")
    esm_path = args.root / "checkpoints/esm2/esm2_t33_650M_UR50D.pt"
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        esm_path.stem, torch.load(esm_path, map_location="cpu", weights_only=False),
        torch.load(esm_path.with_name(esm_path.stem + "-contact-regression.pt"),
                   map_location="cpu", weights_only=True))
    esm_model.eval().to("cuda")
    converter = alphabet.get_batch_converter()

    def embeddings(sequence):
        key = hashlib.sha256(sequence.encode()).hexdigest()
        path = feature_root / (key + ".pt")
        if path.exists():
            record = torch.load(path, map_location="cpu", weights_only=True)
            require(record["sequence"] == sequence and record["policy"] == "fp32_l33_residue_only_v1",
                    "ESM cache metadata mismatch")
            value = record["embeddings"]
        else:
            _, _, tokens = converter([("chain", sequence)])
            with torch.inference_mode():
                value = esm_model(tokens.to("cuda"), repr_layers=[33], return_contacts=False)[
                    "representations"][33][0, 1:len(sequence) + 1].cpu().clone()
            temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
            torch.save({"sequence": sequence, "policy": "fp32_l33_residue_only_v1", "embeddings": value}, temporary)
            temporary.replace(path)
        require(value.shape == (len(sequence), 1280) and value.dtype == torch.float32 and
                bool(torch.isfinite(value).all()), "Invalid ESM cache")
        return value

    with args.reference_csv.open() as handle:
        reference = list(csv.DictReader(handle))
    require(len({r["pdb_id"] for r in reference}) == len(reference), "Duplicate reference rows")
    if args.limit:
        reference = reference[:args.limit]
    tags = safe_tensor_load(tag_path)
    groups = defaultdict(list)
    for ref in reference:
        require(ref["pdb_id"] in tags and ref["ligand_key"] == ref["pdb_id"], "Reference/tag ligand mismatch")
        groups[ref["protein_key"]].append(ref)
    aa3 = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
    aa1 = dict(zip(aa3, "ACDEFGHIKLMNPQRSTVWY"))
    failures, qc, rows, all_y, all_p = [], [], [], [], []
    print(f"START {args.dataset}: {len(reference)} rows / {len(groups)} proteins", flush=True)
    for protein_number, (protein_key, refs) in enumerate(groups.items(), 1):
        stage = "protein_alignment"
        try:
            protein_path = args.benchmark_root / f"features_fixed_conservative/protein_v4/{args.dataset}/{protein_key}/{protein_key}_protein.pt"
            prot = safe_tensor_load(protein_path)
            keys = residue_keys(prot)
            n = min(len(keys), 1800)
            require(n > 0 and all(int(r["n_residues"]) == n for r in refs), "Reference residue count mismatch")
            coords = torch.as_tensor(prot["ca_coords"]).numpy()[:n]
            require(coords.shape == (n, 3) and np.isfinite(coords).all(), "Invalid reference coordinates")
            require(len(prot["residue_names"]) == len(keys), "Residue identity length mismatch")
            tag_axis = legacy_tag_axis(args.benchmark_root / f"p2rank-datasets-fixed_atom_h/{args.dataset}/{protein_key}.pdb")
            protein_sha = sha256_file(protein_path)
            chain_indices = defaultdict(list)
            for i, key in enumerate(keys):
                chain_indices[key[0]].append(i)
            # Build context from whole observed chains; evaluate only the frozen PLC axis.
            segments = []
            for chain, indices in chain_indices.items():
                sequence = "".join(aa1.get(str(prot["residue_names"][i]).upper(), "X") for i in indices)
                for start, stop in windows(len(sequence)):
                    axis = np.asarray(indices[start:stop])
                    if (axis < n).any():
                        segments.append((chain, axis, sequence[start:stop]))
        except (ValueError, OSError, KeyError) as exc:
            for ref in refs:
                failures.append({"sample_key": ref["pdb_id"], "stage": stage, "error": str(exc)})
            write_json(args.output / "failures.json", failures)
            continue

        segment_features = None
        for ref in refs:
            sample = ref["pdb_id"]
            stage = "label_ligand_qc"
            try:
                require(Path(sample).name == sample, "Unsafe sample key")
                sdf = args.benchmark_root / f"ligands_sdf/{args.dataset}/{sample}.sdf"
                tag = tags[sample]
                distances = torch.as_tensor(tag["distances"] if isinstance(tag, dict) else tag)
                verify_tag_distances(distances, tag_axis, [sdf])
                distances = align_distances(distances, tag_axis, prot).numpy()
                labels = distances[:n] < 5.0
                require(int(labels.sum()) == int(ref["n_true_contact"]), "Reference contact count mismatch")
                if labels.any():
                    reference_center = np.array([float(ref[f"true_contact_center_{a}"]) for a in "xyz"])
                    require(np.allclose(coords[labels].mean(0), reference_center, atol=1e-4, rtol=0),
                            "Reference contact center mismatch")
                mol = next((m for m in Chem.SDMolSupplier(str(sdf), removeHs=True, sanitize=True) if m is not None), None)
                require(mol is not None and mol.GetNumAtoms() > 0, "RDKit cannot sanitize ligand; no silent repair")
                smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
                graph = upstream.smiles_to_data(smiles)
                require(graph.x.shape[1] == 40 and graph.num_nodes > 0 and bool(torch.isfinite(graph.x).all()),
                        "Invalid official 40D graph")
                require(graph.edge_index.ndim == 2 and graph.edge_index.shape[0] == 2, "Invalid ligand edges")
                input_id = hashlib.sha256((fingerprint + protein_sha + tag_axis["source_sha256"] + sha256_file(sdf)).encode()).hexdigest()
                prediction_path = cache / (sample + ".npz")
                if prediction_path.exists():
                    with np.load(prediction_path, allow_pickle=False) as saved:
                        require(str(saved["input_id"]) == input_id, "Resume input mismatch")
                        require(np.array_equal(saved["labels"], labels), "Resume labels changed")
                        probabilities = saved["probabilities"].copy()
                else:
                    stage = "inference"
                    if segment_features is None:
                        segment_features = [embeddings(s) for _, _, s in segments]
                    graph_batch = Batch.from_data_list([graph]).to("cuda")
                    total, count = np.zeros(n, np.float64), np.zeros(n, np.int64)
                    with torch.inference_mode():
                        for (_, axis, sequence), feature in zip(segments, segment_features):
                            padded = torch.zeros((1, 1500, 1280), dtype=torch.float32, device="cuda")
                            padded[0, :len(sequence)] = feature.to("cuda")
                            _, logits = model(padded, graph_batch)
                            probs = logits.sigmoid()[0, :len(sequence), 0].cpu().numpy()
                            use = axis < n
                            total[axis[use]] += probs[use]
                            count[axis[use]] += 1
                    require((count > 0).all(), "Uncovered residues")
                    probabilities = (total / count).astype(np.float32)
                    require(np.isfinite(probabilities).all(), "Non-finite predictions")
                    temporary = prediction_path.with_name(prediction_path.name + f".{os.getpid()}.tmp.npz")
                    np.savez_compressed(temporary, probabilities=probabilities, labels=labels,
                                        ca_coords=coords, input_id=input_id, smiles=smiles,
                                        residue_keys=np.asarray(keys[:n], dtype=str))
                    temporary.replace(prediction_path)
                row = {"dataset": args.dataset, "pdb_id": sample, "protein_key": protein_key,
                       "ligand_key": sample, "n_residues": n, "n_atoms": graph.num_nodes}
                row.update(contact_metrics(labels, probabilities))
                centers(row, probabilities, labels, coords)
                rows.append(row)
                all_y.append(labels)
                all_p.append(probabilities)
                qc.append({"sample_key": sample, "input_id": input_id, "smiles": smiles,
                           "n_residues_full": len(keys), "n_residues_evaluated": n,
                           "n_true_contact_excluded_by_reference_cap": int((distances[n:] < 5).sum()),
                           "n_chains_full": len(chain_indices), "n_windows": len(segments),
                           "max_chain_length": max(map(len, chain_indices.values())),
                           "n_unknown_residues": sum(s.count("X") for _, _, s in segments),
                           "label_geometry_verified": True, "reference_counts_center_verified": True})
            except (ValueError, OSError, KeyError) as exc:
                failures.append({"sample_key": sample, "stage": stage, "error": str(exc)})
                print(f"REJECT {sample}: {stage}: {exc}", flush=True)
                write_json(args.output / "failures.json", failures)
        if protein_number % 10 == 0 or protein_number == len(groups):
            progress = {"proteins_processed": protein_number, "proteins_total": len(groups),
                        "successful_rows": len(rows), "failed_rows": len(failures),
                        "elapsed_seconds": time.time() - started}
            write_json(args.output / "progress.json", progress)
            print(json.dumps(progress), flush=True)

    enrich(rows)
    if rows:
        with (args.output / "samples.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    good_keys = {r["pdb_id"] for r in rows}
    matched_reference = []
    for ref in reference:
        if ref["pdb_id"] not in good_keys:
            continue
        converted = {k: v if k in ["dataset", "pdb_id", "protein_key", "ligand_key"] else float(v)
                     for k, v in ref.items()}
        # Recompute candidate-center diagnostics on the identical surviving set.
        converted = clean_json(converted)
        matched_reference.append(converted)
    enrich(matched_reference)
    summary = {"complete": len(rows) == len(reference) and not failures,
               "is_smoke_subset": bool(args.limit), "n_expected": len(reference), "n_failed": len(failures),
               "protocol": protocol, "libre": summarize(rows),
               "plc_frozen_reference_same_subset": summarize(matched_reference),
               "elapsed_seconds": time.time() - started, "job_id": os.environ.get("SLURM_JOB_ID"),
               "cuda_device": torch.cuda.get_device_name(),
               "cuda_peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3}
    if rows:
        summary["pooled_contact_metrics"] = contact_metrics(np.concatenate(all_y), np.concatenate(all_p))
    write_json(args.output / "qc.json", qc)
    write_json(args.output / "failures.json", failures)
    write_json(args.output / "summary.json", summary)
    print(json.dumps(clean_json(summary), indent=2), flush=True)
    if not summary["complete"]:
        raise SystemExit("Partial evaluation: see failures.json; do not claim complete coverage")


if __name__ == "__main__":
    main()
