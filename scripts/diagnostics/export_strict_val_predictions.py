#!/usr/bin/env python3
"""Frozen checkpoint VAL export, reproducing training INPUTS, not broken labels.

No re-normalization of already normalized training caches; no new gate fitting.
Keep the trained sequence/atom budgets, but export the FULL protein label/CA
axis and explicitly flag residues the model did not observe. Fail on internal
axis shifts, missing rows, nonfinite inputs, or split mismatch.
"""

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/infer"))
from infer_v11_internal_ligdep import create_v11_model, model_needs_unimol
from scripts.infer.benchmark_integrity import residue_keys, require_finite, safe_tensor_load, sha256_file
from scripts.diagnostics.strict_assignment_common import digest_object, read_strict_splits, write_csv, write_json
from scripts.diagnostics.prepare_internal_label_alignment import aligned_values, load_overlay
from src.data.pocket_dataset_v8 import PocketDatasetV8
from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11


def validate_axis(protein, esm, distance):
    ca = torch.as_tensor(protein["ca_coords"])
    keys = residue_keys(protein, len(ca))
    skeys = residue_keys(esm, len(esm["embeddings"]))
    if keys[:min(len(keys), len(skeys))] != skeys[:min(len(keys), len(skeys))]:
        raise ValueError("ESM/protein positional axis shift; cannot claim training-loader reproduction with aligned input")
    values = distance["distances"] if isinstance(distance, dict) else distance
    values = torch.as_tensor(values)
    if values.shape != (len(ca),):
        raise ValueError("Full protein/label length mismatch; do not silently clip")
    if isinstance(distance, dict) and "distances_vdw_adjusted" in distance:
        raise ValueError("Expected strict merged dcenter labels, not legacy vdw labels")
    require_finite(ca, "full CA")
    require_finite(values, "distance labels")
    return keys, ca, values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected-val", type=int, default=647)
    ap.add_argument("--label-alignment", type=Path)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Run the GPU export inside a scheduled GPU allocation")
    torch.set_num_threads(1)
    ckpt = safe_tensor_load(args.checkpoint)
    config = copy.deepcopy(ckpt["config"])
    d = config["data"]
    split = args.data_root / "splits.txt"
    groups = read_strict_splits(split)
    label_overlay = load_overlay(args.label_alignment, split, groups["val"])
    original_split = Path(d["split_file"])
    if sha256_file(split) != sha256_file(original_split):
        raise ValueError("Requested split differs from checkpoint training split")
    if len(groups["val"]) != args.expected_val:
        raise ValueError("Unexpected strict val population")
    gate = float(ckpt["val_contact_gate"])
    if not 0. <= gate <= 1.:
        raise ValueError("Invalid checkpoint gate")
    dataset = PocketDatasetV8(
        data_root=str(args.data_root), pdb_ids=groups["val"], split="val", use_nma=False,
        distances_file=str(args.data_root / "residue_ligand_distances_dcenter.pt"),
        esm_features_dir=d["esm_features_dir"], esm3_features_dir=d.get("esm3_features_dir"),
        esm_alignment_features_dir=d.get("esm_alignment_features_dir"),
        protein_features_dir=d["protein_features_dir"], nma_features_dir=d["nma_features_dir"],
        ligand_features_dir=d["ligand_features_dir"], nma_max_conf=0, nma_n_confs=1,
        max_seq_len=int(d["max_seq_len"]), max_atoms=int(d["max_atoms"]),
        atom_truncation_policy=d.get("atom_truncation_policy", "head"),
        label_contact_thresh=float(d["label_contact_thresh"]), label_near_thresh=float(d["label_near_thresh"]),
        use_spatial_clamp=bool(d.get("use_spatial_clamp", False)),
        spatial_clamp_dims=d.get("spatial_clamp_dims"), spatial_clamp_range=d.get("spatial_clamp_range"))
    if dataset.pdb_ids != groups["val"]:
        raise ValueError("Dataset silently filtered/reordered requested VAL rows")
    model, version = create_v11_model(ckpt, config, torch.device("cuda"))
    if model_needs_unimol(model):
        raise ValueError("This diagnostic requires the verified no-UniMol baseline")
    # Metadata-only reference retained after releasing optimizer state from the checkpoint.
    ckpt_epoch = ckpt.get("epoch")
    del ckpt
    args.output.mkdir(parents=True, exist_ok=True)
    pred_dir = args.output / "predictions"
    pred_dir.mkdir(exist_ok=True)
    if any(pred_dir.glob("*.npz")):
        raise ValueError("Use a fresh output directory; partial outputs are never silently reused")
    manifest, metric_rows = [], []
    with torch.inference_mode():
        for i, sid in enumerate(groups["val"]):
            protein_path = dataset._find_base_protein_file(sid)
            esm_path = dataset._find_primary_esm_file(sid)
            protein, esm = safe_tensor_load(protein_path), safe_tensor_load(esm_path)
            raw_distance = dataset.distances_dict[sid]
            aligned_distance = aligned_values(sid, protein, raw_distance, label_overlay)
            keys, ca, distance = validate_axis(protein, esm, aligned_distance)
            sample = dataset[i]
            for name, value in sample.items():
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    require_finite(value, f"{sid}/{name}")
            n = sample["n_residues"]
            expected_n = min(len(ca), len(esm["embeddings"]), int(d["max_seq_len"]))
            if n != expected_n or not torch.equal(sample["gvp_coords"], ca[:n].float()):
                raise ValueError(f"Unexpected feature truncation for {sid}")
            labels = np.where(distance.numpy() < d["label_contact_thresh"], 0,
                              np.where(distance.numpy() < d["label_near_thresh"], 1, 2)).astype(np.int64)
            raw_values = np.asarray(raw_distance["distances"] if isinstance(raw_distance, dict) else raw_distance)
            legacy_labels = np.where(raw_values[:n] < d["label_contact_thresh"], 0,
                                     np.where(raw_values[:n] < d["label_near_thresh"], 1, 2))
            if not np.array_equal(sample["labels"].numpy(), legacy_labels):
                raise ValueError(f"Original training-label reproduction mismatch: {sid}")
            inputs = prepare_model_inputs_v11(pocket_collate_fn_v10([sample]), torch.device("cuda"))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                result = model(**inputs)
            logits = result["logits"] if isinstance(result, dict) else result
            require_finite(logits, sid)
            p = torch.softmax(logits.float(), -1)[0, :n, 0].cpu().numpy()
            full_p = np.zeros(len(ca), np.float32)
            full_p[:n] = p
            observed = np.arange(len(ca)) < n
            pred = observed & (full_p >= gate)
            protein_key = digest_object([keys, protein["residue_names"], np.round(ca.numpy(), 3).tolist()])
            np.savez_compressed(pred_dir / f"{sid}.npz", ca_coords=ca.numpy(), labels=labels,
                                contact_probability=full_p, contact_prediction=pred,
                                observed=observed, residue_keys=np.asarray(keys, dtype=str),
                                protein_key=np.asarray(protein_key))
            from scripts.diagnostics.benchmark_candidate_audit import binary_f1
            metric_rows.append({"sample_id": sid, "n_full": len(ca), "n_observed": n,
                                "label_overlay_used": sid in label_overlay,
                                "changed_contact_labels": int(((labels[:n] == 0) != (legacy_labels == 0)).sum()),
                                "legacy_model_f1": binary_f1(pred[:n], legacy_labels == 0),
                                "model_f1": binary_f1(pred[:n], labels[:n] == 0),
                                "full_axis_f1": binary_f1(pred, labels == 0),
                                "unobserved_contacts": int(((labels == 0) & ~observed).sum())})
            manifest.append({"sample_id": sid, "protein": str(protein_path.resolve()), "esm": str(esm_path.resolve()),
                             "n_full": len(ca), "n_observed": n, "n_esm": len(esm["embeddings"])})
            if (i + 1) % 25 == 0:
                print(f"VAL {i+1}/{len(groups['val'])}", flush=True)
    write_csv(args.output / "per_sample.csv", metric_rows)
    write_json(args.output / "manifest.json", manifest)
    write_json(args.output / "export_summary.json", {
        "split": "val", "n": len(metric_rows), "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint), "checkpoint_epoch": ckpt_epoch,
        "split_sha256": sha256_file(split), "version": version, "gate": gate, "precision": "bf16",
        "max_atoms": d["max_atoms"], "atom_truncation_policy": dataset.atom_truncation_policy,
        "max_seq_len": d["max_seq_len"], "full_axis_proposals": True,
        "model_f1": float(np.mean([r["model_f1"] for r in metric_rows])),
        "legacy_model_f1": float(np.mean([r["legacy_model_f1"] for r in metric_rows])),
        "label_overlay_rows": sum(r["label_overlay_used"] for r in metric_rows),
        "label_overlay_sha256": sha256_file(args.label_alignment) if args.label_alignment else None,
        "full_axis_f1": float(np.mean([r["full_axis_f1"] for r in metric_rows])),
        "truncated_rows": sum(r["n_full"] != r["n_observed"] for r in metric_rows),
        "unobserved_contacts": sum(r["unobserved_contacts"] for r in metric_rows),
        "normalization": "reuse prepared training caches without renormalization",
        "label_axis_evidence": "prepared-cache axes; mismatch overlays bound to full source-PDB/SDF distance verification; legacy-label F1 also reported"})
    print("STRICT_VAL_EXPORT_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
