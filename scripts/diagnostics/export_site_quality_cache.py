#!/usr/bin/env python3
"""Export frozen V11m13 candidate features for TRAIN/VAL only, never TEST.

Protein keys and ligand queries are pooled BEFORE cross-attention. Candidate
masks still use real-pair predictions, so query swaps on the cached head are
NOT full-model ligand-dependency tests. Source labels/features stay untouched.
"""

import argparse
import copy
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.export_strict_val_predictions import (
    create_v11_model, model_needs_unimol, validate_axis, PocketDatasetV8,
    pocket_collate_fn_v10, prepare_model_inputs_v11,
)
from scripts.diagnostics.prepare_internal_label_alignment import aligned_values, load_overlay
from scripts.diagnostics.strict_assignment_common import digest_object, read_strict_splits, write_csv, write_json
from scripts.diagnostics.site_quality_features import candidate_inputs, candidate_targets, pack_candidates, STAT_NAMES
from scripts.infer.benchmark_integrity import require_finite, safe_tensor_load, sha256_file
from scripts.diagnostics.site_inference_precision import (
    LEGACY, STABLE, PROTOCOLS, LIMITS, inference_precision, input_fingerprint,
    load_precision_approval, ACCEPTANCE_V1, ACCEPTANCE_POLICIES, repeat_disagreement_exceeded,
    binary_disagreement_exceeded,
)


def prepared_dataset(root, ids, d):
    return PocketDatasetV8(
        data_root=str(root), pdb_ids=ids, split="val", use_nma=False,
        distances_file=str(root / "residue_ligand_distances_dcenter.pt"),
        esm_features_dir=d["esm_features_dir"], esm3_features_dir=d.get("esm3_features_dir"),
        esm_alignment_features_dir=d.get("esm_alignment_features_dir"),
        protein_features_dir=d["protein_features_dir"], nma_features_dir=d["nma_features_dir"],
        ligand_features_dir=d["ligand_features_dir"], nma_max_conf=0, nma_n_confs=1,
        max_seq_len=int(d["max_seq_len"]), max_atoms=int(d["max_atoms"]),
        atom_truncation_policy=d.get("atom_truncation_policy", "head"),
        label_contact_thresh=float(d["label_contact_thresh"]), label_near_thresh=float(d["label_near_thresh"]),
        use_spatial_clamp=bool(d.get("use_spatial_clamp", False)),
        spatial_clamp_dims=d.get("spatial_clamp_dims"), spatial_clamp_range=d.get("spatial_clamp_range"))


def verify_independent_states(model, inputs, outputs, report_path=None):
    """Keep a strict FP32 dependency test; measure BF16 repeat noise separately.

This is not a relaxed BF16 tolerance. Both identical-input and intervened FP32
states must pass the original absolute 1e-5 bound. Failure remains fatal.
"""
    if any(module.training for module in model.modules()):
        raise ValueError("Independence audit requires every module in eval mode")
    originals = {k: v.detach().clone() for k, v in inputs.items() if torch.is_tensor(v)}
    keys = ("h_prot_global", "h_ligand_context")
    def forward(altered, amp):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(**altered)
        if any(not torch.equal(inputs[k], v) for k, v in originals.items()):
            raise ValueError("Model mutated audit inputs")
        return {k: out[k].detach().float().clone() for k in keys}
    bf16_reference = {k: outputs[k].detach().float().clone() for k in keys}
    bf16_repeat = forward(inputs, True)
    reference = forward(inputs, False)
    repeat = forward(inputs, False)
    checks = dict(policy="FP32 absolute 1e-5 invariance and repeatability; BF16 noise diagnostic only",
                  bf16_repeat_max={k: float((bf16_reference[k]-bf16_repeat[k]).abs().max()) for k in keys},
                  fp32_repeat_max={k: float((reference[k]-repeat[k]).abs().max()) for k in keys})
    failures = []
    if any(v > 1e-5 or not np.isfinite(v) for v in checks['fp32_repeat_max'].values()):
        failures.append('FP32 repeatability')
    for key, invariant, responsive in (
        ("ligand_x", "h_prot_global", "h_ligand_context"),
        ("esm_embeddings", "h_ligand_context", "h_prot_global"),
    ):
        changed = dict(inputs)
        changed[key] = torch.zeros_like(inputs[key])
        result = forward(changed, False)
        stable_error = float((reference[invariant] - result[invariant]).abs().max())
        changed_error = float((reference[responsive] - result[responsive]).abs().max())
        if not np.isfinite([stable_error, changed_error]).all() or stable_error > 1e-5 or changed_error <= 1e-5:
            failures.append(f"{key}: invariant={stable_error}, responsive={changed_error}")
        bf16_changed = forward(changed, True)
        checks[key] = {"invariant": invariant, "fp32_max_error": stable_error,
                       "responsive_fp32_max_change": changed_error,
                       "bf16_max_error": float((bf16_reference[invariant]-bf16_changed[invariant]).abs().max())}
    checks['passed'] = not failures
    checks['failures'] = failures
    if report_path is not None:
        write_json(report_path, checks)
    if failures:
        raise ValueError(f"Pre-cross independence audit failed: {failures}")
    return checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--label-alignment", type=Path, required=True)
    ap.add_argument("--reference-val", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tokens", action="store_true", help="Include once-per-pair tokens for late interaction/refinement")
    ap.add_argument("--inference-protocol", choices=PROTOCOLS, default=LEGACY)
    ap.add_argument("--precision-audit", type=Path, help="Passing full VAL audit; required for the FP32 GVP protocol")
    ap.add_argument("--acceptance-policy", choices=ACCEPTANCE_POLICIES, default=ACCEPTANCE_V1)
    args = ap.parse_args()
    if args.acceptance_policy != ACCEPTANCE_V1 and args.inference_protocol != STABLE:
        raise ValueError("Revised approval is restricted to the audited stable inference protocol")
    if not torch.cuda.is_available():
        raise RuntimeError("Use a scheduled GPU allocation")
    torch.set_num_threads(1)
    torch.manual_seed(44)
    np.random.seed(44)
    split_path = args.root / "splits.txt"
    splits = read_strict_splits(split_path)
    if {k: len(v) for k, v in splits.items()} != {"train": 10317, "val": 647, "test": 647}:
        raise ValueError("Unexpected strict split population")
    ckpt = safe_tensor_load(args.checkpoint)
    config = copy.deepcopy(ckpt["config"])
    d = config["data"]
    if sha256_file(split_path) != sha256_file(d["split_file"]):
        raise ValueError("Requested split differs from frozen checkpoint")
    gate = float(ckpt["val_contact_gate"])
    model, version = create_v11_model(ckpt, config, torch.device("cuda"))
    if version != "v11m13" or model_needs_unimol(model):
        raise ValueError("This first probe requires the audited V11m13 base")
    model.requires_grad_(False)
    checkpoint_hash = sha256_file(args.checkpoint)
    import json
    reference = json.loads((args.reference_val / "export_summary.json").read_text())
    if reference["checkpoint_sha256"] != checkpoint_hash or reference["split_sha256"] != sha256_file(split_path):
        raise ValueError("Reference export provenance mismatch")
    if reference["gate"] != gate or reference["label_overlay_sha256"] != sha256_file(args.label_alignment):
        raise ValueError("Reference gate/label overlay mismatch")
    approval = None
    if args.inference_protocol == STABLE:
        if args.precision_audit is None:
            raise ValueError("FP32 GVP export requires a passing full VAL precision audit")
        approval = load_precision_approval(
            args.precision_audit, root=ROOT, checkpoint_sha256=checkpoint_hash,
            split_sha256=sha256_file(split_path), label_overlay_sha256=sha256_file(args.label_alignment),
            reference_summary_sha256=sha256_file(args.reference_val / "export_summary.json"),
            gate=gate, expected_ids=splits['val'], acceptance_policy=args.acceptance_policy)
    elif args.precision_audit is not None:
        raise ValueError("Precision approval cannot be applied to legacy BF16 export")
    del ckpt
    args.output.mkdir(parents=True, exist_ok=False)
    rows, files, independence = [], [], None
    start = time.time()
    for subset in ("train", "val"):
        ids = splits[subset]
        overlay = load_overlay(args.label_alignment, split_path, ids)
        if args.smoke:
            ids = list(dict.fromkeys(ids[:(64 if subset == "train" else 32)] + sorted(overlay)))
        dataset = prepared_dataset(args.root, ids, d)
        if dataset.pdb_ids != ids:
            raise ValueError("Dataset silently filtered/reordered IDs")
        out = args.output / subset
        out.mkdir()
        with inference_precision(args.inference_protocol), torch.inference_mode():
            for i, sid in enumerate(ids):
                protein = safe_tensor_load(dataset._find_base_protein_file(sid))
                esm = safe_tensor_load(dataset._find_primary_esm_file(sid))
                raw = dataset.distances_dict[sid]
                distance = aligned_values(sid, protein, raw, overlay)
                keys, ca, distance = validate_axis(protein, esm, distance)
                sample = dataset[i]
                n = sample["n_residues"]
                if n != min(len(ca), len(esm["embeddings"]), d["max_seq_len"]) or not torch.equal(sample["gvp_coords"], ca[:n].float()):
                    raise ValueError(f"Feature axis mismatch: {sid}")
                for name, value in sample.items():
                    if isinstance(value, torch.Tensor) and value.is_floating_point():
                        require_finite(value, sid + "/" + name)
                batch = pocket_collate_fn_v10([sample])
                inputs = prepare_model_inputs_v11(batch, torch.device("cuda"))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(**inputs)
                for name in ("logits", "h_prot_global", "h_ligand_context"):
                    require_finite(result[name], sid + "/" + name)
                if independence is None:
                    independence = verify_independent_states(model, inputs, result, args.output / "independence.json")
                    print("PRE_CROSS_INDEPENDENCE", independence, flush=True)
                probability = np.zeros(len(ca), dtype=np.float32)
                probability[:n] = torch.softmax(result["logits"].float(), -1)[0, :n, 0].cpu().numpy()
                observed = np.arange(len(ca)) < n
                active = observed & (probability >= gate)
                if approval is not None and subset == 'val':
                    approved = approval[sid]
                    if (input_fingerprint(inputs) != approved['input_sha256']
                            or sha256_file(args.reference_val / 'predictions' / f'{sid}.npz') != approved['reference_sha256']):
                        raise ValueError(f"Input/reference changed since precision audit: {sid}")
                    with np.load(approved['path'], allow_pickle=False) as audited:
                        repeat_error = float(np.max(np.abs(probability - audited['probability'])))
                        repeat_flips = int(np.count_nonzero(active != audited['prediction']))
                        if repeat_disagreement_exceeded(repeat_error, repeat_flips, args.acceptance_policy):
                            raise ValueError(f"Stable protocol differs from its audited prediction: {sid}")
                        if repeat_error > LIMITS['max_repeat_probability_error']:
                            print(f"TOLERATED_REPEAT_DRIFT {sid} max_abs={repeat_error:.8g} flips={repeat_flips}", flush=True)
                candidates, stats = candidate_inputs(ca.numpy(), probability, active, observed)
                # Missing model tail contributes no hidden state, with coverage explicitly a statistic.
                weights = torch.zeros((len(candidates), n), device="cuda", dtype=torch.float32)
                for j, indices in enumerate(candidates):
                    visible = indices[indices < n]
                    if len(visible):
                        weights[j, torch.as_tensor(visible, device="cuda")] = 1. / len(visible)
                pocket = weights @ result["h_prot_global"][0, :n].float()
                lm = inputs["ligand_mask"][0].bool()
                if not lm.any():
                    raise ValueError(f"Empty ligand: {sid}")
                ligand = result["h_ligand_context"][0, 1:1 + len(lm)][lm].float().mean(0)
                target = distance.numpy() < d["label_contact_thresh"]
                f1, tp, sizes = candidate_targets(candidates, target)
                indices, offsets = pack_candidates(candidates)
                axis = digest_object([keys, list(protein["residue_names"])])
                protein_key = digest_object([keys, protein["residue_names"], np.round(ca.numpy(), 3).tolist()])
                pocket_np = pocket.cpu().numpy().astype(np.float16)
                ligand_np = ligand.cpu().numpy().astype(np.float16)
                if not np.isfinite(pocket_np).all() or not np.isfinite(ligand_np).all():
                    raise ValueError("FP16 cache overflow")
                path = out / f"{sid}.npz"
                tokens = {}
                if args.tokens:
                    tokens = dict(residue_tokens=result["h_prot_global"][0, :n].float().cpu().numpy().astype(np.float16),
                                  atom_tokens=result["h_ligand_context"][0, 1:1+len(lm)][lm].float().cpu().numpy().astype(np.float16),
                                  ca=ca.numpy().astype(np.float32), probability=probability,
                                  contact_target=target, observed=observed)
                    if any(not np.isfinite(v).all() for v in tokens.values()):
                        raise ValueError("Nonfinite token cache")
                np.savez_compressed(path, pocket=pocket_np, ligand=ligand_np, stats=stats,
                                    quality_f1=f1, true_positives=tp, sizes=sizes, n_true=np.array(target.sum()),
                                    candidate_indices=indices, candidate_offsets=offsets,
                                    inference_protocol=np.array(args.inference_protocol),
                                    acceptance_policy=np.array(args.acceptance_policy),
                                    sample_id=np.array(sid), subset=np.array(subset), axis_key=np.array(axis),
                                    protein_key=np.array(protein_key), n_full=np.array(len(ca)), n_observed=np.array(n), **tokens)
                ref_disagreement = None
                if subset == "val":
                    with np.load(args.reference_val / "predictions" / f"{sid}.npz", allow_pickle=False) as ref:
                        if not np.array_equal(target, ref["labels"] == 0):
                            raise ValueError(f"Reference VAL label mismatch: {sid}")
                        ref_flips = int(np.sum(active != ref["contact_prediction"]))
                        ref_disagreement = ref_flips / len(active)
                        error = float(np.max(np.abs(probability - ref["contact_probability"])))
                        if (binary_disagreement_exceeded(ref_flips, len(active), args.acceptance_policy)
                                or error > LIMITS['max_probability_error']):
                            raise ValueError(f"Reference prediction mismatch: {sid}; max_abs={error}, disagreement={ref_disagreement}")
                row = {"sample_id": sid, "subset": subset, "axis_key": axis, "protein_key": protein_key,
                       "n_candidates": len(candidates), "baseline_f1": float(f1[0]), "oracle_f1": float(f1.max()),
                       "n_full": len(ca), "n_observed": n, "n_true": int(target.sum()),
                       "changed_training_contacts": int(np.sum((sample["labels"].numpy() == 0) != target[:n])),
                       "reference_disagreement": ref_disagreement}
                rows.append(row)
                files.append({"sample_id": sid, "subset": subset, "sha256": sha256_file(path)})
                if (i + 1) % 25 == 0:
                    print(f"CACHE {subset} {i+1}/{len(ids)} elapsed={time.time()-start:.1f}s", flush=True)
        write_csv(args.output / "rows.csv", rows)
    val_rows = [r for r in rows if r["subset"] == "val"]
    if np.mean([r["reference_disagreement"] for r in val_rows]) > LIMITS['max_mean_disagreement']:
        raise ValueError("Excessive mean disagreement with audited frozen export")
    if not args.smoke and abs(np.mean([r["baseline_f1"] for r in val_rows]) - reference["full_axis_f1"]) > LIMITS['max_mean_f1_delta']:
        raise ValueError("Frozen VAL baseline reproduction failed")
    source_paths = [Path(__file__), ROOT / "scripts/diagnostics/site_quality_features.py",
                    ROOT / "scripts/diagnostics/strict_val_hybrid_candidates.py", ROOT / "scripts/diagnostics/strict_val_candidates.py",
                    ROOT / "scripts/diagnostics/site_inference_precision.py"]
    code_hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in source_paths}
    for p in sorted((ROOT / "src/models").glob("*.py")):
        code_hashes[str(p.relative_to(ROOT))] = sha256_file(p)
    write_json(args.output / "manifest.json", files)
    write_json(args.output / "summary.json", {
        "complete": True, "smoke": args.smoke, "split_sha256": sha256_file(split_path),
        "token_cache": args.tokens,
        "inference_protocol": args.inference_protocol,
        "acceptance_policy": args.acceptance_policy,
        "precision_audit_sha256": sha256_file(args.precision_audit) if args.precision_audit else None,
        "val_baseline_f1": float(np.mean([r['baseline_f1'] for r in val_rows])),
        "checkpoint_sha256": checkpoint_hash, "checkpoint": str(args.checkpoint),
        "label_overlay_sha256": sha256_file(args.label_alignment), "code_hashes": code_hashes,
        "manifest_sha256": sha256_file(args.output / "manifest.json"), "stats_names": STAT_NAMES,
        "counts": {s: sum(r["subset"] == s for r in rows) for s in ("train", "val")},
        "gate": gate, "max_seq_len": d["max_seq_len"], "max_atoms": d["max_atoms"],
        "atom_truncation_policy": d.get("atom_truncation_policy", "head"),
        "independence_checks": independence, "elapsed_seconds": time.time() - start,
        "pool": "unchanged hybrid union incl original; full protein CA, unobserved p=0",
        "warning": "Frozen backbone was trained on these TRAIN examples (not OOF); VAL only model selection, no TEST/external evaluation.",
    })
    print("SITE_QUALITY_CACHE_COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
