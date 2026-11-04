#!/usr/bin/env python3
"""Frozen strict-VAL flag interventions and chain-order reversal, never TEST.

All multichain VAL rows plus the first50 single-chain rows of each source.
ESM is already chainwise in every pass; the model still sees full complexes.
Keep ligand, labels, coordinates and prediction budget unchanged. Reversing
chain order permutes every residue-aligned model input, then unpermutes scores.
"""
import argparse
import copy
import csv
import inspect
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/diagnostics")]
from prepare_chain_boundary_controls import FLAGS, VARIANTS, sha, verify

RESIDUE_INPUTS = (
    "esm_embeddings", "protein_physchem", "protein_spatial_scalar", "protein_spatial_vector",
    "protein_gvp_vectors", "protein_coords", "protein_mask", "chain_ids",
)


def reverse_chains(inputs):
    if inputs["protein_mask"].shape[0] != 1 or not inputs["protein_mask"].all():
        raise ValueError("Permutation audit requires one unpadded complex")
    chains = inputs["chain_ids"][0]
    order = list(dict.fromkeys(chains.tolist()))
    index = torch.cat([torch.where(chains == c)[0] for c in reversed(order)])
    result = dict(inputs)
    for key in RESIDUE_INPUTS:
        result[key] = inputs[key].index_select(1, index)
    return result, index.argsort()


def permutation_stats(reference, other):
    k = min(20, len(reference))
    a, b = (set(np.argsort(-p, kind="stable")[:k].tolist()) for p in (reference, other))
    return dict(mean_abs_probability_change=float(np.abs(reference-other).mean()),
                max_abs_probability_change=float(np.abs(reference-other).max()),
                top20_jaccard=len(a & b) / len(a | b))


@torch.inference_mode()
def run(args):
    provenance = verify(args.experiment)
    model_root = args.experiment / "model_code"
    sys.path[:0] = [str(model_root), str(model_root / "scripts/train"), str(model_root / "scripts/diagnostics")]
    from src.data.pocket_dataset_v8 import PocketDatasetV8
    from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11
    from scripts.infer.infer_v11_internal_ligdep import create_v11_model
    from scripts.diagnostics.strict_assignment_common import read_strict_splits
    from scripts.diagnostics.eval_joint_val_chain_context import ValAxisDataset, axis_spec, summarize_records, identity_collate
    import src.models.gvp_layers as graph
    assert Path(graph.__file__).resolve().is_relative_to(model_root.resolve())
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dest = args.experiment / "val_boundary_audit"
    dest.mkdir(exist_ok=False)
    (dest / "predictions").mkdir()
    upstream = Path(provenance["upstream"])
    base = yaml.safe_load((upstream / "prepared/G0_chainwise_esm_complex.yaml").read_text())
    groups = read_strict_splits(Path(base["data"]["split_file"]))
    with open(base["scan_metadata_file"]) as f:
        metadata = {r["sample_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    singles = {s: [i for i in groups["val"] if metadata[i]["source"] == s and int(metadata[i]["n_chains"]) == 1][:50]
               for s in ("pdbbind", "biolip")}
    selected = [i for i in groups["val"] if int(metadata[i]["n_chains"]) > 1 or any(i in x for x in singles.values())]
    assert all(metadata[i]["split"] == "val" for i in selected)
    allowed = inspect.signature(PocketDatasetV8).parameters
    kwargs = {k: v for k, v in base["data"].items() if k in allowed}
    kwargs.update(pdb_ids=selected, split="val", use_nma=False, nma_n_confs=1, verbose=False)
    ds = PocketDatasetV8(**kwargs)
    specs = {i: axis_spec(ds.distances_dict[i], dict(source=metadata[i]["source"],
        protein_id=metadata[i]["pdb_id"], component_id=metadata[i]["component_id"]), base["data"]["max_seq_len"]) for i in selected}
    loader = torch.utils.data.DataLoader(ValAxisDataset(ds, specs), batch_size=1,
        shuffle=False, num_workers=args.workers, collate_fn=identity_collate)
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    expected = json.loads((upstream / "val_esm_swap/protocol.json").read_text())
    assert sha(args.checkpoint) == expected["checkpoint_sha256"]
    gate = float(cp["val_contact_gate"])
    cfg = copy.deepcopy(cp["config"])
    cfg["model"].update(dict(zip(FLAGS, (False, False))))
    device = torch.device("cuda")
    torch.manual_seed(44)
    model, _ = create_v11_model(cp, cfg, device)
    model.eval()
    del cp
    variants = {"baseline": (False, False), **VARIANTS}
    protocol = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        epoch=expected["epoch"], gate=gate, samples=selected, variants=variants,
        precision="bf16", batch_size=1, esm="chainwise in every mode", full_complex=True,
        no_test=True, no_threshold_tuning=True, normalization="unchanged TRAIN statistics",
        source_sha256=provenance["model_hashes"], note="Frozen interventions, not retrained performance")
    (dest / "protocol.json").write_text(json.dumps(protocol, indent=2))
    records = {n: [] for n in variants}
    rows, started = [], time.monotonic()

    def predict(inputs):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs)["logits"].float()
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite VAL logits")
        return logits.softmax(-1)[0, :, 0]

    for samples in loader:
        sample = samples[0]
        sid = sample["pdb_id"]
        inputs = prepare_model_inputs_v11(pocket_collate_fn_v10(samples), device)
        changed, inverse = reverse_chains(inputs)
        boundary = (inputs["chain_ids"][0, 1:] != inputs["chain_ids"][0, :-1])
        distances = (inputs["protein_coords"][0, 1:] - inputs["protein_coords"][0, :-1]).norm(dim=-1)[boundary].cpu().numpy()
        row = dict(sample_id=sid, source=specs[sid]["source"], protein_id=specs[sid]["protein_id"],
                   n_chains=specs[sid]["n_chains"], boundary_distances_A=distances.tolist(),
                   false_directed_backbone_bridges=int(2*len(distances)),
                   bridges_outside_radius=int(2*(distances > cfg["model"]["gvp_radius"]).sum()), modes={})
        stored = dict(labels=sample["labels"].numpy(), residue_keys=np.asarray(sample["residue_keys"], dtype=str))
        for name, flags in variants.items():
            model.config.update(dict(zip(FLAGS, flags)))
            pred = predict(inputs).cpu().numpy()
            perm = predict(changed)[inverse].cpu().numpy()
            stored[name] = pred
            stored[name + "_permuted"] = perm
            row["modes"][name] = permutation_stats(pred, perm)
            records[name].append(dict(spec=specs[sid], labels=stored["labels"], keys=stored["residue_keys"],
                full=stored["baseline"], chain=pred))
        np.savez_compressed(dest / "predictions" / f"{sid}.npz", **stored)
        rows.append(row)
        if len(rows) % 25 == 0:
            progress = dict(done=len(rows), total=len(selected), seconds=time.monotonic()-started)
            (dest / "progress.json").write_text(json.dumps(progress))
            print("CHAIN_BOUNDARY_VAL", progress, flush=True)
    assert [r["sample_id"] for r in rows] == selected
    results = {}
    for name in variants:
        summary = summarize_records(records[name], gate)
        results[name] = {g: v["chain"] for g, v in summary["results"].items()}
    output = dict(complete=True, no_test=True, n=len(rows), seconds=time.monotonic()-started,
        protocol_sha256=sha(dest / "protocol.json"), results=results,
        bridge_count=sum(r["false_directed_backbone_bridges"] for r in rows),
        beyond_radius_bridge_count=sum(r["bridges_outside_radius"] for r in rows),
        chain_reversal={name: {group: {k: float(np.mean([r["modes"][name][k] for r in rows
            if (r["n_chains"] > 1) == multi])) for k in ("mean_abs_probability_change", "top20_jaccard")}
            for group, multi in (("single_chain", False), ("multichain", True))} for name in variants})
    (dest / "rows.json").write_text(json.dumps(rows, indent=2))
    (dest / "summary.json").write_text(json.dumps(output, indent=2))
    print("CHAIN_BOUNDARY_VAL_COMPLETE", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    torch.set_num_threads(4)
    run(p.parse_args())
