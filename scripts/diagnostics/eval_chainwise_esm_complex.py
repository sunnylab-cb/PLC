#!/usr/bin/env python3
"""Frozen strict VAL paired ESM-cache swap; BOTH passes use full complexes.

Not independent-chain inference. No gate tuning, TEST, label union or resume.
"""
import argparse
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
sys.path.insert(0, str(ROOT))
from preparation.protein.esm2_chainwise_complex import VARIANT, verify, locate, residue_keys
from scripts.diagnostics.recompute_joint_topk_reference import sha, write_json


@torch.inference_mode()
def run(args):
    verify(args.experiment)
    manifest = json.loads((args.experiment / "prepared/manifest.json").read_text())
    old = Path(manifest["training"]) / "code_v1"
    sys.path[:0] = [str(old), str(old / "scripts/train"), str(old / "scripts/diagnostics")]
    from src.data.pocket_dataset_v8 import PocketDatasetV8
    from src.data.collate import pocket_collate_fn_v10, prepare_model_inputs_v11
    from scripts.infer.infer_v11_internal_ligdep import create_v11_model
    from scripts.diagnostics.strict_assignment_common import read_strict_splits
    from scripts.diagnostics.eval_joint_val_chain_context import (
        ValAxisDataset, axis_spec, summarize_records, identity_collate,
    )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 Slurm GPU required")
    cfg = yaml.safe_load(Path(manifest["original_config"]).read_text())
    new = yaml.safe_load((args.experiment / "prepared" / f"{VARIANT}.yaml").read_text())
    cp = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if cp["config"] != cfg or args.checkpoint.name != "best_f1_c0_model.pt":
        raise ValueError("Expected frozen original G0 best-F1 checkpoint")
    gate = float(cp["val_contact_gate"])
    if not np.isfinite(gate) or not 0 < gate < 1:
        raise ValueError("Invalid frozen VAL gate")
    groups = read_strict_splits(Path(cfg["data"]["split_file"]))
    with (args.experiment / "prepared/metadata.tsv").open() as f:
        metadata = {r["sample_id"]: r for r in csv.DictReader(f, delimiter="\t")}
    allowed = inspect.signature(PocketDatasetV8).parameters
    kwargs = {k: v for k, v in cfg["data"].items() if k in allowed}
    kwargs.update(pdb_ids=groups["val"], split="val", use_nma=False, nma_n_confs=1, verbose=False)
    base = PocketDatasetV8(**kwargs)
    specs = {}
    for sid in groups["val"]:
        r = metadata[sid]
        if r["split"] != "val":
            raise ValueError("Not strict VAL")
        member = dict(source=r["source"], protein_id=r["pdb_id"], component_id=r["component_id"])
        specs[sid] = axis_spec(base.distances_dict[sid], member, cfg["data"]["max_seq_len"])
    ds = ValAxisDataset(base, specs)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False, num_workers=args.workers,
                                          collate_fn=identity_collate)
    torch.manual_seed(44)
    device = torch.device("cuda")
    model, version = create_v11_model(cp, cfg, device)
    model.eval()
    dest = args.experiment / "val_esm_swap"
    dest.mkdir(exist_ok=False)
    (dest / "predictions").mkdir()
    protocol = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=sha(args.checkpoint),
        epoch=int(cp["epoch"]) + 1, val_gate=gate, version=version,
        populations=groups["val"], context="FULL COMPLEX in BOTH modes",
        modes=["legacy_esm", "chainwise_esm"], no_threshold_tuning=True, test_forward=False,
        config_sha256=sha(manifest["original_config"]),
        feature_provenance_sha256=sha(args.experiment / "prepared/provenance.json"))
    write_json(dest / "protocol.json", protocol)
    del cp

    def predict(samples):
        inputs = prepare_model_inputs_v11(pocket_collate_fn_v10(samples), device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**inputs)["logits"]
        if not torch.isfinite(logits[inputs["protein_mask"].bool()]).all():
            raise ValueError("Nonfinite valid logits")
        prob = logits.float().softmax(-1).cpu().numpy()
        return [prob[i, :s["n_residues"]].copy() for i, s in enumerate(samples)]

    records, hashes = [], {}
    started = time.monotonic()
    for samples in loader:
        legacy = predict(samples)
        replaced = []
        for sample in samples:
            sid = sample["pdb_id"]
            path = locate(Path(new["data"]["esm_features_dir"]), sid)
            d = torch.load(path, map_location="cpu", weights_only=False)
            n = sample["n_residues"]
            expected = [(str(c), int(i), str(code)) for c, i, code in sample["residue_keys"]]
            if residue_keys(d)[:n] != expected or len(d["embeddings"]) != specs[sid]["raw_n_residues"]:
                raise ValueError(f"New ESM axis mismatch: {sid}")
            candidate = dict(sample)
            candidate["esm"] = d["embeddings"][:n].float()
            if sid not in manifest["changed"] and not torch.equal(candidate["esm"], sample["esm"].float()):
                raise ValueError("Single-chain/BioLiP control embeddings changed")
            replaced.append(candidate)
        chainwise = predict(replaced)
        for sample, oldp, newp in zip(samples, legacy, chainwise):
            sid = sample["pdb_id"]
            y = sample["labels"].numpy()
            path = dest / "predictions" / f"{sid}.npz"
            np.savez_compressed(path, legacy_probs=oldp, chainwise_probs=newp, labels=y,
                                residue_keys=np.asarray(sample["residue_keys"], dtype=str))
            hashes[sid] = sha(path)
            # Reuse the tested paired metric aggregator; rename internal keys below.
            records.append(dict(spec=specs[sid], labels=y, keys=np.asarray(sample["residue_keys"]),
                                full=oldp[:, 0], chain=newp[:, 0]))
        if len(records) % 100 == 0:
            print("FULL_COMPLEX_ESM_SWAP", len(records), len(ds), flush=True)
            write_json(dest / "progress.json", dict(done=len(records), total=len(ds), seconds=time.monotonic()-started))
    if list(hashes) != groups["val"]:
        raise ValueError("Incomplete VAL inference")
    summary = summarize_records(records, gate)
    summary["results"] = {group: {"legacy_esm": values["full"], "chainwise_esm": values["chain"]}
                          for group, values in summary["results"].items()}
    summary["chainwise_minus_legacy"] = summary.pop("chain_minus_full")
    summary.update(complete=True, context="FULL COMPLEX in BOTH modes", val_gate=gate,
                   seconds=time.monotonic()-started, no_test=True,
                   warning="Frozen-checkpoint distribution-shift diagnostic, not scratch-training performance")
    write_json(dest / "prediction_audit.json", hashes)
    write_json(dest / "summary.json", summary)
    print("FULL_COMPLEX_ESM_SWAP_COMPLETE", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--workers", type=int, default=8)
    torch.set_num_threads(4)
    run(p.parse_args())
