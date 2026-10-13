#!/usr/bin/env python3
"""Offline LiBRe smoke test on official examples, not a benchmark claim.

Preserves the upstream fixed 1500-position model input and reproduces its ESM
batch slicing. Separately measures the effect of removing retained EOS tokens.
Never alters upstream source or installs/downloads dependencies.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    manifest_path = args.root / "manifests" / (args.revision + ".json")
    manifest = json.loads(manifest_path.read_text())
    require(manifest["commit"] == args.revision, "Revision mismatch")
    for record in manifest["assets"] + manifest["esm2_reused"]:
        path = record.get("local_path", record.get("path"))
        require(sha256(path) == record["sha256"], "Asset hash mismatch: " + path)
    print("All pinned assets verified", flush=True)

    import numpy as np
    import torch
    import torch_geometric
    import esm
    import rdkit
    from sklearn.metrics import f1_score
    from torch_geometric.data import Batch

    require(torch.cuda.is_available(), "Run on an allocated GPU, not a login CPU")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = args.root / "source" / args.revision
    sys.path.insert(0, str(source / "src"))
    from models.libre import LiBRe
    import utils as upstream

    csv_path = source / "data/test/test_example.csv"
    with csv_path.open() as handle:
        rows = list(csv.DictReader(handle))[:8]
    require(len(rows) == 8, "Expected 8 official examples")
    sequences = []
    labels = []
    lengths = []
    graphs = []
    for row in rows:
        seq = re.sub(r"[UZOB]", "X", row["Sequence"])
        length = int(row["Sequence_length"])
        require(0 < length <= 1500 and length == len(seq), "Sequence length mismatch")
        label = row["label_sequence"]
        require(len(label) == length and set(label) <= {"0", "1"}, "Label mismatch")
        graph = upstream.smiles_to_data(row["Ligand_smiles"])
        require(graph.x.shape[0] > 0 and graph.x.shape[1] == 40, "Invalid ligand features")
        require(bool(torch.isfinite(graph.x).all()), "Non-finite atom features")
        sequences.append(seq)
        labels.append(label)
        lengths.append(length)
        graphs.append(graph)
    sample_csv = out / "examples.csv"
    with sample_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    checkpoint = args.root / "checkpoints" / args.revision / "model.pt"
    model = LiBRe(use_cnn_lstm=True, use_ligand=True, residue_input_dim=1280)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    print("Strict LiBRe checkpoint load passed", flush=True)

    esm_path = args.root / "checkpoints/esm2/esm2_t33_650M_UR50D.pt"
    regression_path = esm_path.with_name(esm_path.stem + "-contact-regression.pt")
    # The pre-existing, hash-verified ESM checkpoint contains legacy config objects.
    esm_data = torch.load(esm_path, map_location="cpu", weights_only=False)
    regression = torch.load(regression_path, map_location="cpu", weights_only=True)
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_core(
        esm_path.stem, esm_data, regression
    )
    del esm_data, regression
    esm_model.eval().to("cuda")
    _, _, tokens = alphabet.get_batch_converter()(list(enumerate(sequences)))
    tokens = tokens.to("cuda")
    valid = tokens != alphabet.padding_idx
    print("Extracting ESM2 layer 33 for 8 examples in FP32", flush=True)
    with torch.inference_mode():
        representations = esm_model(tokens, repr_layers=[33], return_contacts=False)["representations"][33]
        official = torch.zeros((8, 1500, 1280), dtype=torch.float32)
        clean = torch.zeros_like(official)
        retained = []
        for i, length in enumerate(lengths):
            original = representations[i][1:-1][valid[i][1:-1]].cpu()
            require(original.shape[0] in (length, length + 1), "Unexpected token count")
            official[i, :original.shape[0]] = original
            clean[i, :length] = representations[i, 1:1 + length].cpu()
            require(torch.equal(official[i, :length], clean[i, :length]), "Residue axis mismatch")
            retained.append(int(original.shape[0] - length))
    require(bool(torch.isfinite(official).all()), "Non-finite ESM embeddings")
    del esm_model, representations, tokens, valid
    gc.collect()
    torch.cuda.empty_cache()
    torch.save(official, out / "examples_protein_embedding.pt")
    with (out / "examples_ligand.pkl").open("wb") as handle:
        pickle.dump(graphs, handle)

    model.eval().to("cuda")
    batch_graph = Batch.from_data_list(graphs).to("cuda")
    with torch.inference_mode():
        binding, logits = model(official.to("cuda"), batch_graph)
        _, clean_logits = model(clean.to("cuda"), batch_graph)
        _, repeat = model(official.to("cuda"), batch_graph)
    require(tuple(logits.shape) == (8, 1500, 1), "Invalid output shape")
    require(bool(torch.isfinite(logits).all() and torch.isfinite(binding).all()), "Non-finite output")
    require(bool(torch.isfinite(clean_logits).all()), "Non-finite corrected-padding output")
    torch.testing.assert_close(logits, repeat, rtol=1e-5, atol=1e-6)
    probabilities = logits.sigmoid().squeeze(-1).cpu().numpy()
    corrected = clean_logits.sigmoid().squeeze(-1).cpu().numpy()

    upstream.DEVICE = torch.device("cuda")
    loader = upstream.create_dataloader(
        [r["padded_Sequence"] for r in rows], lengths, labels, official, graphs,
        batch_size=8, shuffle=False,
    )
    official_metrics = {k: float(v) for k, v in upstream.evaluate(loader, model).items()}
    all_labels = np.concatenate([np.asarray(list(label), dtype=np.int64) for label in labels])
    all_probs = np.concatenate([p[:n] for p, n in zip(probabilities, lengths)])
    clean_probs = np.concatenate([p[:n] for p, n in zip(corrected, lengths)])
    global_f1 = float(f1_score(all_labels, all_probs > 0.6, zero_division=0))
    require(abs(global_f1 - official_metrics["F1"]) < 1e-9, "Official F1 cross-check failed")
    samples = []
    for i, row in enumerate(rows):
        n = lengths[i]
        samples.append({"row_index": i, "pdb_id": row["PDB_ID"], "chain": row["Chain"],
                        "ligand_code": row["Ligand_code"], "length": n,
                        "n_atoms": int(graphs[i].num_nodes), "true_contacts": labels[i].count("1"),
                        "pred_contacts": int((probabilities[i, :n] > 0.6).sum()),
                        "f1": float(f1_score(list(map(int, labels[i])), probabilities[i, :n] > 0.6,
                                             zero_division=0)),
                        "retained_eos_tokens": retained[i]})
    np.savez_compressed(out / "predictions.npz", probabilities=probabilities,
                        residue_only_probabilities=corrected, lengths=np.asarray(lengths),
                        concatenated_labels=all_labels)
    (out / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    print("Official metric cross-check passed: " + json.dumps(official_metrics), flush=True)
    del model, binding, logits, clean_logits, repeat, batch_graph
    gc.collect()
    torch.cuda.empty_cache()

    # Exercise the released CLI and disk input path as well as the in-memory path.
    compat = Path(__file__).with_name("libre_cli_compat.py")
    command = [sys.executable, str(compat), str(source / "src/evaluate.py"),
               "--test_csv", str(sample_csv), "--test_emb", str(out / "examples_protein_embedding.pt"),
               "--test_ligand", str(out / "examples_ligand.pkl"), "--model_path", str(checkpoint),
               "--use_cnn_lstm", "--use_ligand", "--residue_input_dim", "1280", "--batch_size", "8"]
    with (out / "official_cli.log").open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    require(completed.returncode == 0, "Official CLI failed; inspect official_cli.log")
    cli_text = (out / "official_cli.log").read_text()
    match = re.search(r"^\s*F1:\s*([\d.]+)", cli_text, flags=re.MULTILINE)
    require(match is not None and abs(float(match.group(1)) - global_f1) < 0.000051,
            "Official CLI F1 differs from wrapper")
    report = {"complete": True, "purpose": "official-example inference smoke; not benchmark evidence",
              "revision": args.revision, "checkpoint_sha256": sha256(checkpoint),
              "input_csv_sha256": sha256(csv_path), "manifest_sha256": sha256(manifest_path),
              "runner_sha256": sha256(__file__), "sample_selection": "first 8 official test examples",
              "cli_compat_sha256": sha256(compat),
              "cli_compatibility_fix": "Explicit string dtype for label_sequence; upstream source unchanged",
              "job_id": os.environ.get("SLURM_JOB_ID"), "device": torch.cuda.get_device_name(),
              "versions": {"torch": torch.__version__, "pyg": torch_geometric.__version__,
                           "esm": esm.__version__, "rdkit": rdkit.__version__},
              "precision": "float32", "esm_layer": 33, "threshold": 0.6,
              "strict_checkpoint_load": True, "finite_outputs": True, "repeat_forward_passed": True,
              "official_cli_exit_code": completed.returncode, "official_metrics": official_metrics,
              "sample_mean_f1": float(np.mean([s["f1"] for s in samples])),
              "n_samples": len(rows), "n_residues": sum(lengths), "n_skipped": 0,
              "preprocessing_audit": {"fixed_input_length": 1500, "retained_eos_per_sample": retained,
                                      "residue_only_global_f1": float(f1_score(all_labels, clean_probs > 0.6)),
                                      "max_residue_probability_change": float(np.abs(all_probs-clean_probs).max()),
                                      "mean_residue_probability_change": float(np.abs(all_probs-clean_probs).mean())},
              "elapsed_seconds": time.time() - started}
    (out / "complete.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
