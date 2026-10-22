#!/usr/bin/env python3
"""Released LiBRe versus frozen PLC on the current joint TEST, reference only.

Public-checkpoint training overlap is accepted and NOT a filtering predicate.
Inputs, labels and thresholds are locked before prediction. All requested rows
are accounted for; chemistry failures are reported, never silently repaired.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.benchmark.run_libre_benchmark import windows, write_json
from scripts.diagnostics.recompute_joint_topk_reference import (
    Aggregate, METRICS, REFERENCE_SHA, score_row, sha, validate_rows,
)
from scripts.infer.benchmark_integrity import safe_tensor_load

REVISION = "e9151a85bab2d05b67b953bc55a74c780aa2bf24"
POLICY = dict(reference_only=True, test_driven_iteration_allowed=False,
              training_overlap_accepted=True, overlap_filter=False,
              mode="real", libre_threshold="FP32 p > FP32(0.6)",
              plc_threshold="existing VAL gate, never refit",
              fixed_k=[10, 20, 40, 80], length_divisors=[20, 10, 5],
              primary_equal_budget="top20", select_best_k=False,
              max_seq_len=1800, chain_window=1500, overlap=250,
              context="all observed chains, no GT binding-chain selection",
              ligand="official 40D graph from sanitized native SDF; no atom cap",
              embedding="raw FP32 ESM2 L33, official UZOB->X, BOS/EOS excluded, no PLC normalization",
              inference="FP32, batch1, no AMP/compile or parameter changes",
              failure_policy="account all rows, report matched accepted subset as partial")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_hashes(hashes):
    for path, digest in hashes.items():
        require(sha(path) == digest, f"Changed input: {path}")


def assets(root):
    manifest = root / "manifests" / (REVISION + ".json")
    data = json.loads(manifest.read_text())
    require(data["commit"] == REVISION, "Changed official revision")
    hashes = {str(manifest): sha(manifest)}
    for record in data["assets"] + data["esm2_reused"]:
        path = record.get("local_path", record.get("path"))
        hashes[path] = record["sha256"]
    check_hashes(hashes)
    return hashes


def reference(root):
    protocol_path = root / "protocol.json"
    require(sha(protocol_path) == REFERENCE_SHA, "Not the frozen joint TEST")
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads((root / "reference_test_summary.json").read_text())
    require(summary["complete"] and summary["reference_only"] and
            not summary["test_driven_iteration_allowed"], "Invalid reference policy")
    rows = torch.load(root / "predictions_real.pt", map_location="cpu", weights_only=True)
    require(validate_rows(rows, protocol, "real") ==
            summary["summaries"]["real"]["runtime"]["residue_axis_sha256"], "Changed reference axis")
    require(len(rows) == 17395 and protocol["epoch"] == 25, "Changed reference population")
    return protocol, rows


def make_segments(keys, chains, expected_axis_sha, n):
    require(len(keys) == len(set(map(tuple, keys))), "Duplicate residue keys")
    indices = defaultdict(list)
    for i, key in enumerate(keys):
        indices[str(key[0])].append(i)
    require(set(indices) == set(chains), "Chain identity mismatch")
    axis = [None] * len(keys)
    segments = []
    for chain, ix in indices.items():
        sequence = chains[chain]
        require(len(sequence) == len(ix), "Chain length mismatch")
        for i, aa in zip(ix, sequence):
            axis[i] = [*keys[i], aa]
        sequence = sequence.translate(str.maketrans({aa: "X" for aa in "UZOB"}))
        require(set(sequence) <= set("ACDEFGHIKLMNPQRSTVWYX"), "Unsupported sequence token")
        for start, stop in windows(len(sequence)):
            positions = ix[start:stop]
            if any(i < n for i in positions):
                segments.append(dict(sequence=sequence[start:stop], axis=positions))
    digest = hashlib.sha256(json.dumps(axis, separators=(",", ":")).encode()).hexdigest()
    require(digest == expected_axis_sha, "Sequence/residue identity hash mismatch")
    require(0 < n <= len(keys), "Invalid reference cap")
    return segments


def validate_target(label_data, saved_labels):
    d = torch.as_tensor(label_data["distances"])
    p = label_data["contact_label_protocol"]
    require(p["contact_dmin_angstrom"] == 5. and p["comparison"] == "<" and
            p["heavy_atoms_only"], "Not the frozen heavy-atom <5A labels")
    require(d.ndim == 1 and bool(torch.isfinite(d).all()) and bool((d >= 0).all()), "Invalid distances")
    n = min(len(d), POLICY["max_seq_len"])
    y = torch.where(d[:n] < 5, 0, torch.where(d[:n] < 8, 1, 2)).to(torch.uint8)
    require(y.tolist() == saved_labels, "Reference target differs from QC label axis")
    require(torch.equal(torch.as_tensor(label_data["contact_mask"]).bool(), d < 5), "Contact mask mismatch")
    return n


def worker_init(source):
    torch.set_num_threads(1)
    sys.path.insert(0, str(Path(source) / "src"))
    global graph_from_smiles
    from utils import smiles_to_data
    graph_from_smiles = smiles_to_data


def prepare_one(task):
    sample, saved_labels, sdf_root = task
    sid = sample["sample_id"]
    stage = "axis"
    try:
        qc = sample["qc"]
        require(qc["error"] is None, "QC pool contains a failed row")
        label_path = Path(qc["labels_path"])
        require(sha(label_path) == qc["labels_sha256"], "Changed QC labels")
        label = safe_tensor_load(label_path)
        n = validate_target(label, saved_labels)
        segments = make_segments(label["residue_keys"], qc["chains"], qc["axis_sha256"], n)
        stage = "ligand"
        ligand_path = Path(sample["ligand_path"])
        ligand = safe_tensor_load(ligand_path)
        if sample["source"] == "pdbbind":
            found = list(Path(sdf_root).glob(f"*/{sid}/{sid}_ligand.sdf"))
            require(len(found) == 1, "Missing/ambiguous staged PDBbind SDF")
            sdf = found[0]
        else:
            sdf = Path(ligand["ligand_path"])
        require(sdf.is_file(), f"Missing native SDF: {sdf}")
        from rdkit import Chem
        supplier = Chem.SDMolSupplier(str(sdf), removeHs=False, sanitize=True)
        require(len(supplier) == 1, "Expected exactly one native ligand molecule")
        mol = supplier[0]
        require(mol is not None and mol.GetNumConformers() == 1, "Cannot sanitize native ligand; no repair")
        heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        xyz = np.asarray(mol.GetConformer().GetPositions())[heavy]
        native = torch.as_tensor(ligand["coords_original"]).numpy()
        require(xyz.shape == native.shape and np.allclose(xyz, native, rtol=0, atol=.002),
                "SDF heavy-atom order/geometry differs from label ligand")
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), canonical=True, isomericSmiles=True)
        graph = graph_from_smiles(smiles)
        require(graph.x.shape == (len(heavy), 40) and bool(torch.isfinite(graph.x).all()),
                "Official graph has changed atom coverage or invalid features")
        require(graph.edge_index.ndim == 2 and graph.edge_index.shape[0] == 2,
                "Invalid official graph edges")
        if graph.edge_index.numel():
            require(int(graph.edge_index.min()) >= 0 and int(graph.edge_index.max()) < len(heavy),
                    "Official graph edge out of bounds")
        files = {str(sdf): sha(sdf), str(ligand_path): sha(ligand_path),
                 str(label_path): qc["labels_sha256"]}
        result = dict(sample_id=sid, source=sample["source"], protein_key=sample["pdb_id"],
                      n_residues=n, segments=segments, smiles=smiles,
                      graph_x=graph.x.cpu().numpy(), graph_edge_index=graph.edge_index.cpu().numpy(),
                      input_sha256=files, axis_sha256=qc["axis_sha256"])
        result["input_id"] = fingerprint(dict(files=files, axis=qc["axis_sha256"], smiles=smiles))
        return result, None
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        return None, dict(sample_id=sid, source=sample["source"], stage=stage,
                          error=f"{type(exc).__name__}: {exc}")


def prepare(args):
    torch.set_num_threads(1)
    output = args.work / "prepared"
    require(not output.exists(), "Preparation already exists; inspect before retry")
    official_hashes = assets(args.libre)
    protocol, reference_rows = reference(args.reference)
    pool = json.loads((args.pool / "pool_summary.json").read_text())
    require(sha(args.pool / "samples.jsonl") == pool["output_sha256"]["samples.jsonl"], "Changed QC pool")
    samples = {}
    wanted = {r["source"] + ":" + r["sample_id"] for r in reference_rows}
    with (args.pool / "samples.jsonl").open() as handle:
        for line in handle:
            sample = json.loads(line)
            if sample["key"] in wanted:
                require(sample["key"] not in samples, "Duplicate pool row")
                samples[sample["key"]] = sample
    require(set(samples) == wanted, "QC pool does not cover exact reference population")
    input_hashes = {str(p): sha(p) for p in [args.pool / "pool_summary.json", args.pool / "samples.jsonl",
        args.reference / "protocol.json", args.reference / "predictions_real.pt",
        args.reference / "reference_test_summary.json"]}
    code_hashes = {str(p): sha(p) for p in [Path(__file__),
        ROOT / "scripts/benchmark/run_libre_benchmark.py",
        ROOT / "scripts/diagnostics/recompute_joint_topk_reference.py",
        ROOT / "scripts/infer/benchmark_integrity.py"]}
    output.mkdir(parents=True)
    locked = dict(policy=POLICY, reference_protocol_sha256=REFERENCE_SHA,
                  val_gate=protocol["val_gate"], requested=len(reference_rows),
                  input_sha256=input_hashes, official_sha256=official_hashes, code_sha256=code_hashes)
    write_json(output / "protocol.json", locked)
    tasks = [(samples[r["source"] + ":" + r["sample_id"]], r["labels"].tolist(), str(args.sdf_root))
             for r in reference_rows]
    ready, failed = [], []
    start = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                             initializer=worker_init, initargs=(args.libre / "source" / REVISION,)) as executor:
        for i, (row, error) in enumerate(executor.map(prepare_one, tasks, chunksize=16), 1):
            if error:
                failed.append(error)
            else:
                # Avoid retaining two multiprocessing shared-memory FDs per row.
                row["graph_x"] = torch.from_numpy(row["graph_x"])
                row["graph_edge_index"] = torch.from_numpy(row["graph_edge_index"])
                ready.append(row)
            if i % 250 == 0 or i == len(tasks):
                progress = dict(processed=i, requested=len(tasks), accepted=len(ready), failed=len(failed),
                                seconds=time.monotonic()-start)
                write_json(output / "progress.json", progress)
                print("PREPARE", json.dumps(progress), flush=True)
    require(len(ready) + len(failed) == len(reference_rows) and ready, "Incomplete preparation accounting")
    torch.save(ready, output / "ready.pt")
    write_json(output / "failures.json", failed)
    check_hashes(input_hashes)
    check_hashes(code_hashes)
    write_json(output / "complete.json", dict(preparation_complete=True,
        all_inputs_accepted=not failed, requested=len(tasks), accepted=len(ready), rejected=len(failed),
        accepted_by_source=dict(Counter(r["source"] for r in ready)),
        ready_sha256=sha(output / "ready.pt"), failures_sha256=sha(output / "failures.json"),
        protocol_sha256=sha(output / "protocol.json"), job_id=os.environ.get("SLURM_JOB_ID")))


def embedding(sequence, feature_root, model, converter):
    key = hashlib.sha256(sequence.encode()).hexdigest()
    path = feature_root / (key + ".pt")
    policy = "fp32_l33_residue_only_v1"
    if path.exists():
        record = torch.load(path, map_location="cpu", weights_only=True)
        require(record["sequence"] == sequence and record["policy"] == policy, "ESM cache identity mismatch")
        value = record["embeddings"]
    else:
        _, _, tokens = converter([("chain", sequence)])
        with torch.inference_mode():
            value = model(tokens.to("cuda"), repr_layers=[33], return_contacts=False)[
                "representations"][33][0, 1:len(sequence)+1].cpu().clone()
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(dict(sequence=sequence, policy=policy, embeddings=value), temporary)
        temporary.replace(path)
    require(value.shape == (len(sequence), 1280) and value.dtype == torch.float32 and
            bool(torch.isfinite(value).all()), "Invalid raw ESM embedding")
    return value


def merge_probabilities(segments, values, n):
    total, count = np.zeros(n, np.float64), np.zeros(n, np.int64)
    require(len(segments) == len(values), "Missing segment predictions")
    for segment, probs in zip(segments, values):
        axis = np.asarray(segment["axis"])
        require(len(probs) == len(axis) and np.isfinite(probs).all(), "Invalid segment prediction")
        use = axis < n
        np.add.at(total, axis[use], probs[use])
        np.add.at(count, axis[use], 1)
    require(bool((count > 0).all()), "Uncovered reference residues")
    value = (total / count).astype(np.float32)
    require(bool(((value >= 0) & (value <= 1)).all()), "Invalid probabilities")
    return value


def compare(rows, libre_scores, gate):
    from sklearn.metrics import average_precision_score, roc_auc_score
    output = {}
    for model in ("libre", "plc"):
        aggregate = defaultdict(Aggregate)
        scores, targets = defaultdict(list), defaultdict(list)
        for row in rows:
            p = libre_scores[row["sample_id"]] if model == "libre" else row["scores"].numpy()
            y = row["labels"].numpy() == 0
            rules = score_row(p, y, gate)
            rules.pop("at_val_gate")
            selected = (np.asarray(p, dtype=np.float32) > np.float32(.6) if model == "libre"
                        else np.asarray(p, dtype=np.float64) >= gate)
            tp = int(y[selected].sum())
            rules["native_threshold"] = dict(n=len(y), positive=int(y.sum()), predicted=int(selected.sum()),
                                              tp=tp, expected_tp=float(tp), cutoff_tie=0)
            for source in ("all", row["source"]):
                scores[source].append(p)
                targets[source].append(y)
                for rule, counts in rules.items():
                    aggregate[(source, rule)].add(counts)
        output[model] = {}
        for source in scores:
            p, y = np.concatenate(scores[source]), np.concatenate(targets[source])
            output[model][source] = dict(
                rules={r: a.finish() for (s, r), a in aggregate.items() if s == source},
                auprc=float(average_precision_score(y, p)) if y.any() else None,
                auroc=float(roc_auc_score(y, p)) if y.any() and not y.all() else None)
    gaps = {s: {rule: {agg: {k: output["libre"][s]["rules"][rule][agg][k] -
        output["plc"][s]["rules"][rule][agg][k] for k in METRICS}
        for agg in ("pooled", "sample_macro")} for rule in output["libre"][s]["rules"]}
        for s in output["libre"]}
    return dict(models=output, libre_minus_plc=gaps)


def infer(args):
    import esm
    from torch_geometric.data import Data, Batch
    torch.set_num_threads(4)
    require(torch.cuda.is_available(), "Slurm GPU allocation required")
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    prepared = args.work / "prepared"
    done = json.loads((prepared / "complete.json").read_text())
    require(done["preparation_complete"] and sha(prepared / "protocol.json") == done["protocol_sha256"],
            "Incomplete/changed preflight")
    locked = json.loads((prepared / "protocol.json").read_text())
    require(locked["policy"] == POLICY, "Changed prediction policy")
    for name in ("code_sha256", "input_sha256", "official_sha256"):
        check_hashes(locked[name])
    require(sha(prepared / "ready.pt") == done["ready_sha256"] and
            sha(prepared / "failures.json") == done["failures_sha256"], "Changed prepared rows")
    ready = torch.load(prepared / "ready.pt", map_location="cpu", weights_only=True)
    _, original = reference(args.reference)
    expected = {r["sample_id"]: r for r in original}
    require(len({r["sample_id"] for r in ready}) == len(ready) == done["accepted"], "Duplicate/missing prepared rows")
    for row in ready:
        ref = expected[row["sample_id"]]
        require(row["source"] == ref["source"] and row["n_residues"] == len(ref["labels"]), "Changed ready axis")
    output = args.work / "results"
    output.mkdir(exist_ok=True)
    require(not (output / "summary.json").exists(), "Finished results already exist; do not rerun")
    cache = output / "predictions"
    cache.mkdir(exist_ok=True)
    features = args.libre / "features/chainwise_fp32_l33_v1"
    features.mkdir(exist_ok=True, parents=True)
    sys.path.insert(0, str(args.libre / "source" / REVISION / "src"))
    from models.libre import LiBRe
    model = LiBRe(use_cnn_lstm=True, use_ligand=True, residue_input_dim=1280)
    checkpoint = args.libre / "checkpoints" / REVISION / "model.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().to("cuda")
    esm_path = args.libre / "checkpoints/esm2/esm2_t33_650M_UR50D.pt"
    with torch.serialization.safe_globals([argparse.Namespace]):
        weights = torch.load(esm_path, map_location="cpu", weights_only=True)
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_core(esm_path.stem, weights,
        torch.load(esm_path.with_name(esm_path.stem + "-contact-regression.pt"), map_location="cpu", weights_only=True))
    del weights
    esm_model.eval().to("cuda")
    converter = alphabet.get_batch_converter()
    start = time.monotonic()
    all_scores = {}
    write_json(output / "protocol.json", dict(locked, prepared_sha256=done["ready_sha256"],
        job_id=os.environ.get("SLURM_JOB_ID"), checkpoint_sha256=sha(checkpoint)))
    for i, row in enumerate(ready, 1):
        sid = row["sample_id"]
        path = cache / (sid + ".pt")
        input_id = fingerprint([done["protocol_sha256"], done["ready_sha256"], row["input_id"]])
        if path.exists():
            cached = torch.load(path, map_location="cpu", weights_only=True)
            require(cached["input_id"] == input_id, "Prediction cache identity mismatch")
            scores = cached["scores"].numpy()
        else:
            graph = Batch.from_data_list([Data(x=row["graph_x"], edge_index=row["graph_edge_index"])]).to("cuda")
            values = []
            with torch.inference_mode():
                for segment in row["segments"]:
                    sequence = segment["sequence"]
                    feature = embedding(sequence, features, esm_model, converter)
                    padded = torch.zeros((1, 1500, 1280), device="cuda", dtype=torch.float32)
                    padded[0, :len(sequence)] = feature.to("cuda")
                    _, logits = model(padded, graph)
                    values.append(logits.sigmoid()[0, :len(sequence), 0].cpu().numpy())
            scores = merge_probabilities(row["segments"], values, row["n_residues"])
            temporary = path.with_suffix(".tmp")
            torch.save(dict(sample_id=sid, input_id=input_id, scores=torch.from_numpy(scores)), temporary)
            temporary.replace(path)
        require(scores.shape == (row["n_residues"],) and np.isfinite(scores).all() and
                ((scores >= 0) & (scores <= 1)).all(), "Invalid saved prediction")
        all_scores[sid] = scores
        if i % 100 == 0 or i == len(ready):
            progress = dict(completed=i, accepted=len(ready), original_requested=len(original),
                            preflight_rejected=done["rejected"], seconds=time.monotonic()-start)
            write_json(output / "progress.json", progress)
            print("INFERENCE", json.dumps(progress), flush=True)
    matched = [r for r in original if r["sample_id"] in all_scores]
    require(len(matched) == len(ready), "Prediction coverage mismatch")
    result = compare(matched, all_scores, locked["val_gate"])
    result.update(complete=not done["rejected"], traversal_complete=True, requested=len(original),
        accepted=len(ready), rejected=done["rejected"], reference_only=True,
        test_driven_iteration_allowed=False, training_overlap_accepted=True,
        si30_for_released_checkpoint_certified=False, policy=POLICY,
        prepared_protocol_sha256=done["protocol_sha256"], job_id=os.environ.get("SLURM_JOB_ID"),
        elapsed_seconds=time.monotonic()-start,
        cuda_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        cuda_peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
    write_json(output / "summary.json", result)
    print("LIBRE_JOINT_TRAVERSAL_COMPLETE", json.dumps({k: result[k] for k in
          ("complete", "requested", "accepted", "rejected", "elapsed_seconds")}), flush=True)
    if done["rejected"]:
        raise SystemExit("Partial input coverage: matched comparison saved; inspect prepared/failures.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=["prepare", "infer"])
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--libre", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--sdf-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.phase == "prepare":
        require(args.pool is not None and args.sdf_root is not None and args.workers > 0, "Missing prep inputs")
        prepare(args)
    else:
        infer(args)
