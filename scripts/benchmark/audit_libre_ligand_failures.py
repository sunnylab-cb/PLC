#!/usr/bin/env python3
"""Audit rejected SDF chemistry against RCSB CCD without changing any inputs.

Downloads only small chemical-component JSON records on the machine running this
script. A candidate requires RDKit sanitization and an isomorphic heavy-atom
connectivity graph. Unsupported metal valence is recorded, never bypassed.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import urllib.request

import networkx as nx
from rdkit import Chem


def heavy_graph(mol):
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() != 1:
            graph.add_node(atom.GetIdx(), element=atom.GetAtomicNum())
    for bond in mol.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if a in graph and b in graph:
            graph.add_edge(a, b)
    return graph


def same_connectivity(a, b):
    return nx.is_isomorphic(heavy_graph(a), heavy_graph(b),
                           node_match=lambda x, y: x["element"] == y["element"])


def element_counts(mol):
    return Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() != 1)


def compare_candidate(observed, smiles):
    candidate = Chem.MolFromSmiles(smiles)
    report = {"sanitizes": candidate is not None, "same_heavy_connectivity": False}
    if candidate is not None:
        counts, original = element_counts(candidate), element_counts(observed)
        report.update(
            heavy_elements=dict(counts),
            missing_from_observed=dict(counts - original),
            extra_in_observed=dict(original - counts),
            same_heavy_composition=counts == original,
            same_heavy_connectivity=same_connectivity(observed, candidate),
        )
    return report


def get_record(code, cache):
    path = cache / (code + ".json")
    url = "https://data.rcsb.org/rest/v1/core/chemcomp/" + code
    if not path.exists():
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "PLC-LiBRe-input-QC/1.0"})
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = response.read(4 * 1024 * 1024)
                record = json.loads(data)
                if record["chem_comp"]["id"] != code:
                    raise ValueError("CCD identity mismatch")
                path.write_bytes(data)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2 * (attempt + 1))
    record = json.loads(path.read_text())
    if record["chem_comp"]["id"] != code:
        raise ValueError("Cached CCD identity mismatch")
    return record, url, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-ccd", action="append", default=[], metavar="SAMPLE=CODE",
                        help="Research-only alternate identity to inspect; never replaces benchmark input")
    args = parser.parse_args()
    candidates = {}
    for value in args.candidate_ccd:
        sample, separator, code = value.partition("=")
        if not separator or not sample or not code.isalnum() or sample in candidates:
            parser.error("--candidate-ccd expects a unique SAMPLE=CODE with an alphanumeric code")
        candidates[sample] = code.upper()
    cache = args.root / "chemistry_qc/ccd"
    cache.mkdir(parents=True, exist_ok=True)
    audits = []
    for dataset in ["coach420", "holo4k"]:
        path = args.results_root / dataset / "failures.json"
        if not path.exists():
            continue
        for failure in json.loads(path.read_text()):
            row = {"dataset": dataset, **failure, "candidates": []}
            sample = failure["sample_key"]
            try:
                declared_code = sample.split("_")[1]
                code = candidates.get(sample, declared_code)
                row["declared_code"] = declared_code
                row["candidate_identity_only"] = code != declared_code
                if not code.isalnum():
                    raise ValueError("Invalid CCD code")
                sdf = args.benchmark_root / "ligands_sdf" / dataset / (sample + ".sdf")
                mol = next((m for m in Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)
                            if m is not None), None)
                if mol is None:
                    raise ValueError("Cannot read original SDF even without sanitization")
                row["sdf_sha256"] = hashlib.sha256(sdf.read_bytes()).hexdigest()
                row["heavy_elements"] = dict(element_counts(mol))
                row["sdf_properties"] = {key: mol.GetProp(key) for key in mol.GetPropNames()}
                record, url, digest = get_record(code, cache)
                row.update(ccd_code=code, ccd_url=url, ccd_sha256=digest,
                           ccd_name=record["chem_comp"].get("name"))
                seen = set()
                for descriptor in record.get("pdbx_chem_comp_descriptor", []):
                    if "SMILES" not in descriptor.get("type", ""):
                        continue
                    smiles = descriptor["descriptor"]
                    if smiles in seen:
                        continue
                    seen.add(smiles)
                    entry = {"smiles": smiles, "type": descriptor["type"],
                             "program": descriptor.get("program"), **compare_candidate(mol, smiles)}
                    row["candidates"].append(entry)
                row["has_valid_same_connectivity_candidate"] = any(c["same_heavy_connectivity"] for c in row["candidates"])
            except Exception as exc:
                row["audit_error"] = str(exc)
            audits.append(row)
            print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audits, indent=2) + "\n")


if __name__ == "__main__":
    main()
