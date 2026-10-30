"""Opt-in chain/chemical-identity task; never reinterpret legacy checkpoints.

One query is a receptor chain and ONE chemical ligand, not a ligand copy.
Native copy distances are minimized before deriving all three target classes.
Unknown chemistry stays a labelled singleton, never a name-based merge.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

TASK_ID = "chain_exactchem_union_fixed5_v1"
RESIDUE_FIELDS = (
    "esm", "esm3", "protein_physchem", "protein_spatial_scalar",
    "protein_spatial_vector", "protein_rsa_raw", "protein_rsa_mask",
    "gvp_coords", "gvp_vectors", "chain_ids", "labels", "labels_original",
    "raw_distances", "esm_valid_mask", "esm3_valid_mask",
)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def normalized_keys(keys):
    out = [(str(c), int(i), str(code or "").strip()) for c, i, code in keys]
    if not out or len(set(out)) != len(out):
        raise ValueError("Empty/duplicate residue axis")
    return out


def geometry_digest(keys, coords):
    keys = normalized_keys(keys)
    xyz = torch.as_tensor(coords).detach().cpu().float().contiguous().numpy().astype("<f4")
    if xyz.shape != (len(keys), 3) or not np.isfinite(xyz).all():
        raise ValueError("Invalid chain coordinates")
    return hashlib.sha256(digest_json(keys).encode() + xyz.tobytes()).hexdigest()


def chemical_identity(sdf, expected_heavy=None):
    """No tautomer, charge, stereo or fragment normalization beyond H removal."""
    from rdkit import Chem
    path = Path(sdf)
    if not path.is_file():
        raise ValueError(f"Missing native SDF: {path}")
    mols = list(Chem.SDMolSupplier(str(path), removeHs=False, sanitize=True))
    if len(mols) != 1 or mols[0] is None:
        raise ValueError("Expected exactly one fully sanitized molecule")
    mol = Chem.RemoveHs(mols[0])
    if expected_heavy is not None and mol.GetNumHeavyAtoms() != int(expected_heavy):
        raise ValueError("SDF and cached ligand heavy-atom count disagree")
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    if not smiles:
        raise ValueError("Empty chemical identity")
    return dict(key=hashlib.sha256(smiles.encode()).hexdigest(), smiles=smiles,
                sdf=str(path), sdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), status="exact")


class UnionBuilder:
    def __init__(self, cap=1800, minimum_contacts=3):
        if cap < 1 or minimum_contacts < 1:
            raise ValueError("Positive chain cap and contact minimum required")
        self.cap, self.minimum_contacts = cap, minimum_contacts
        self.groups, self.partitions, self.seen = {}, {}, set()

    def add(self, row):
        keys = normalized_keys(row["residue_keys"])
        idx = list(map(int, row["indices"]))
        d = torch.as_tensor(row["distances"], dtype=torch.float32).clone()
        if len(idx) != len(keys) or d.shape != (len(keys),) or not torch.isfinite(d).all() or (d < 0).any():
            raise ValueError("Bad chain distance/index axis")
        if len(set(idx)) != len(idx) or min(idx) < 0 or max(idx) >= row["parent_n"]:
            raise ValueError("Invalid parent indices")
        chain = row["chain"]
        if any(key[0] != chain for key in keys):
            raise ValueError("Union input crosses receptor chains")
        member = (row["source"], row["sample_id"], chain)
        if member in self.seen:
            raise ValueError("Duplicate member would change observation weights")
        self.seen.add(member)
        chem = row["chemistry"]
        identity = chem["key"] if chem["status"] == "exact" else "singleton:" + row["sample_id"]
        # Keep source-specific feature preparations separate even for the same PDB.
        scope = (row["source"], row["pdb_id"].lower(), chain, row["geometry_sha256"], identity)
        previous = self.partitions.setdefault(scope, row["split"])
        if previous != row["split"]:
            raise ValueError("Equivalent structure/chemical query crosses frozen splits")
        key = (row["split"],) + scope
        gid = "cu_" + digest_json(key)[:32]
        if gid not in self.groups:
            self.groups[gid] = dict(sample_id=gid, split=row["split"], source=row["source"],
                pdb_id=row["pdb_id"], chain=chain, component_id=row["component_id"],
                chemistry=chem, geometry_sha256=row["geometry_sha256"],
                residue_keys=keys, distances=d, representative=row["sample_id"],
                indices=idx, parent_n=row["parent_n"], members=[row["sample_id"]])
        else:
            g = self.groups[gid]
            if keys != g["residue_keys"] or row["component_id"] != g["component_id"]:
                raise ValueError("Equivalent union keys have inconsistent axis/component")
            g["distances"] = torch.minimum(g["distances"], d)
            g["members"].append(row["sample_id"])
            # Representative selection is independent of labels and model scores.
            if row["sample_id"] < g["representative"]:
                g.update(representative=row["sample_id"], indices=idx, parent_n=row["parent_n"])

    def finish(self):
        rows = []
        for gid in sorted(self.groups):
            g = self.groups[gid]
            g["members"].sort()
            d = g["distances"]
            g.update(n_residues=min(len(d), self.cap), n_contact_full=int((d < 5).sum()),
                     n_contact=int((d[:self.cap] < 5).sum()), n_members=len(g["members"]))
            g["eligible"] = g["n_contact"] >= self.minimum_contacts
            rows.append(g)
        return dict(task_id=TASK_ID, cap=self.cap, minimum_contacts=self.minimum_contacts, rows=rows,
                    policy=dict(contact="native heavy-atom dmin <5A", near="dmin <8A",
                        target="min distance over observed same-chemistry copies on one chain",
                        copy_weight="one query once", representative="lexicographically first sample_id",
                        chemistry="canonical isomeric SMILES; unknowns retained as flagged singletons",
                        scope="split/source/PDB/receptor-chain/exact-key-and-CA-geometry",
                        context="slice cached full-receptor descriptors; rebuild model graph per chain",
                        cap_order="split chain first, cap each chain second"))


def union_sample(parent, row, cap):
    if parent["pdb_id"] != row["representative"] or parent["n_residues"] != row["parent_n"]:
        raise ValueError("Parent changed or was truncated before chain extraction")
    index = torch.tensor(row["indices"], dtype=torch.long)
    if geometry_digest(row["residue_keys"], parent["gvp_coords"][index]) != row["geometry_sha256"]:
        raise ValueError("Cached receptor geometry differs from union preparation")
    if "residue_keys" in parent:
        actual = [normalized_keys(parent["residue_keys"])[i] for i in row["indices"]]
        if actual != normalized_keys(row["residue_keys"]):
            raise ValueError("Parent residue keys changed")
    index = index[:cap]
    result = dict(parent)
    for name in RESIDUE_FIELDS:
        if name in parent:
            if parent[name].shape[0] != row["parent_n"]:
                raise ValueError(f"Invalid residue tensor: {name}")
            result[name] = parent[name].index_select(0, index)
    d = row["distances"][:cap].clone()
    labels = torch.full((len(d),), 2, dtype=torch.long)
    labels[d < 8] = 1
    labels[d < 5] = 0
    result.update(pdb_id=row["sample_id"], protein_key=row["pdb_id"],
                  ligand_key=row["chemistry"].get("key", row["representative"]),
                  chain_ids=torch.zeros(len(d), dtype=torch.long), n_residues=len(d),
                  labels=labels, raw_distances=d, residue_keys=row["residue_keys"][:cap])
    # Old per-copy auxiliary labels must never survive the task switch.
    result.pop("labels_original", None)
    return result


class ChainChemUnionDataset(torch.utils.data.Dataset):
    """View over audited normalized caches: no feature copies or renormalization."""
    _cache = {}

    def __init__(self, manifest_file, pdb_ids, split, **kwargs):
        from src.data.pocket_dataset_v8 import PocketDatasetV8
        path = str(Path(manifest_file).resolve())
        if path not in self._cache:
            self._cache[path] = torch.load(path, map_location="cpu", weights_only=False)
        manifest = self._cache[path]
        if manifest["task_id"] != TASK_ID:
            raise ValueError("Wrong chain/chemistry task version")
        self.cap = manifest["cap"]
        if kwargs.get("max_seq_len", self.cap) != self.cap:
            raise ValueError("Chain cap must match frozen label eligibility")
        lookup = {r["sample_id"]: r for r in manifest["rows"]}
        self.pdb_ids = list(pdb_ids)
        self.rows = [lookup[sid] for sid in self.pdb_ids]
        if any(r["split"] != split or not r["eligible"] for r in self.rows):
            raise ValueError("Wrong split or ineligible training/VAL union query")
        if kwargs.get("nma_n_confs", 1) != 1 or kwargs.get("nma_max_conf", 0) != 0:
            raise ValueError("Initial chain-union controls use fixed conf0 only")
        if kwargs.get("assignment_negatives_file") or kwargs.get("distances_file_original"):
            raise ValueError("Legacy per-copy auxiliary targets cannot be reused")
        parents = sorted({r["representative"] for r in self.rows})
        kwargs.update(max_seq_len=2**31 - 1, use_nma=False)
        self.base = PocketDatasetV8(pdb_ids=parents, split=split, **kwargs)
        if self.base.pdb_ids != parents or len(self.base) != len(parents):
            raise ValueError("Underlying loader dropped/reordered parent rows")
        self.parent_index = {sid: i for i, sid in enumerate(parents)}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        parent = self.base[self.parent_index[row["representative"]]]
        parent["residue_keys"] = self.base.distances_dict[row["representative"]]["residue_keys"]
        return union_sample(parent, row, self.cap)
