"""Strict, parameter-free input checks for benchmark re-evaluation."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def safe_tensor_load(path, map_location="cpu"):
    # Historical feature caches contain NumPy arrays/scalars as well as tensors.
    allowed = [np.ndarray, np.dtype, np.core.multiarray._reconstruct,
               np.core.multiarray.scalar]
    allowed += [(np.core.multiarray._reconstruct, f"{prefix}.multiarray._reconstruct")
                for prefix in ("numpy.core", "numpy._core")]
    allowed += [(np.core.multiarray.scalar, f"{prefix}.multiarray.scalar")
                for prefix in ("numpy.core", "numpy._core")]
    allowed += [type(np.dtype(s)) for s in
                ("float16", "float32", "float64", "int64", "int32", "int16",
                 "int8", "uint8", "uint32", "uint64", "bool", "U1", "S1", "O")]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location=map_location, weights_only=True)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def residue_keys(data, expected_length=None, allow_duplicates=False):
    chains = data.get("chain_ids", data.get("chain_id"))
    numbers = data.get("residue_indices", data.get("residue_index"))
    if chains is None or numbers is None:
        raise ValueError("Missing chain/residue key metadata")
    insertions = data.get("insertion_codes", data.get("insertion_code"))
    if insertions is None:
        insertions = [""] * len(numbers)
    if not len(chains) == len(numbers) == len(insertions):
        raise ValueError("Residue metadata lengths disagree")
    keys = [(str(c), int(n), str(i).strip()) for c, n, i in zip(chains, numbers, insertions)]
    if expected_length is not None and len(keys) != expected_length:
        raise ValueError(f"Residue key count {len(keys)} != tensor length {expected_length}")
    if not allow_duplicates and len(keys) != len(set(keys)):
        duplicates = [key for key, count in Counter(keys).items() if count > 1]
        raise ValueError(f"Duplicate residue keys: {duplicates[:5]}")
    return keys


def require_finite(tensor, name):
    tensor = torch.as_tensor(tensor)
    bad = ~torch.isfinite(tensor)
    if bad.any():
        raise ValueError(f"{name}: {int(bad.sum())} non-finite values; first={bad.nonzero()[:5].tolist()}")


def align_embedding(source, reference, name):
    values = source["embeddings"].float()
    require_finite(values, name)
    if all(k in source for k in ("atom37_positions", "atom37_mask", "residue_name_1")) and "ca_coords" in reference:
        return align_embedding_by_ca(source, reference, name)
    try:
        skeys = residue_keys(source, values.shape[0])
    except ValueError as exc:
        raise ValueError(f"{name}: {exc}") from exc
    rkeys = residue_keys(reference)
    mapping = {key: i for i, key in enumerate(skeys)}
    valid = torch.tensor([key in mapping for key in rkeys], dtype=torch.bool)
    if not valid.any():
        raise ValueError(f"{name}: no shared residue keys")
    out = values.new_zeros((len(rkeys), *values.shape[1:]))
    indices = torch.tensor([mapping[key] for key in rkeys if key in mapping])
    out[valid] = values[indices]
    src_names = source.get("residue_name_3", source.get("residue_names"))
    ref_names = reference.get("residue_names", reference.get("residue_name_3"))
    if src_names is not None and ref_names is not None:
        if len(src_names) != len(skeys) or len(ref_names) != len(rkeys):
            raise ValueError(f"{name}: residue name metadata lengths disagree")
        canonical = set("ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split())
        for i, key in enumerate(rkeys):
            if key not in mapping:
                continue
            a, b = str(ref_names[i]).upper(), str(src_names[mapping[key]]).upper()
            if a in canonical and b in canonical and a != b:
                raise ValueError(f"{name}: residue identity mismatch at {key}: {a} != {b}")
    return out, valid, {
        "raw_length": len(skeys), "matched": int(valid.sum()),
        "missing": int((~valid).sum()),
        "positional_mismatches": sum(a != b for a, b in zip(skeys, rkeys)),
    }


def align_embedding_by_ca(source, reference, name, tolerance=0.01):
    """Resolve auxiliary-cache numbering defects by same-chain CA identity.

    This is an input identity check, not protein-ligand geometry or nearest-site
    matching. Only unique CA matches within PDB rounding tolerance are accepted.
    """
    from scipy.spatial import cKDTree

    values = source["embeddings"].float()
    skeys = residue_keys(source, len(values), allow_duplicates=True)
    rkeys = residue_keys(reference)
    positions = torch.as_tensor(source["atom37_positions"])
    masks = torch.as_tensor(source["atom37_mask"])
    if positions.shape != (len(values), 37, 3) or masks.shape != (len(values), 37):
        raise ValueError(f"{name}: malformed atom37 metadata")
    ca = positions[:, 1].cpu().numpy()
    has_ca = masks[:, 1].bool().cpu().numpy()
    rca = torch.as_tensor(reference["ca_coords"]).cpu().numpy()
    if rca.shape != (len(rkeys), 3) or len(source["residue_name_1"]) != len(skeys):
        raise ValueError(f"{name}: CA/sequence metadata length mismatch")
    require_finite(ca[has_ca], f"{name} source CA")
    require_finite(rca, f"{name} reference CA")
    trees = {}
    for chain in {k[0] for k in rkeys}:
        indices = np.asarray([i for i, k in enumerate(skeys) if k[0] == chain and has_ca[i]], dtype=int)
        if len(indices):
            trees[chain] = (indices, cKDTree(ca[indices]))
    names = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
    aa1 = dict(zip(names, "ACDEFGHIKLMNPQRSTVWY"))
    reference_names = reference.get("residue_names", reference.get("residue_name_3"))
    if reference_names is None or len(reference_names) != len(rkeys):
        raise ValueError(f"{name}: reference identities required for coordinate alignment")
    out = values.new_zeros((len(rkeys), *values.shape[1:]))
    valid = torch.zeros(len(rkeys), dtype=torch.bool)
    rekeyed = duplicate_resolved = name3_conflicts = 0
    source_counts = Counter(skeys)
    matched_indices, errors = [], []
    for i, key in enumerate(rkeys):
        if key[0] not in trees:
            continue
        indices, tree = trees[key[0]]
        candidates = tree.query_ball_point(rca[i], tolerance)
        if not candidates:
            continue
        if len(candidates) != 1:
            raise ValueError(f"{name}: ambiguous CA identity at {key}: {len(candidates)} matches")
        j = int(indices[candidates[0]])
        ref_name = str(reference_names[i]).upper()
        src_letter = str(source["residue_name_1"][j]).upper()
        if ref_name in aa1 and src_letter != aa1[ref_name]:
            raise ValueError(f"{name}: CA matches but sequence identity disagrees at {key}: "
                             f"{src_letter} != {aa1[ref_name]}")
        out[i], valid[i] = values[j], True
        matched_indices.append(j)
        errors.append(float(np.linalg.norm(ca[j] - rca[i])))
        rekeyed += int(skeys[j] != key)
        duplicate_resolved += int(source_counts[skeys[j]] > 1)
        src_names = source.get("residue_name_3")
        if src_names is not None:
            name3_conflicts += int(str(src_names[j]).upper() != ref_name)
    if len(matched_indices) != len(set(matched_indices)):
        raise ValueError(f"{name}: source CA assigned to multiple reference residues")
    if not valid.any():
        raise ValueError(f"{name}: no shared residue CA identities")
    return out, valid, {
        "raw_length": len(skeys), "matched": int(valid.sum()), "missing": int((~valid).sum()),
        "positional_mismatches": sum(a != b for a, b in zip(skeys, rkeys)),
        "alignment_evidence": "same_chain_unique_CA_and_sequence_identity",
        "ca_tolerance_angstrom": tolerance, "max_ca_error": max(errors),
        "coordinate_rekeyed": rekeyed, "duplicate_key_matches_resolved": duplicate_resolved,
        "source_duplicate_rows": len(skeys) - len(set(skeys)),
        "source_name3_conflicts": name3_conflicts,
    }


def align_distances(distances, tag_metadata, reference):
    require_finite(distances, "labels/distances")
    tkeys = residue_keys(tag_metadata, len(distances))
    pkeys = residue_keys(reference)
    index = {key: i for i, key in enumerate(tkeys)}
    missing = [key for key in pkeys if key not in index]
    if missing:
        raise ValueError(f"Protein residues missing from tag axis: {missing[:5]}")
    return distances[torch.tensor([index[key] for key in pkeys])]


def legacy_tag_axis(path):
    """Reproduce benchmark/flat tag enumeration, not the protein feature parser."""
    from Bio.PDB import PDBParser

    structure = PDBParser(QUIET=True).get_structure("tag_axis", str(path))
    rows = []
    for chain in structure[0]:
        for residue in chain:
            if residue.id[0] != " ":
                continue
            coords = np.asarray([atom.get_coord() for atom in residue.get_atoms()
                                 if atom.element != "H"])
            if len(coords):
                rows.append((chain.id, int(residue.id[1]), str(residue.id[2]).strip(),
                             residue.resname, coords))
    # Historical sorting omitted insertion codes; stable sorting preserves ties.
    rows.sort(key=lambda row: (row[0], row[1]))
    if not rows:
        raise ValueError(f"No tag residues in {path}")
    return {
        "chain_ids": [r[0] for r in rows], "residue_indices": [r[1] for r in rows],
        "insertion_codes": [r[2] for r in rows], "residue_names": [r[3] for r in rows],
        "heavy_coords": [r[4] for r in rows],
        "source_pdb": str(path), "source_sha256": sha256_file(path),
    }


def verify_tag_distances(distances, metadata, sdf_paths, tolerance=1e-4):
    """Verify recovered tag order against actual distances, using labels only for QC."""
    from rdkit import Chem
    from scipy.spatial import cKDTree

    residue_keys(metadata, len(distances))
    ligand_coords, sources = [], []
    for path in sdf_paths:
        supplier = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False,
                                     strictParsing=False)
        mol = next((candidate for candidate in supplier if candidate is not None), None)
        if mol is None or not mol.GetNumConformers():
            raise ValueError(f"No ligand conformer in {path}")
        positions = mol.GetConformer().GetPositions()
        ligand_coords.extend(positions[atom.GetIdx()] for atom in mol.GetAtoms()
                             if atom.GetAtomicNum() != 1)
        sources.append({"path": str(path), "sha256": sha256_file(path)})
    if not ligand_coords:
        raise ValueError("No heavy ligand coordinates for tag verification")
    coords = np.concatenate(metadata["heavy_coords"])
    require_finite(coords, "tag source protein coordinates")
    require_finite(np.asarray(ligand_coords), "tag source ligand coordinates")
    atom_distances = cKDTree(np.asarray(ligand_coords)).query(coords, workers=1)[0]
    starts = np.cumsum([0] + [len(x) for x in metadata["heavy_coords"][:-1]])
    expected = np.minimum.reduceat(atom_distances, starts)
    actual = distances.detach().cpu().numpy()
    require_finite(actual, "cached tag distances")
    error = float(np.max(np.abs(expected - actual)))
    if error > tolerance:
        raise ValueError(f"Tag/source geometry mismatch: max error={error:.6g}, "
                         f"tolerance={tolerance}, protein={metadata['source_pdb']}")
    return {"distance_verified": True, "max_distance_error": error,
            "distance_tolerance": tolerance, "ligand_sources": sources}


def checked_ligand_features(features, policy, name):
    values = features.float()
    bad = ~torch.isfinite(values)
    count = int(bad.sum())
    if not count:
        return values, []
    positions = bad.nonzero().tolist()
    if policy != "sasa_zero" or any(column not in (38, 39) for _, column in positions):
        require_finite(values, name)
    # Match the extractor's documented fallback, before training-stat normalization.
    out = values.clone()
    out[bad] = 0.0
    return out, positions
