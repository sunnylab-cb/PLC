"""Fail-closed, split-independent feature/geometry QC for the two new cohorts."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

from internal_contact_geometry import count_contacts, ligand_elements, protein_elements


@lru_cache(maxsize=3)
def load_tensor(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def residue_axis(data):
    chains = data.get("chain_ids", data.get("chain_id"))
    indices = data.get("residue_indices", data.get("residue_index"))
    codes = data.get("insertion_codes", data.get("insertion_code"))
    if chains is None or indices is None or codes is None or not len(chains):
        raise ValueError("missing residue axis")
    if not len(chains) == len(indices) == len(codes):
        raise ValueError("residue-axis length mismatch")
    keys = [(str(c), int(i), str(code or "").strip()) for c, i, code in zip(chains, indices, codes)]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate residue keys")
    return keys


def finite_shape(value, shape, name):
    x = torch.as_tensor(value)
    if tuple(x.shape) != tuple(shape) or not bool(torch.isfinite(x).all()):
        raise ValueError(f"shape/nonfinite {name}: {tuple(x.shape)} != {tuple(shape)}")
    return x


def ligand_check(ligand, sample):
    x = torch.as_tensor(ligand["atom_features"])
    n = len(x)
    finite_shape(x, (n, 109), "ligand features")
    finite_shape(ligand["coords"], (n, 3), "ligand coords")
    if n < 1 or ligand.get("feature_normalization"):
        raise ValueError("ligand must be non-empty and raw, not already normalized")
    edges = torch.as_tensor(ligand["edge_index"])
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.is_floating_point():
        raise ValueError("invalid edge index shape/dtype")
    if edges.numel() and (edges.min() < 0 or edges.max() >= n):
        raise ValueError("edge index out of bounds")
    finite_shape(ligand["edge_attr"], (edges.shape[1], 6), "edge attributes")
    heavy = int(ligand["num_heavy_atoms"])
    if heavy < 6:
        raise ValueError("ligand has fewer than six heavy atoms")
    # Geometry labels must use native bound coordinates, never optimized conformers.
    coords = finite_shape(ligand["coords_original"], (heavy, 3), "native ligand coords")
    if sample["source"] == "biolip":
        row = sample["ligand_metadata"]
        if not 6 <= heavy <= 128 or heavy != int(row["heavy_atoms_meta"]):
            raise ValueError("BioLiP heavy-atom metadata/range mismatch")
        if int(ligand.get("ligand_route", -1)) != 5:
            raise ValueError("BioLiP ligand route is not regular organic (5)")
        for field in ("pdb_id", "protein_chain", "binding_site_id", "ligand_id"):
            if str(ligand.get(field, "")) != row[field]:
                raise ValueError(f"BioLiP ligand identity mismatch: {field}")
        mask = finite_shape(ligand["x_group_mask"], (n, 14), "ligand group validity")
        if not bool(((mask == 0) | (mask == 1)).all()):
            raise ValueError("nonbinary ligand group validity")
        completeness = float(mask.float().mean())
        if completeness < .8 or abs(completeness-float(ligand["feature_completeness_ratio"])) >= 1e-5:
            raise ValueError("invalid/insufficient ligand feature completeness")
    elif not bool(ligand.get("has_original_coords", False)):
        raise ValueError("PDBbind ligand lacks native-coordinate provenance")
    return coords.numpy(), heavy


@lru_cache(maxsize=2)
def cif_atoms(path, chain):
    import gemmi
    structure = gemmi.read_structure(path)
    result = {}
    for c in structure[0]:
        if c.name.strip() != chain:
            continue
        for residue in c:
            heavy = [a for a in residue if a.element.name not in ("H", "D")]
            alt = lambda a: a.altloc.strip("\x00 .?")
            alt_ids = sorted({alt(a) for a in heavy if alt(a)})
            chosen = max(alt_ids, key=lambda k: sum(a.occ for a in heavy if alt(a) == k)) if alt_ids else ""
            heavy = [a for a in heavy if not alt(a) or alt(a) == chosen]
            names = [a.name.strip() for a in heavy]
            if len(names) != len(set(names)):
                raise ValueError("Duplicate CIF atom after alternate-conformer selection")
            cas = [a for a in heavy if a.name.strip() == "CA" and a.element.name == "C"]
            if not cas:
                continue
            key = (chain, int(residue.seqid.num), residue.seqid.icode.strip())
            if key in result:
                raise ValueError(f"ambiguous CIF residue: {key}")
            result[key] = (residue.name, np.asarray(cas[0].pos.tolist()),
                           np.asarray([a.pos.tolist() for a in heavy]),
                           [a.element.name for a in heavy], names)
    return result


def heavy_distances(residue_coords, ligand_coords):
    arrays = [np.asarray(x, dtype=np.float64) for x in residue_coords]
    if any(x.ndim != 2 or x.shape[1] != 3 or not len(x) or not np.isfinite(x).all() for x in arrays):
        raise ValueError("missing/nonfinite residue heavy atoms")
    if not arrays:
        raise ValueError("empty protein")
    per_atom = cKDTree(ligand_coords).query(np.concatenate(arrays), workers=1)[0]
    starts = np.cumsum([0] + [len(a) for a in arrays[:-1]])
    return torch.tensor(np.minimum.reduceat(per_atom, starts), dtype=torch.float32)


def qc_sample(task):
    from prepare_internal_codataset_sequences import AA3, chain_sequences
    torch.set_num_threads(1)
    key, sample, output = task
    paths = [Path(sample[k]) for k in ("v4_path", "esm_path", "ligand_path")]
    provenance = {str(p): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in paths}
    record = dict(key=key, sources=provenance, error=None)
    try:
        protein, esm, ligand = (load_tensor(str(p)) for p in paths)
        pk, ek = residue_axis(protein), residue_axis(esm)
        positions = [i for i, k in enumerate(pk) if sample["selected_chain"] is None
                     or k[0] == sample["selected_chain"]]
        selected_keys = [pk[i] for i in positions]
        if not selected_keys or selected_keys != ek:
            raise ValueError("protein/ESM exact residue-key alignment mismatch")
        pnames = [str(protein["residue_names"][i]).upper() for i in positions]
        aa = [AA3.get(name, "X") for name in pnames]
        if aa != list(map(str, esm["residue_name_1"])):
            raise ValueError("protein/ESM residue identity mismatch")
        n_full, n = len(pk), len(positions)
        tails = {"physchem_features": (62,), "spatial_scalar_raw_features": (89,),
                 "spatial_vector_features": (8, 3), "ca_coords": (3,)}
        for field, tail in tails.items():
            finite_shape(protein[field], (n_full, *tail), field)
        finite_shape(esm["embeddings"], (n, 1280), "ESM2")
        ligand_xyz, n_heavy = ligand_check(ligand, sample)
        ligand_symbols, element_method, native_source = ligand_elements(ligand, ligand_xyz)
        if native_source:
            st = native_source.stat()
            provenance[str(native_source)] = dict(size=st.st_size, mtime_ns=st.st_mtime_ns)
        if sample["source"] == "pdbbind":
            atoms = protein["all_atom_coords"]
            names = protein["all_atom_names"]
            if len(atoms) != n_full or len(names) != n_full:
                raise ValueError("protein atom metadata length mismatch")
            heavy_atoms, heavy_symbols = [], []
            for i in positions:
                if len(atoms[i]) != len(names[i]):
                    raise ValueError("protein atom/name length mismatch")
                keep = [j for j, name in enumerate(names[i])
                        if not str(name).lstrip("0123456789").startswith(("H", "D"))]
                heavy_atoms.append(np.asarray(atoms[i])[keep])
                heavy_symbols.append(protein_elements(str(protein["residue_names"][i]).upper(),
                                                      [names[i][j] for j in keep]))
            protein_path = str(paths[0])
        else:
            root = Path(sample["biolip_root"])
            fixed = root/"fixed_receptors"/sample["pdb_id"]/"receptor_fixed.cif"
            original = root/"shared_receptors"/sample["pdb_id"]/"receptor_full.cif"
            receptor = fixed if protein.get("cif_source", "fixed") == "fixed" else original
            # Do not silently substitute another structure/coordinate frame.
            if not receptor.is_file():
                raise ValueError("declared source CIF missing")
            provenance[str(receptor)] = dict(size=receptor.stat().st_size, mtime_ns=receptor.stat().st_mtime_ns)
            parsed = cif_atoms(str(receptor), sample["selected_chain"])
            heavy_atoms, heavy_symbols, heavy_names = [], [], []
            for i, key_res, aa_res in zip(positions, selected_keys, aa):
                name, ca, xyz, symbols, names = parsed[key_res]
                if AA3.get(name, "X") != aa_res or np.linalg.norm(ca-np.asarray(protein["ca_coords"][i])) > .02:
                    raise ValueError("CIF/V4 residue identity or CA geometry mismatch")
                heavy_atoms.append(xyz)
                heavy_symbols.append(symbols)
                heavy_names.append(names)
            sliced = dict(protein)
            for field in (*tails, "spatial_scalar_features"):
                sliced[field] = torch.as_tensor(protein[field])[positions].clone()
            for field in ("residue_names", "residue_indices", "insertion_codes", "chain_ids"):
                sliced[field] = [protein[field][i] for i in positions]
            sliced["n_residues"] = n
            # PocketDatasetV8 also constructs GVP vectors from these atom lists.
            sliced["all_atom_coords"] = heavy_atoms
            sliced["all_atom_names"] = heavy_names
            protein_path = str(Path(output)/"chain_sidecars"/(sample["sample_id"]+"_protein.pt"))
        labels = count_contacts(heavy_atoms, heavy_symbols, ligand_xyz, ligand_symbols,
                                sample.get("min_atom_contacts", 2))
        distances = labels["distances"]
        n_contact = int(labels["contact_mask"].sum())
        record.update(n_contact=n_contact,
                      n_contact_casp1=int(labels["contact_mask_casp1"].sum()),
                      n_contact_biolip2=int(labels["contact_mask_biolip2"].sum()),
                      n_contact_lt5=int((distances < 5.).sum()),
                      n_contact_le5p5=int((distances <= 5.5).sum()),
                      ligand_element_source=element_method,
                      contact_label_protocol=labels["contact_label_protocol"])
        if n_contact < 3:
            raise ValueError(f"fewer_than_3_actual_contact_tags:{n_contact}")
        if int(labels["contact_mask"][:1800].sum()) < 3:
            raise ValueError("fewer_than_3_contact_tags_after_current_residue_cap_1800")
        if sample["source"] == "biolip":
            dst = Path(protein_path)
            dst.parent.mkdir(exist_ok=True, parents=True)
            torch.save(sliced, dst)
        dst = Path(output)/"labels"/(sample["sample_id"]+".pt")
        dst.parent.mkdir(exist_ok=True, parents=True)
        labels.update(n_residues=n, residue_keys=selected_keys,
                      source="native ligand heavy atoms vs aligned input residue heavy atoms",
                      ligand_element_source=element_method)
        torch.save(labels, dst)
        sequences, axis_hash = chain_sequences(esm)
        record.update(chains=sequences, axis_sha256=axis_hash,
                      n_residues=n, n_heavy_atoms=n_heavy, aligned_protein_path=protein_path,
                      labels_path=str(dst), labels_sha256=hashlib.sha256(dst.read_bytes()).hexdigest())
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record
