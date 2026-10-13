"""Heavy-atom pair counting; no default element/radius or optimized coordinates."""

from pathlib import Path
import sys

import numpy as np
import torch
from scipy.spatial.distance import cdist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.contact_labels import VDW_RADII, contact_protocol


ATOM_TYPES = ("C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "H", "Other")
STANDARD_ATOMS = {
    "ALA": "N CA C O CB OXT", "ARG": "N CA C O CB CG CD NE CZ NH1 NH2 OXT",
    "ASN": "N CA C O CB CG OD1 ND2 OXT", "ASP": "N CA C O CB CG OD1 OD2 OXT",
    "CYS": "N CA C O CB SG OXT", "GLN": "N CA C O CB CG CD OE1 NE2 OXT",
    "GLU": "N CA C O CB CG CD OE1 OE2 OXT", "GLY": "N CA C O OXT",
    "HIS": "N CA C O CB CG ND1 CD2 CE1 NE2 OXT", "ILE": "N CA C O CB CG1 CG2 CD1 OXT",
    "LEU": "N CA C O CB CG CD1 CD2 OXT", "LYS": "N CA C O CB CG CD CE NZ OXT",
    "MET": "N CA C O CB CG SD CE OXT", "PHE": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ OXT",
    "PRO": "N CA C O CB CG CD OXT", "SER": "N CA C O CB OG OXT",
    "THR": "N CA C O CB OG1 CG2 OXT", "TRP": "N CA C O CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2 OXT",
    "TYR": "N CA C O CB CG CD1 CD2 CE1 CE2 CZ OH OXT", "VAL": "N CA C O CB CG1 CG2 OXT",
    "MSE": "N CA C O CB CG SE CE OXT", "SEC": "N CA C O CB SE OXT",
}
RESIDUE_ALIASES = dict(HID="HIS", HIE="HIS", HIP="HIS", ASH="ASP", GLH="GLU",
                       CYX="CYS", CYM="CYS", LYN="LYS")


def protein_elements(resname, names):
    """Canonical amino-acid nomenclature: CA is carbon, not calcium."""
    resname = RESIDUE_ALIASES.get(resname, resname)
    allowed = set(STANDARD_ATOMS.get(resname, "").split())
    result, seen = [], set()
    for name in names:
        name = str(name).strip().upper()
        if name in seen:
            raise ValueError("Duplicate protein atom name/alternate conformer")
        seen.add(name)
        if name not in allowed:
            raise ValueError(f"Protein element unavailable for {resname}:{name}")
        result.append("Se" if name == "SE" else name[0])
    return result


def ligand_elements(ligand, native_xyz):
    """Use source SDF where available, otherwise exact raw V5 identity channels.

V5 (and regular-organic V6) saves native heavy coordinates in original atom
order and only appends hydrogens. The Other channel cannot supply an element.
"""
    x = torch.as_tensor(ligand["atom_features"])[:, :11]
    if not bool(((x == 0) | (x == 1)).all()) or not bool((x.sum(1) == 1).all()):
        raise ValueError("Invalid raw ligand atom-type one-hot")
    all_symbols = [ATOM_TYPES[i] for i in x.argmax(1).tolist()]
    symbols = [s for s in all_symbols if s != "H"]
    if len(symbols) != len(native_xyz):
        raise ValueError("Ligand native-coordinate/element count mismatch")
    source = Path(ligand.get("ligand_path", "__missing_native_source__"))
    if source.is_file() and source.suffix.lower() in (".sdf", ".mol"):
        from rdkit import Chem
        if source.suffix.lower() == ".sdf":
            mol = next(Chem.SDMolSupplier(str(source), sanitize=False, removeHs=False), None)
        else:
            mol = Chem.MolFromMolFile(str(source), sanitize=False, removeHs=False)
        if mol is None or mol.GetNumConformers() < 1:
            raise ValueError("Cannot parse native ligand source")
        atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
        xyz = np.asarray(mol.GetConformer().GetPositions())[[a.GetIdx() for a in atoms]]
        resolved = [a.GetSymbol() for a in atoms]
        if xyz.shape != native_xyz.shape or not np.allclose(xyz, native_xyz, atol=.002, rtol=0):
            raise ValueError("Native source/cache heavy-atom order or coordinates disagree")
        if any(a != b and a != "Other" for a, b in zip(symbols, resolved)):
            raise ValueError("Native source/cache ligand atom types disagree")
        symbols = resolved
        method = "coordinate-checked native SDF/MOL"
    else:
        method = "raw V5/V6 identity one-hot, extractor-preserved native heavy order"
    if any(s not in VDW_RADII for s in symbols):
        raise ValueError(f"Unsupported/unknown ligand element; no radius fallback: {sorted(set(symbols)-VDW_RADII.keys())}")
    return symbols, method, source if source.is_file() else None


def count_contacts(residue_xyz, residue_symbols, ligand_xyz, ligand_symbols, min_atom_contacts=2):
    """Each unique protein-atom/ligand-atom pair contributes at most one contact."""
    protocol = contact_protocol(min_atom_contacts)
    ligand_xyz = np.asarray(ligand_xyz, dtype=np.float64)
    if (ligand_xyz.shape != (len(ligand_symbols), 3) or not len(ligand_xyz)
            or not np.isfinite(ligand_xyz).all()):
        raise ValueError("Invalid ligand geometry")
    if len(residue_xyz) != len(residue_symbols) or not len(residue_xyz):
        raise ValueError("Invalid protein geometry/element axes")
    lr = np.asarray([VDW_RADII[s] for s in ligand_symbols])
    distances, gaps, counts = [], [], []
    for xyz, symbols in zip(residue_xyz, residue_symbols):
        xyz = np.asarray(xyz, dtype=np.float64)
        if xyz.shape != (len(symbols), 3) or not len(xyz) or not np.isfinite(xyz).all():
            raise ValueError("Invalid residue geometry")
        pr = np.asarray([VDW_RADII[s] for s in symbols])
        d = cdist(xyz, ligand_xyz)
        radii_sum = pr[:, None] + lr[None, :]
        distances.append(float(d.min()))
        gaps.append(float((d-radii_sum).min()))
        counts.append(int(np.count_nonzero(d <= radii_sum + protocol["tolerance_angstrom"])))
    counts = torch.tensor(counts, dtype=torch.long)
    return dict(contact_label_protocol=protocol,
        distances=torch.tensor(distances, dtype=torch.float32),
        vdw_surface_gap=torch.tensor(gaps, dtype=torch.float32), atom_contact_counts=counts,
        contact_mask=counts >= min_atom_contacts,
        contact_mask_casp1=counts >= 1, contact_mask_biolip2=counts >= 2)
