"""Versioned contact labels shared by preparation and the production loader.

Legacy distance files are intentionally not reinterpreted by these helpers.
"""

import torch


PROTOCOL_ID = "plc_vdw_atom_pairs_v1"
# Explicit project table, not a claim of bitwise reproduction of CASP/BioLiP.
# Values retain the existing PLC Bondi/extended table; unknowns have no fallback.
VDW_RADII = {
    "C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80, "P": 1.80,
    "F": 1.47, "Cl": 1.75, "Br": 1.85, "I": 1.98, "Se": 1.90,
    "B": 1.92, "Si": 2.10, "As": 1.85,
}


def contact_protocol(min_atom_contacts=2):
    if min_atom_contacts not in (1, 2):
        raise ValueError("min_atom_contacts must be 1 (CASP-style) or 2 (BioLiP-style)")
    return dict(
        id=PROTOCOL_ID, min_atom_contacts=min_atom_contacts, tolerance_angstrom=0.5,
        comparison="<=", heavy_atoms_only=True,
        radii_angstrom=dict(VDW_RADII), radii_table="PLC_Bondi_extended_v1",
        near_noncontact_dmin_angstrom=8.0, near_comparison="<",
        minimum_binding_residues_per_sample=3,
    )


def decode_contact_labels(entry):
    """Return class IDs and physical d_min, checking stored masks against counts.

Class 0 is the versioned contact mask. Of the remaining residues, class 1 has
physical d_min <8 A and class 2 is farther away. This near/far distinction is a
PLC auxiliary convention, not part of the CASP/BioLiP binding-site definition.
"""
    protocol = entry["contact_label_protocol"]
    if protocol != contact_protocol(protocol.get("min_atom_contacts")):
        raise ValueError("Unsupported or changed contact-label protocol")
    distances = torch.as_tensor(entry["distances"], dtype=torch.float32)
    counts = torch.as_tensor(entry["atom_contact_counts"])
    mask = torch.as_tensor(entry["contact_mask"])
    if (distances.ndim != 1 or counts.shape != distances.shape or mask.shape != distances.shape
            or counts.is_floating_point() or counts.dtype == torch.bool or mask.dtype != torch.bool
            or not bool(torch.isfinite(distances).all()) or bool((distances < 0).any())
            or bool((counts < 0).any())):
        raise ValueError("Invalid contact label shapes, dtypes or values")
    if not torch.equal(mask, counts >= protocol["min_atom_contacts"]):
        raise ValueError("Contact mask disagrees with atom-contact counts")
    for name, minimum in (("contact_mask_casp1", 1), ("contact_mask_biolip2", 2)):
        other = torch.as_tensor(entry[name])
        if other.dtype != torch.bool or not torch.equal(other, counts >= minimum):
            raise ValueError(f"Inconsistent diagnostic mask: {name}")
    if int(entry["n_residues"]) != len(distances) or len(entry["residue_keys"]) != len(distances):
        raise ValueError("Contact label residue axis mismatch")
    labels = torch.full_like(counts, 2, dtype=torch.long)
    labels[distances < protocol["near_noncontact_dmin_angstrom"]] = 1
    labels[mask] = 0
    return labels, distances


def validate_contact_axis(entry, protein, esm):
    """Fail rather than silently prefix-truncate a new-protocol label axis."""
    expected = [(str(c), int(i), str(code or "").strip()) for c, i, code in entry["residue_keys"]]
    if len(set(expected)) != len(expected):
        raise ValueError("Duplicate contact-label residue key")
    for data in (protein, esm):
        chains = data.get("chain_ids", data.get("chain_id"))
        indices = data.get("residue_indices", data.get("residue_index"))
        codes = data.get("insertion_codes", data.get("insertion_code"))
        if chains is None or indices is None or codes is None:
            raise ValueError("Contact labels require explicit feature residue axes")
        if not len(chains) == len(indices) == len(codes) == len(expected):
            raise ValueError("Contact label/feature axis length mismatch")
        actual = [(str(c), int(i), str(code or "").strip()) for c, i, code in zip(chains, indices, codes)]
        if actual != expected:
            raise ValueError("Contact label/feature residue-key mismatch")
