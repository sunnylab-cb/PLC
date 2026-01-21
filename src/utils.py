import numpy as np
import warnings
import pypropel.str as ppstr
import pypropel.dist as ppdist
import pypropel.mol as ppmol
import pypropel.fpsite as fpsite


def load_structure(pdb_path):
    """Parses a PDB file and returns a Bio.PDB structure object using pypropel."""
    return ppstr.load_pdb(pdb_path)


def load_ligand_sdf(sdf_path):
    """Parses a ligand SDF file and returns an RDKit Mol object using pypropel."""
    return ppmol.load_sdf(sdf_path)


def load_ligand_mol2(mol2_path):
    """Parses a ligand MOL2 file and returns an RDKit Mol object using pypropel."""
    return ppmol.load_mol2(mol2_path)


def get_atom_coords(residue):
    """Returns atom coordinates for a residue using pypropel."""
    return fpsite.residue_coords(residue)


def get_ligand_coords(ligand):
    """Returns atom coordinates for an RDKit ligand using pypropel."""
    return ppmol.ligand_coords(ligand)


def calculate_min_distance(residue_coords, ligand_coords):
    """
    Calculates the minimum distance between any atom in the residue
    and any atom in the ligand using pypropel.
    """
    return ppdist.residue_ligand_distance(residue_coords, ligand_coords)


def get_sequence_from_structure(structure):
    """Extracts the amino acid sequence from all polypeptide chains using pypropel."""
    return ppstr.structure_to_sequence(structure)
