import numpy as np
import Bio.PDB
from rdkit import Chem
import warnings

def load_structure(pdb_path):
    """Parses a PDB file and returns a Bio.PDB structure object."""
    parser = Bio.PDB.PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
        return structure
    except Exception as e:
        print(f"Error loading PDB {pdb_path}: {e}")
        return None

def load_ligand_sdf(sdf_path):
    """Parses a ligand SDF file and returns an RDKit Mol object."""
    suppl = Chem.SDMolSupplier(sdf_path)
    for mol in suppl:
        if mol is not None:
             return mol
    return None

def load_ligand_mol2(mol2_path):
    """Parses a ligand MOL2 file and returns an RDKit Mol object."""
    return Chem.MolFromMol2File(mol2_path)

def get_atom_coords(residue):
    """Returns a list of coordinates for all atoms in a residue."""
    coords = []
    for atom in residue:
        coords.append(atom.get_coord())
    return np.array(coords)

def get_ligand_coords(ligand):
    """Returns a list of coordinates for all atoms in an RDKit ligand."""
    conf = ligand.GetConformer()
    coords = []
    for i in range(ligand.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        coords.append(np.array([pos.x, pos.y, pos.z]))
    return np.array(coords)

def calculate_min_distance(residue_coords, ligand_coords):
    """
    Calculates the minimum distance between any atom in the residue
    and any atom in the ligand.
    """
    if len(residue_coords) == 0 or len(ligand_coords) == 0:
        return float('inf')
    
    # Calculate pairwise distances
    # residue_coords: (N, 3), ligand_coords: (M, 3)
    # expand dims to broadcasting: (N, 1, 3) - (1, M, 3) -> (N, M, 3)
    diff = residue_coords[:, np.newaxis, :] - ligand_coords[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1) # (N, M)
    min_dist = np.sqrt(np.min(dist_sq))
    return min_dist

def get_sequence_from_structure(structure):
    """Extracts the amino acid sequence from the first model/chain."""
    ppb = Bio.PDB.PPBuilder()
    seq = ""
    for pp in ppb.build_peptides(structure):
        seq += str(pp.get_sequence())
    return seq
