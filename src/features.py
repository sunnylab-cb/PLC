import numpy as np
import pypropel.fpseq as fpseq
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

# Amino Acid Mapping
AA_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

def get_residue_one_hot(residue_name):
    """Returns a 20-dim one-hot vector for the residue."""
    vec = np.zeros(20, dtype=np.float32)
    if residue_name in AA_MAP:
        vec[AA_MAP[residue_name]] = 1.0
    return vec

def get_residue_physchem(residue_name):
    """
    Returns simple physicochemical properties: [Charge, Hydrophobicity].
    Charge: -1 (Acidic), +1 (Basic), 0 (Neutral).
    Hydrophobicity: Kyte-Doolittle scale (normalized).
    """
    # Simple dictionary for demonstration
    # Charge
    charge_map = {
        'ARG': 1, 'LYS': 1, 'HIS': 0.5, # Slightly positive at pH 7
        'ASP': -1, 'GLU': -1,
    }
    charge = charge_map.get(residue_name, 0.0)

    # Kyte-Doolittle Hydrophobicity
    kd_map = {
        'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
        'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
        'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
        'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
    }
    hydro = kd_map.get(residue_name, 0.0) / 4.5 # Normalize roughly to [-1, 1]

    return np.array([charge, hydro], dtype=np.float32)

def get_global_protein_features(sequence):
    """
    Uses PyPropel to calculate global features for the whole sequence.
    Features:
    1. AAC (Amino Acid Composition) -> 20 dims
    2. DAC (Dipeptide Composition) -> 400 dims
    3. AVEANF (Amphiphilic Pseudo-Amino Acid Composition) -> 20 dims
    Total: 440 dimensions.
    """
    try:
        # 1. AAC (20 dims)
        aac_dict = fpseq.composition(seq=sequence, mode='aac')
        aac_values = np.array(list(aac_dict.values()), dtype=np.float32)
        
        # 2. DAC (400 dims)
        # Returns list of [pair, value], e.g. [['AA', 0.01], ['AC', 0.02]...]
        dac_list = fpseq.composition(seq=sequence, mode='dac')
        # Extract just the values
        dac_values = np.array([item[1] for item in dac_list], dtype=np.float32)
        
        # 3. AVEANF (20 dims)
        # Returns dict of values
        aveanf_dict = fpseq.composition(seq=sequence, mode='aveanf')
        aveanf_values = np.array(list(aveanf_dict.values()), dtype=np.float32)
        
        return np.concatenate([aac_values, dac_values, aveanf_values])
        
    except Exception as e:
        print(f"PyPropel Error: {e}")
        # Return zeros if failure (20 + 400 + 20 = 440)
        return np.zeros(440, dtype=np.float32)

def get_ligand_features(mol):
    """
    Extracts global ligand properties using RDKit.
    [LogP, MolWt, NumHDonors, NumHAcceptors]
    """
    if mol is None:
        return np.zeros(4, dtype=np.float32)
    
    logp = Crippen.MolLogP(mol)
    mw = Descriptors.MolWt(mol) / 500.0 # Normalize roughly
    h_donors = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    
    return np.array([logp, mw, h_donors, h_acceptors], dtype=np.float32)
