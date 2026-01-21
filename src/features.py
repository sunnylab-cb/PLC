import numpy as np
import pypropel.fpseq as fpseq
import pypropel.fpsite as fpsite
import pypropel.mol as ppmol


def get_residue_one_hot(residue_name):
    """Returns a 20-dim one-hot vector for the residue using pypropel."""
    return fpsite.residue_one_hot(residue_name)


def get_residue_physchem(residue_name):
    """
    Returns simple physicochemical properties: [Charge, Hydrophobicity] using pypropel.
    """
    return fpsite.residue_physchem(residue_name)


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
    Extracts global ligand properties using pypropel.
    [LogP, MolWt, NumHDonors, NumHAcceptors]
    """
    return ppmol.get_ligand_features(mol)
