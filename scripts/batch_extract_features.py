#!/usr/bin/env python3
"""
Batch Feature Extraction for GVP-Fusion v2 Model

Extracts and saves features for protein-ligand complexes:
- ESM-2 embeddings (optional, requires fair-esm): [N, 1280]
- Protein scalar features (d=35): AA one-hot + SASA + SS + physchem
- Protein vector features (4 vectors): [N, 4, 3]
- Ligand atom features (d=64): atom props + global tag
- Distance classification labels: [N,] with 3 classes

Features are saved as .npz files in {complex}/features/ directory.

Usage:
    # Test on small subset
    python scripts/batch_extract_features.py --data_dir database/pdbbind/P-L --limit 10
    
    # Full extraction with ESM
    python scripts/batch_extract_features.py --data_dir database/pdbbind/P-L --esm --esm_device cuda
    
    # Skip DSSP (faster, uses zeros for SASA/SS)
    python scripts/batch_extract_features.py --data_dir database/pdbbind/P-L --no_dssp
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
from tqdm import tqdm
import traceback

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Direct Module Loading ====================
# Avoid triggering pypropel.__init__ which may have missing deps

def load_module(name: str, path: str):
    """Load module directly by file path."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Get pypropel path
SCRIPT_DIR = Path(__file__).parent
PYPROPEL_DIR = SCRIPT_DIR.parent / 'pypropel_unibio' / 'pypropel'

# Load modules (fpsite not needed - scalar features are self-contained)
gvp_module = None
mol_module = None
graph_module = None


def init_modules():
    """Initialize pypropel modules."""
    global gvp_module, mol_module, graph_module
    
    if gvp_module is None:
        gvp_module = load_module('gvp', str(PYPROPEL_DIR / 'gvp.py'))
    if mol_module is None:
        mol_module = load_module('mol', str(PYPROPEL_DIR / 'mol.py'))
    if graph_module is None:
        graph_module = load_module('graph', str(PYPROPEL_DIR / 'graph.py'))


# ==================== Three-Letter to One-Letter AA ====================

AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}


def three_to_one(res_name: str) -> str:
    """Convert 3-letter AA code to 1-letter."""
    return AA_3TO1.get(res_name.upper(), 'X')


# ==================== Protein Scalar Features (Self-Contained) ====================

AA_MAP = {
    'ALA': 0, 'ARG': 1, 'ASN': 2, 'ASP': 3, 'CYS': 4,
    'GLN': 5, 'GLU': 6, 'GLY': 7, 'HIS': 8, 'ILE': 9,
    'LEU': 10, 'LYS': 11, 'MET': 12, 'PHE': 13, 'PRO': 14,
    'SER': 15, 'THR': 16, 'TRP': 17, 'TYR': 18, 'VAL': 19
}

CHARGE_MAP = {'ARG': 1.0, 'LYS': 1.0, 'HIS': 0.5, 'ASP': -1.0, 'GLU': -1.0}

KD_HYDRO = {
    'ILE': 4.5, 'VAL': 4.2, 'LEU': 3.8, 'PHE': 2.8, 'CYS': 2.5,
    'MET': 1.9, 'ALA': 1.8, 'GLY': -0.4, 'THR': -0.7, 'SER': -0.8,
    'TRP': -0.9, 'TYR': -1.3, 'PRO': -1.6, 'HIS': -3.2, 'GLU': -3.5,
    'GLN': -3.5, 'ASP': -3.5, 'ASN': -3.5, 'LYS': -3.9, 'ARG': -4.5
}

AROMATIC_AAS = {'PHE', 'TYR', 'TRP', 'HIS'}
HBOND_DONOR = {'ARG': 3, 'ASN': 1, 'GLN': 1, 'HIS': 1, 'LYS': 1, 'SER': 1, 'THR': 1, 'TRP': 1, 'TYR': 1, 'CYS': 1}
HBOND_ACCEPTOR = {'ASN': 1, 'ASP': 2, 'GLN': 1, 'GLU': 2, 'HIS': 1, 'SER': 1, 'THR': 1, 'TYR': 1}


def compute_protein_scalar_features(structure, chain_id=None) -> np.ndarray:
    """
    Self-contained protein scalar feature extraction (d=35).
    
    [0-19] AA one-hot, [20-22] SS placeholder, [23] SASA placeholder,
    [24] Charge, [25] Hydro, [26] Aromatic, [27] H-Donor, [28] H-Accept, [29-34] Reserved
    """
    from Bio.PDB import Polypeptide
    
    res_names = []
    for model in structure:
        for chain in model:
            if chain_id is not None and chain.get_id() != chain_id:
                continue
            for residue in chain:
                if Polypeptide.is_aa(residue, standard=True):
                    res_names.append(residue.get_resname())
    
    n = len(res_names)
    if n == 0:
        return np.zeros((0, 35), dtype=np.float32)
    
    features = np.zeros((n, 35), dtype=np.float32)
    
    for i, res_name in enumerate(res_names):
        # [0-19] AA one-hot
        if res_name in AA_MAP:
            features[i, AA_MAP[res_name]] = 1.0
        
        # [20-22] SS placeholder (zeros without DSSP)
        # [23] SASA placeholder (zero without DSSP)
        
        # [24] Charge
        features[i, 24] = CHARGE_MAP.get(res_name, 0.0)
        
        # [25] Hydrophobicity (normalized 0-1)
        hydro = KD_HYDRO.get(res_name, 0.0)
        features[i, 25] = (hydro + 4.5) / 9.0  # Normalize
        
        # [26] Aromatic
        features[i, 26] = 1.0 if res_name in AROMATIC_AAS else 0.0
        
        # [27-28] H-bond
        features[i, 27] = HBOND_DONOR.get(res_name, 0)
        features[i, 28] = HBOND_ACCEPTOR.get(res_name, 0)
    
    return features


# ==================== Feature Extraction Functions ====================


def extract_protein_features(
    pdb_path: str,
    k_neighbors: int = 10,
    use_dssp: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract protein features for GVP-Fusion v2.
    
    Returns:
        Dict with:
        - 'ca_coords': (N, 3) - Cα coordinates
        - 'vector_features': (N, 4, 3) - 4 GVP vectors
        - 'scalar_features': (N, 35) - Physchem features
        - 'sequence': str - AA sequence
        - 'edge_index': (2, E) - k-NN edges
        - 'edge_distances': (E,) - Edge distances
    """
    init_modules()
    from Bio.PDB import PDBParser, Polypeptide
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    # Extract GVP features (now returns 4 vectors)
    ca_coords, vector_features = gvp_module.get_gvp_node_features(
        structure, k_neighbors=k_neighbors
    )
    
    if len(ca_coords) == 0:
        raise ValueError("No valid residues found in protein")
    
    # Extract scalar features (d=35) - self-contained, no DSSP
    scalar_features = compute_protein_scalar_features(structure)
    
    # Build k-NN edges
    edge_index, edge_distances = gvp_module.build_knn_edges(
        ca_coords, k=20, radius=10.0
    )
    
    # Extract sequence
    sequence = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if Polypeptide.is_aa(residue, standard=True):
                    aa = three_to_one(residue.get_resname())
                    sequence.append(aa)
    
    return {
        'ca_coords': ca_coords,
        'vector_features': vector_features,
        'scalar_features': scalar_features,
        'sequence': ''.join(sequence),
        'edge_index': edge_index,
        'edge_distances': edge_distances,
    }


def extract_ligand_features(sdf_path: str) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract ligand features for GVP-Fusion v2.
    
    Returns:
        Dict with:
        - 'atom_features': (M, 64) - Extended atom features
        - 'atom_coords': (M, 3) - 3D coordinates
        - 'edge_index': (2, E) - Bond connectivity
        - 'bond_features': (E, 4) - Bond type one-hot
        - 'global_features': (132,) - Global molecular features
    """
    init_modules()
    
    try:
        mol = mol_module.load_sdf(sdf_path)
    except ImportError:
        logger.error("RDKit not available")
        return None
    
    if mol is None:
        return None
    
    # Extended atom features (d=64)
    atom_features = mol_module.get_atom_features_extended(mol)
    atom_coords = mol_module.ligand_coords(mol)
    global_features = mol_module.get_ligand_global_features(mol)
    
    # Graph structure
    ligand_graph = graph_module.build_ligand_graph(mol)
    
    return {
        'atom_features': atom_features,
        'atom_coords': atom_coords,
        'edge_index': ligand_graph['edge_index'],
        'bond_features': ligand_graph['edge_attr'],
        'global_features': global_features,
    }


def extract_distance_labels(
    pdb_path: str,
    sdf_path: str,
    thresholds: List[float] = [3.5, 6.0]
) -> Optional[Dict[str, np.ndarray]]:
    """
    Extract distance-based classification labels.
    
    Returns:
        Dict with:
        - 'labels': (N,) - Class labels (0=contact, 1=near, 2=far)
        - 'distances': (N,) - Min distances to ligand
        - 'class_weights': (num_classes,) - Inverse frequency weights
    """
    init_modules()
    from Bio.PDB import PDBParser, Polypeptide
    
    # Load ligand
    try:
        mol = mol_module.load_sdf(sdf_path)
    except ImportError:
        return None
    
    if mol is None:
        return None
    
    ligand_coords = mol_module.ligand_coords(mol)
    
    # Load protein
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    # Compute min distance per residue
    distances = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not Polypeptide.is_aa(residue, standard=True):
                    continue
                
                res_coords = []
                for atom in residue:
                    if atom.element != 'H':
                        res_coords.append(atom.get_coord())
                
                if len(res_coords) == 0:
                    distances.append(float('inf'))
                    continue
                
                res_coords = np.array(res_coords)
                diff = res_coords[:, None, :] - ligand_coords[None, :, :]
                dist_matrix = np.linalg.norm(diff, axis=2)
                distances.append(dist_matrix.min())
    
    distances = np.array(distances, dtype=np.float32)
    
    # Classify
    labels = np.zeros(len(distances), dtype=np.int64)
    for i, d in enumerate(distances):
        for j, thresh in enumerate(sorted(thresholds)):
            if d < thresh:
                labels[i] = j
                break
        else:
            labels[i] = len(thresholds)
    
    # Class weights
    n_classes = len(thresholds) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    weights = n_classes / (n_classes * counts)
    
    return {
        'labels': labels,
        'distances': distances,
        'class_weights': weights,
    }


def extract_esm_embeddings(
    sequence: str,
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = 'cpu'
) -> Optional[np.ndarray]:
    """Extract ESM-2 embeddings (L, 1280)."""
    try:
        import torch
        import esm
    except ImportError:
        logger.warning("ESM not available. Install: pip install fair-esm")
        return None
    
    global _esm_model, _esm_batch_converter, _esm_device
    
    # Load model (cached)
    if '_esm_model' not in globals() or _esm_model is None:
        logger.info(f"Loading ESM model: {model_name}...")
        _esm_model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        _esm_model.eval()
        _esm_model = _esm_model.to(device)
        _esm_batch_converter = alphabet.get_batch_converter()
        _esm_device = device
    
    # Extract
    data = [("protein", sequence)]
    _, _, batch_tokens = _esm_batch_converter(data)
    batch_tokens = batch_tokens.to(_esm_device)
    
    with torch.no_grad():
        results = _esm_model(batch_tokens, repr_layers=[_esm_model.num_layers])
    
    embeddings = results["representations"][_esm_model.num_layers][0, 1:-1].cpu().numpy()
    return embeddings.astype(np.float32)


# ==================== Main Processing ====================

def find_complexes(data_dir: str) -> List[Dict]:
    """Find all protein-ligand complexes."""
    complexes = []
    data_path = Path(data_dir)
    
    for subdir in data_path.iterdir():
        if not subdir.is_dir():
            continue
        
        for pdb_dir in subdir.iterdir():
            if not pdb_dir.is_dir():
                continue
            
            pdb_id = pdb_dir.name
            protein_file = pdb_dir / f"{pdb_id}_protein.pdb"
            ligand_file = pdb_dir / f"{pdb_id}_ligand.sdf"
            
            if protein_file.exists() and ligand_file.exists():
                complexes.append({
                    'pdb_id': pdb_id,
                    'protein_pdb': str(protein_file),
                    'ligand_sdf': str(ligand_file),
                    'output_dir': str(pdb_dir),
                })
    
    return complexes


def process_complex(
    complex_info: Dict,
    extract_esm: bool = False,
    esm_device: str = 'cpu',
    k_neighbors: int = 10,
    thresholds: List[float] = [3.5, 6.0],
    use_dssp: bool = True,
    overwrite: bool = False,
) -> Dict:
    """Process a single complex."""
    pdb_id = complex_info['pdb_id']
    output_dir = Path(complex_info['output_dir']) / 'features'
    
    result = {'pdb_id': pdb_id, 'success': False, 'error': None}
    
    if not overwrite and (output_dir / 'protein_features.npz').exists():
        result['success'] = True
        result['skipped'] = True
        return result
    
    try:
        output_dir.mkdir(exist_ok=True)
        
        # Protein features
        protein = extract_protein_features(
            complex_info['protein_pdb'],
            k_neighbors=k_neighbors,
            use_dssp=use_dssp
        )
        
        np.savez_compressed(
            output_dir / 'protein_features.npz',
            ca_coords=protein['ca_coords'],
            vector_features=protein['vector_features'],
            scalar_features=protein['scalar_features'],
            edge_index=protein['edge_index'],
            edge_distances=protein['edge_distances'],
        )
        
        # Save sequence
        with open(output_dir / 'sequence.txt', 'w') as f:
            f.write(protein['sequence'])
        
        # Ligand features
        ligand = extract_ligand_features(complex_info['ligand_sdf'])
        if ligand is not None:
            np.savez_compressed(
                output_dir / 'ligand_features.npz',
                atom_features=ligand['atom_features'],
                atom_coords=ligand['atom_coords'],
                edge_index=ligand['edge_index'],
                bond_features=ligand['bond_features'],
                global_features=ligand['global_features'],
            )
        
        # Labels
        labels = extract_distance_labels(
            complex_info['protein_pdb'],
            complex_info['ligand_sdf'],
            thresholds=thresholds,
        )
        if labels is not None:
            np.savez_compressed(
                output_dir / 'labels.npz',
                labels=labels['labels'],
                distances=labels['distances'],
                class_weights=labels['class_weights'],
                thresholds=np.array(thresholds),
            )
        
        # ESM embeddings
        if extract_esm and protein['sequence']:
            embeddings = extract_esm_embeddings(protein['sequence'], device=esm_device)
            if embeddings is not None:
                np.savez_compressed(output_dir / 'esm_embeddings.npz', embeddings=embeddings)
        
        result['success'] = True
        result['n_residues'] = len(protein['ca_coords'])
        result['n_atoms'] = len(ligand['atom_features']) if ligand else 0
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error processing {pdb_id}: {e}")
        logger.debug(traceback.format_exc())
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Batch extract GVP-Fusion v2 features')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--esm', action='store_true')
    parser.add_argument('--esm_device', type=str, default='cpu')
    parser.add_argument('--k_neighbors', type=int, default=10)
    parser.add_argument('--thresholds', type=float, nargs='+', default=[3.5, 6.0])
    parser.add_argument('--no_dssp', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--output_stats', type=str, default=None)
    
    args = parser.parse_args()
    
    logger.info(f"Finding complexes in {args.data_dir}...")
    complexes = find_complexes(args.data_dir)
    logger.info(f"Found {len(complexes)} complexes")
    
    if args.limit:
        complexes = complexes[:args.limit]
        logger.info(f"Limited to {len(complexes)} complexes")
    
    success, skip, error = 0, 0, 0
    results = []
    
    for info in tqdm(complexes, desc="Processing"):
        result = process_complex(
            info,
            extract_esm=args.esm,
            esm_device=args.esm_device,
            k_neighbors=args.k_neighbors,
            thresholds=args.thresholds,
            use_dssp=not args.no_dssp,
            overwrite=args.overwrite,
        )
        results.append(result)
        
        if result.get('skipped'):
            skip += 1
        elif result['success']:
            success += 1
        else:
            error += 1
    
    logger.info("=" * 50)
    logger.info(f"Success: {success}, Skipped: {skip}, Errors: {error}")
    logger.info("=" * 50)
    
    if args.output_stats:
        with open(args.output_stats, 'w') as f:
            json.dump({'success': success, 'skipped': skip, 'errors': error, 'results': results}, f, indent=2)


if __name__ == '__main__':
    main()
