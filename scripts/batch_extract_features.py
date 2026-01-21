#!/usr/bin/env python3
"""
Batch Feature Extraction for GVP-Fusion Model

This script extracts and saves features for protein-ligand complexes:
- ESM-2 embeddings (optional, requires fair-esm)
- GVP vector features (Cα/Cβ coords, orientations, backbone)
- Ligand atom features (types, hybridization, H-bond, charges)
- Morgan fingerprints
- Distance classification labels

Features are saved as .npz files alongside each complex.

Usage:
    python scripts/batch_extract_features.py --data_dir /path/to/pdbbind/P-L --limit 10
    python scripts/batch_extract_features.py --data_dir /path/to/pdbbind/P-L --esm  # Include ESM embeddings
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from tqdm import tqdm
import traceback

# Add pypropel to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'pypropel_unibio'))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Feature Extraction Functions ====================

def extract_protein_features(
    pdb_path: str,
    k_neighbors: int = 10
) -> Dict[str, np.ndarray]:
    """
    Extract protein features for GVP encoder.
    
    Returns:
        Dict with keys:
        - 'ca_coords': (N, 3) Cα coordinates
        - 'vector_features': (N, 3, 3) GVP vector features
        - 'sequence': str amino acid sequence
        - 'residue_ids': list of residue IDs
    """
    from Bio.PDB import PDBParser, Polypeptide
    
    # Handle different BioPython versions
    try:
        from Bio.PDB.Polypeptide import three_to_one
    except ImportError:
        try:
            from Bio.SeqUtils import seq3, seq1
            def three_to_one(res_name):
                return seq1(res_name)
        except ImportError:
            # Fallback mapping
            AA_MAP = {
                'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
            }
            def three_to_one(res_name):
                return AA_MAP.get(res_name.upper(), 'X')
    
    # Load directly to avoid pypropel init issues
    import importlib.util
    gvp_path = os.path.join(os.path.dirname(__file__), '..', 'pypropel_unibio', 'pypropel', 'gvp.py')
    spec = importlib.util.spec_from_file_location('gvp', gvp_path)
    gvp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gvp)
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    
    # Extract GVP features
    ca_coords, vector_features = gvp.get_gvp_node_features(structure, k_neighbors=k_neighbors)
    
    # Extract sequence and residue IDs
    sequence = []
    residue_ids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if Polypeptide.is_aa(residue, standard=True):
                    try:
                        aa = three_to_one(residue.get_resname())
                        sequence.append(aa)
                        residue_ids.append((chain.get_id(), residue.get_id()[1]))
                    except (KeyError, ValueError):
                        continue
    
    return {
        'ca_coords': ca_coords,
        'vector_features': vector_features,
        'sequence': ''.join(sequence),
        'residue_ids': residue_ids,
    }


def extract_ligand_features(sdf_path: str) -> Dict[str, np.ndarray]:
    """
    Extract ligand features for GIN encoder.
    
    Returns:
        Dict with keys:
        - 'atom_features': (N, 21) combined atom features
        - 'atom_coords': (N, 3) 3D coordinates
        - 'edge_index': (2, E) bond connectivity
        - 'bond_features': (E, 4) bond type one-hot
        - 'global_features': (132,) global molecular features
    """
    import importlib.util
    mol_path = os.path.join(os.path.dirname(__file__), '..', 'pypropel_unibio', 'pypropel', 'mol.py')
    spec = importlib.util.spec_from_file_location('mol', mol_path)
    mol_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mol_module)
    
    graph_path = os.path.join(os.path.dirname(__file__), '..', 'pypropel_unibio', 'pypropel', 'graph.py')
    spec = importlib.util.spec_from_file_location('graph', graph_path)
    graph_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_module)
    
    mol = mol_module.load_sdf(sdf_path)
    if mol is None:
        return None
    
    # Atom features
    atom_features = mol_module.get_atom_features(mol)
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
    protein_pdb: str,
    ligand_sdf: str,
    thresholds: List[float] = [3.5, 6.0]
) -> Dict[str, np.ndarray]:
    """
    Extract distance-based classification labels.
    
    Returns:
        Dict with keys:
        - 'labels': (N,) classification labels (0=contact, 1=near, 2=far)
        - 'distances': (N,) minimum distances to ligand
        - 'class_weights': (num_classes,) inverse frequency weights
    """
    from Bio.PDB import PDBParser, Polypeptide
    
    import importlib.util
    mol_path = os.path.join(os.path.dirname(__file__), '..', 'pypropel_unibio', 'pypropel', 'mol.py')
    spec = importlib.util.spec_from_file_location('mol', mol_path)
    mol_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mol_module)
    
    # Load ligand coords
    mol = mol_module.load_sdf(ligand_sdf)
    if mol is None:
        return None
    ligand_coords = mol_module.ligand_coords(mol)
    
    # Load protein and compute distances
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', protein_pdb)
    
    distances = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if not Polypeptide.is_aa(residue, standard=True):
                    continue
                
                # Get all heavy atom coords for residue
                res_coords = []
                for atom in residue:
                    if atom.element != 'H':
                        res_coords.append(atom.get_coord())
                
                if len(res_coords) == 0:
                    distances.append(float('inf'))
                    continue
                
                res_coords = np.array(res_coords)
                
                # Compute minimum distance to ligand
                diff = res_coords[:, None, :] - ligand_coords[None, :, :]
                dist_matrix = np.linalg.norm(diff, axis=2)
                min_dist = dist_matrix.min()
                distances.append(min_dist)
    
    distances = np.array(distances, dtype=np.float32)
    
    # Classify distances
    labels = np.zeros(len(distances), dtype=np.int64)
    sorted_thresholds = sorted(thresholds)
    for i, d in enumerate(distances):
        for j, thresh in enumerate(sorted_thresholds):
            if d < thresh:
                labels[i] = j
                break
        else:
            labels[i] = len(sorted_thresholds)
    
    # Compute class weights
    n_classes = len(thresholds) + 1
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)  # Avoid division by zero
    weights = 1.0 / counts
    weights = weights * n_classes / weights.sum()  # Normalize
    
    return {
        'labels': labels,
        'distances': distances,
        'class_weights': weights,
    }


def extract_esm_embeddings(
    sequence: str,
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = 'cpu'
) -> np.ndarray:
    """
    Extract ESM-2 embeddings for a protein sequence.
    
    Returns:
        embeddings: (L, 1280) per-residue embeddings
    """
    try:
        import torch
        import esm
    except ImportError:
        logger.warning("ESM not available. Install with: pip install fair-esm")
        return None
    
    # Load model (cached globally)
    global _esm_model, _esm_batch_converter
    if '_esm_model' not in globals() or _esm_model is None:
        logger.info(f"Loading ESM model: {model_name}...")
        _esm_model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        _esm_model.eval()
        _esm_model = _esm_model.to(device)
        _esm_batch_converter = alphabet.get_batch_converter()
    
    # Prepare input
    data = [("protein", sequence)]
    _, _, batch_tokens = _esm_batch_converter(data)
    batch_tokens = batch_tokens.to(device)
    
    # Extract embeddings
    with torch.no_grad():
        results = _esm_model(batch_tokens, repr_layers=[_esm_model.num_layers])
    
    # Get last layer embeddings, exclude BOS/EOS tokens
    embeddings = results["representations"][_esm_model.num_layers][0, 1:-1].cpu().numpy()
    
    return embeddings.astype(np.float32)


# ==================== Main Processing Functions ====================

def find_complexes(data_dir: str) -> List[Dict]:
    """
    Find all protein-ligand complexes in PDBbind directory structure.
    
    Expected structure:
        data_dir/
            2011-2019/
                {pdb_id}/
                    {pdb_id}_protein.pdb
                    {pdb_id}_ligand.sdf
    """
    complexes = []
    data_path = Path(data_dir)
    
    # Handle different directory structures
    for subdir in data_path.iterdir():
        if not subdir.is_dir():
            continue
        
        # Check if this is a year-range directory or directly PDB IDs
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
    
    # Also check for direct subdirectories (e.g., P-NA structure)
    for pdb_dir in data_path.iterdir():
        if not pdb_dir.is_dir():
            continue
        if pdb_dir.name.startswith('.'):
            continue
        
        # Check for complex.pdb format
        pdb_id = pdb_dir.name
        if (pdb_dir / f"{pdb_id}_protein.pdb").exists():
            continue  # Already added above
        
        # Check for {id}_complex.pdb format
        complex_file = pdb_dir / f"{pdb_id}_complex.pdb"
        if complex_file.exists():
            # This format doesn't have separate ligand file
            pass
    
    return complexes


def process_complex(
    complex_info: Dict,
    extract_esm: bool = False,
    esm_device: str = 'cpu',
    k_neighbors: int = 10,
    thresholds: List[float] = [3.5, 6.0],
    overwrite: bool = False,
) -> Dict:
    """
    Process a single protein-ligand complex.
    
    Saves features to:
        {output_dir}/features/
            protein_features.npz
            ligand_features.npz
            labels.npz
            esm_embeddings.npz (optional)
    """
    pdb_id = complex_info['pdb_id']
    output_dir = Path(complex_info['output_dir']) / 'features'
    
    result = {'pdb_id': pdb_id, 'success': False, 'error': None}
    
    # Check if already processed
    if not overwrite and (output_dir / 'protein_features.npz').exists():
        result['success'] = True
        result['skipped'] = True
        return result
    
    try:
        output_dir.mkdir(exist_ok=True)
        
        # Extract protein features
        protein_features = extract_protein_features(
            complex_info['protein_pdb'],
            k_neighbors=k_neighbors
        )
        
        if protein_features['ca_coords'].shape[0] == 0:
            raise ValueError("No valid residues found in protein")
        
        # Save protein features
        np.savez_compressed(
            output_dir / 'protein_features.npz',
            ca_coords=protein_features['ca_coords'],
            vector_features=protein_features['vector_features'],
            sequence=protein_features['sequence'],
        )
        
        # Save residue IDs as JSON (for debugging/mapping)
        with open(output_dir / 'residue_ids.json', 'w') as f:
            json.dump(protein_features['residue_ids'], f)
        
        # Extract ligand features
        ligand_features = extract_ligand_features(complex_info['ligand_sdf'])
        if ligand_features is not None:
            np.savez_compressed(
                output_dir / 'ligand_features.npz',
                atom_features=ligand_features['atom_features'],
                atom_coords=ligand_features['atom_coords'],
                edge_index=ligand_features['edge_index'],
                bond_features=ligand_features['bond_features'],
                global_features=ligand_features['global_features'],
            )
        
        # Extract labels
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
        
        # Extract ESM embeddings (optional)
        if extract_esm and protein_features['sequence']:
            embeddings = extract_esm_embeddings(
                protein_features['sequence'],
                device=esm_device
            )
            if embeddings is not None:
                np.savez_compressed(
                    output_dir / 'esm_embeddings.npz',
                    embeddings=embeddings,
                )
        
        result['success'] = True
        result['n_residues'] = protein_features['ca_coords'].shape[0]
        result['n_atoms'] = ligand_features['atom_features'].shape[0] if ligand_features else 0
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Error processing {pdb_id}: {e}")
        logger.debug(traceback.format_exc())
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Batch extract features for GVP-Fusion model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to PDBbind P-L directory')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit number of complexes to process (for testing)')
    parser.add_argument('--esm', action='store_true',
                       help='Extract ESM-2 embeddings (requires fair-esm)')
    parser.add_argument('--esm_device', type=str, default='cpu',
                       help='Device for ESM model (cpu, cuda)')
    parser.add_argument('--k_neighbors', type=int, default=10,
                       help='Number of neighbors for GVP features')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[3.5, 6.0],
                       help='Distance thresholds for classification')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing feature files')
    parser.add_argument('--output_stats', type=str, default=None,
                       help='Path to save processing statistics')
    
    args = parser.parse_args()
    
    logger.info(f"Finding complexes in {args.data_dir}...")
    complexes = find_complexes(args.data_dir)
    logger.info(f"Found {len(complexes)} complexes")
    
    if args.limit:
        complexes = complexes[:args.limit]
        logger.info(f"Limited to {len(complexes)} complexes for testing")
    
    # Process complexes
    results = []
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for complex_info in tqdm(complexes, desc="Processing complexes"):
        result = process_complex(
            complex_info,
            extract_esm=args.esm,
            esm_device=args.esm_device,
            k_neighbors=args.k_neighbors,
            thresholds=args.thresholds,
            overwrite=args.overwrite,
        )
        results.append(result)
        
        if result.get('skipped'):
            skip_count += 1
        elif result['success']:
            success_count += 1
        else:
            error_count += 1
    
    # Summary
    logger.info("=" * 50)
    logger.info(f"Processing complete:")
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped: {skip_count}")
    logger.info(f"  Errors:  {error_count}")
    logger.info("=" * 50)
    
    # Save statistics
    if args.output_stats:
        stats = {
            'total': len(complexes),
            'success': success_count,
            'skipped': skip_count,
            'errors': error_count,
            'thresholds': args.thresholds,
            'results': results,
        }
        with open(args.output_stats, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Statistics saved to {args.output_stats}")


if __name__ == '__main__':
    main()
