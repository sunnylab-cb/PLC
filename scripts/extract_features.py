#!/usr/bin/env python3
"""
Batch feature extraction script for PDBbind dataset.

This script uses the pypropel library to extract all features needed for the
Pocket-GVP-Fusion v2 model from protein-ligand complexes.

Usage:
    python extract_features.py --data_dir /path/to/pdbbind --output /path/to/output
    python extract_features.py --data_dir /path/to/pdbbind --output /path/to/output --device cuda

Features Extracted:
    Protein:
        - ESM-2 embeddings (1280-dim per residue)
        - GVP vector features (4 vectors of 3-dim per residue)
        - Scalar features (35-dim per residue): one-hot, SS, SASA, charge, hydro, etc.
        
    Ligand:
        - Atom features (64-dim per atom with global tag)
        - Atom coordinates
        
    Labels:
        - 3-class distance classification per residue
        - Raw distances for analysis
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm


def setup_pypropel():
    """Ensure pypropel is importable and suppress RDKit warnings."""
    # Suppress RDKit warnings
    import warnings
    warnings.filterwarnings('ignore', message='.*molecule is tagged as 2D.*')
    warnings.filterwarnings('ignore', message='.*DEPRECATION WARNING.*')
    
    try:
        from rdkit import RDLogger
        RDLogger.DisableLog('rdApp.*')
    except ImportError:
        pass
    
    try:
        import pypropel
        return True
    except ImportError:
        print("Error: pypropel not found. Please install it first.")
        print("  pip install -e /path/to/pypropel_unibio")
        return False


def find_complexes(data_dir: str, recursive: bool = True) -> List[Tuple[str, str, str]]:
    """
    Find all protein-ligand complexes in the PDBbind directory.
    
    Searches for pattern: {pdb_id}/{pdb_id}_protein.pdb + {pdb_id}_ligand.sdf/mol2
    
    Parameters
    ----------
    data_dir : str
        Root directory to search.
    recursive : bool
        If True, search all subdirectories recursively.
    
    Returns
    -------
    List of (pdb_id, protein_path, ligand_path) tuples.
    """
    data_path = Path(data_dir)
    complexes = []
    
    # Get all subdirectories to check
    if recursive:
        # Use glob to find all directories that might contain complexes
        subdirs = list(data_path.rglob('*'))
        subdirs = [d for d in subdirs if d.is_dir()]
        subdirs.append(data_path)  # Also check root
    else:
        subdirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    for subdir in subdirs:
        pdb_id = subdir.name
        
        # Skip common non-PDB directories
        if pdb_id in {'features', 'output', '__pycache__', '.git'}:
            continue
        
        protein_path = subdir / f"{pdb_id}_protein.pdb"
        ligand_sdf = subdir / f"{pdb_id}_ligand.sdf"
        ligand_mol2 = subdir / f"{pdb_id}_ligand.mol2"
        
        if protein_path.exists():
            if ligand_sdf.exists():
                complexes.append((pdb_id, str(protein_path), str(ligand_sdf)))
            elif ligand_mol2.exists():
                complexes.append((pdb_id, str(protein_path), str(ligand_mol2)))
    
    return sorted(complexes)


def extract_esm_embeddings(
    sequences: Dict[str, str],
    model_name: str = "esm2_t33_650M_UR50D",
    device: str = "cuda",
    batch_size: int = 4
) -> Dict[str, np.ndarray]:
    """
    Extract ESM-2 embeddings for all sequences.
    """
    import pypropel.esm as ppesm
    
    print(f"Loading ESM model: {model_name} on {device}...")
    model, alphabet, batch_converter = ppesm.load_esm_model(model_name, device=device)
    
    embeddings = {}
    pdb_ids = list(sequences.keys())
    seqs = [sequences[pid] for pid in pdb_ids]
    
    print(f"Extracting ESM embeddings for {len(seqs)} sequences...")
    emb_list = ppesm.get_esm_embeddings_batch(seqs, model=model, device=device, batch_size=batch_size)
    
    for pdb_id, emb in zip(pdb_ids, emb_list):
        embeddings[pdb_id] = emb
    
    return embeddings


def extract_features_single(
    pdb_id: str,
    protein_path: str,
    ligand_path: str,
    esm_embedding: Optional[np.ndarray] = None,
    use_dssp: bool = True
) -> Dict[str, np.ndarray]:
    """
    Extract all features for a single protein-ligand complex.
    """
    import pypropel.str as ppstr
    import pypropel.mol as ppmol
    import pypropel.gvp as ppgvp
    import pypropel.fpsite as fpsite
    import pypropel.dist as ppdist
    
    result = {'pdb_id': pdb_id}
    
    # Load structures
    try:
        structure = ppstr.load_pdb(protein_path)
    except Exception as e:
        print(f"  Error loading protein {protein_path}: {e}")
        return None
    
    try:
        if ligand_path.endswith('.sdf'):
            mol = ppmol.load_sdf(ligand_path)
        else:
            mol = ppmol.load_mol2(ligand_path)
    except Exception as e:
        print(f"  Error loading ligand {ligand_path}: {e}")
        return None
    
    if mol is None:
        print(f"  Failed to load ligand: {ligand_path}")
        return None
    
    # --- Protein Features ---
    
    # GVP features (4 vectors)
    try:
        gvp_coords, gvp_vectors = ppgvp.get_gvp_node_features(structure, k_neighbors=10)
        result['gvp_coords'] = gvp_coords  # (N, 3)
        result['gvp_vectors'] = gvp_vectors  # (N, 4, 3)
    except Exception as e:
        print(f"  Error extracting GVP features: {e}")
        return None
    
    n_residues = len(gvp_coords)
    
    # Scalar features (35-dim)
    try:
        scalar_features = fpsite.get_protein_scalar_features(structure, use_dssp=use_dssp)
        result['protein_scalar'] = scalar_features  # (N, 35)
    except Exception as e:
        print(f"  Warning: DSSP failed, using fallback: {e}")
        scalar_features = fpsite.get_protein_scalar_features(structure, use_dssp=False)
        result['protein_scalar'] = scalar_features
    
    # ESM embeddings (pre-computed)
    if esm_embedding is not None:
        if len(esm_embedding) == n_residues:
            result['esm'] = esm_embedding  # (N, 1280)
        else:
            print(f"  Warning: ESM length mismatch ({len(esm_embedding)} vs {n_residues})")
            result['esm'] = np.zeros((n_residues, 1280), dtype=np.float32)
    else:
        result['esm'] = np.zeros((n_residues, 1280), dtype=np.float32)
    
    # --- Ligand Features ---
    
    try:
        ligand_coords = ppmol.ligand_coords(mol)
        ligand_features = ppmol.get_atom_features_extended(mol, global_tag_dim=45)
        result['ligand_coords'] = ligand_coords  # (M, 3)
        result['ligand_features'] = ligand_features  # (M, 64)
    except Exception as e:
        print(f"  Error extracting ligand features: {e}")
        return None
    
    # --- Labels ---
    
    try:
        # 3-class: Contact (<3.5), Near (3.5-6.0), Far (>6.0)
        labels = ppdist.get_distance_labels(structure, ligand_coords, thresholds=[3.5, 6.0])
        result['labels'] = labels  # (N,)
        
        # Raw distances for analysis
        distances_df = ppdist.protein_ligand_distances(structure, ligand_coords)
        result['distances'] = distances_df['distance'].values.astype(np.float32)
    except Exception as e:
        print(f"  Error computing labels: {e}")
        return None
    
    result['n_residues'] = n_residues
    result['n_ligand_atoms'] = len(ligand_coords)
    
    return result


def save_features(features: Dict, output_path: str):
    """Save features to a compressed numpy file."""
    np.savez_compressed(output_path, **features)


def main():
    parser = argparse.ArgumentParser(
        description='Extract features for PDBbind dataset using pypropel',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--data_dir', required=True, 
                       help='Path to PDBbind directory with subdirs per complex')
    parser.add_argument('--output', default=None,
                       help='Output directory (default: data_dir/features)')
    parser.add_argument('--esm_model', default='esm2_t33_650M_UR50D',
                       help='ESM model name')
    parser.add_argument('--device', default='cuda',
                       help='Device for ESM inference (cuda/cpu)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for ESM inference')
    parser.add_argument('--no_esm', action='store_true',
                       help='Skip ESM embedding extraction')
    parser.add_argument('--no_dssp', action='store_true',
                       help='Skip DSSP feature extraction')
    parser.add_argument('--max_complexes', type=int, default=None,
                       help='Maximum number of complexes to process')
    parser.add_argument('--resume', action='store_true',
                       help='Skip already processed complexes')
    
    args = parser.parse_args()
    
    # Setup
    if not setup_pypropel():
        sys.exit(1)
    
    import pypropel.str as ppstr
    
    output_dir = args.output or os.path.join(args.data_dir, 'features')
    os.makedirs(output_dir, exist_ok=True)
    
    # Find complexes
    print(f"Scanning {args.data_dir} for complexes...")
    complexes = find_complexes(args.data_dir)
    print(f"Found {len(complexes)} complexes")
    
    if args.max_complexes:
        complexes = complexes[:args.max_complexes]
        print(f"Processing first {len(complexes)} complexes")
    
    # Filter already processed if resuming
    if args.resume:
        remaining = []
        for pdb_id, prot_path, lig_path in complexes:
            out_path = os.path.join(output_dir, f"{pdb_id}_features.npz")
            if not os.path.exists(out_path):
                remaining.append((pdb_id, prot_path, lig_path))
        print(f"Resuming: {len(remaining)} complexes remaining")
        complexes = remaining
    
    if len(complexes) == 0:
        print("No complexes to process")
        return
    
    # Extract sequences for ESM
    sequences = {}
    if not args.no_esm:
        print("Extracting sequences for ESM...")
        for pdb_id, prot_path, _ in tqdm(complexes, desc="Reading sequences"):
            try:
                seq = ppstr.read_pdb_sequence(prot_path)
                if seq:
                    sequences[pdb_id] = seq
            except Exception as e:
                print(f"  Error reading sequence from {pdb_id}: {e}")
    
    # Extract ESM embeddings in batch
    esm_embeddings = {}
    if sequences and not args.no_esm:
        try:
            esm_embeddings = extract_esm_embeddings(
                sequences, 
                model_name=args.esm_model,
                device=args.device,
                batch_size=args.batch_size
            )
        except Exception as e:
            print(f"ESM extraction failed: {e}")
            print("Continuing without ESM embeddings...")
    
    # Extract features for each complex
    success_count = 0
    fail_count = 0
    
    for pdb_id, prot_path, lig_path in tqdm(complexes, desc="Extracting features"):
        out_path = os.path.join(output_dir, f"{pdb_id}_features.npz")
        
        esm_emb = esm_embeddings.get(pdb_id, None)
        
        features = extract_features_single(
            pdb_id, prot_path, lig_path,
            esm_embedding=esm_emb,
            use_dssp=not args.no_dssp
        )
        
        if features is not None:
            save_features(features, out_path)
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nDone! Processed {success_count} complexes successfully, {fail_count} failed.")
    print(f"Features saved to: {output_dir}")


if __name__ == '__main__':
    main()
