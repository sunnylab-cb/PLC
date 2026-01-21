"""
Pipeline for extracting distance-based features for regression task.
Stores actual distance (in Angstroms) instead of binary labels.
"""

import os
import argparse
import numpy as np
import pandas as pd
import pypropel.fpsite as fpsite
import utils
import features


def process_complex_regression(pdb_id, pdbbind_dir, max_distance=30.0):
    """
    Process a single PDBbind complex for regression.
    Returns actual distance instead of binary label.
    
    Args:
        pdb_id: PDB identifier
        pdbbind_dir: Root directory containing the PDB folder
        max_distance: Maximum distance to record (clip values above this)
    """
    complex_dir = os.path.join(pdbbind_dir, pdb_id)
    pdb_file = os.path.join(complex_dir, f"{pdb_id}_protein.pdb")
    ligand_sdf = os.path.join(complex_dir, f"{pdb_id}_ligand.sdf")
    ligand_mol2 = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")
    
    # Load Structure
    if not os.path.exists(pdb_file):
        return None
    
    structure = utils.load_structure(pdb_file)
    if structure is None:
        return None
        
    # Load Ligand
    mol = None
    if os.path.exists(ligand_mol2):
        mol = utils.load_ligand_mol2(ligand_mol2)
    elif os.path.exists(ligand_sdf):
        mol = utils.load_ligand_sdf(ligand_sdf)
        
    if mol is None:
        print(f"Skipping {pdb_id}: Ligand not found or unparseable")
        return None

    # Get Ligand Attributes
    ligand_coords = utils.get_ligand_coords(mol)
    ligand_feat_vec = features.get_ligand_features(mol)  # 4 dims
    
    # Get Global Protein Attributes
    full_sequence = utils.get_sequence_from_structure(structure)
    if not full_sequence:
        print(f"Skipping {pdb_id}: No sequence found")
        return None
        
    global_prot_feat = features.get_global_protein_features(full_sequence)  # 440 dims
    
    # Iterate Residues - keep ALL residues (no margin filtering)
    dataset_rows = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if not fpsite.is_standard_aa(residue):
                    continue
                
                res_name = residue.get_resname()
                
                # Residue Features
                res_onehot = features.get_residue_one_hot(res_name)  # 20 dims
                res_phys = features.get_residue_physchem(res_name)  # 2 dims
                
                # Calculate actual distance
                res_coords = utils.get_atom_coords(residue)
                min_dist = utils.calculate_min_distance(res_coords, ligand_coords)
                
                # Clip distance to max_distance
                min_dist = min(min_dist, max_distance)
                
                # Construct Feature Vector (466 dims total)
                # [Residue(22), Global(440), Ligand(4)]
                feature_vector = np.concatenate([res_onehot, res_phys, global_prot_feat, ligand_feat_vec])
                
                row = {
                    'pdb_id': pdb_id,
                    'chain': chain.get_id(),
                    'res_id': residue.get_id()[1],
                    'res_name': res_name,
                    'features': feature_vector,
                    'distance': min_dist  # Actual distance in Angstroms
                }
                dataset_rows.append(row)
                
    return dataset_rows


def main():
    parser = argparse.ArgumentParser(description="Extract distance regression features from PDBbind")
    parser.add_argument('--pdb-id', type=str, help='Single PDB ID to process')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to PDBbind P-L directory')
    parser.add_argument('--output-dir', type=str, default='data/processed_regression', help='Output directory')
    parser.add_argument('--max-distance', type=float, default=30.0, help='Maximum distance to record')
    parser.add_argument('--batch-size', type=int, default=100, help='Number of PDBs per batch file')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find PDB folders
    pdb_list = []
    if args.pdb_id:
        for root, dirs, files in os.walk(args.data_dir):
            if args.pdb_id in dirs:
                pdb_list.append((args.pdb_id, root))
                break
    else:
        print(f"Scanning {args.data_dir} for PDBs...")
        for root, dirs, files in os.walk(args.data_dir):
            for d in dirs:
                if len(d) == 4 and d.isalnum():
                    pdb_list.append((d, root))
    
    print(f"Found {len(pdb_list)} complexes.")
    
    batch_data = []
    sample_header = ['pdb_id', 'chain', 'res_id', 'res_name', 'features', 'distance']
    batch_idx = 0
    
    for i, (pdb_id, parent_path) in enumerate(pdb_list):
        try:
            data = process_complex_regression(pdb_id, parent_path, max_distance=args.max_distance)
            if data:
                batch_data.extend(data)
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")
        
        # Progress
        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1}/{len(pdb_list)} complexes...")
        
        if (i + 1) % args.batch_size == 0 or (i + 1) == len(pdb_list):
            if batch_data:
                df = pd.DataFrame(batch_data)
                df = df[sample_header]
                out_file = os.path.join(args.output_dir, f"features_batch_{batch_idx}.csv")
                df.to_csv(out_file, index=False)
                print(f"Saved batch {batch_idx} with {len(batch_data)} rows to {out_file}")
                batch_data = []
                batch_idx += 1
    
    # Create README
    readme_content = """# PDBbind Processed Dataset (Regression)

This directory contains features with **distance labels** (in Angstroms) for regression.

## File Format
Files are named `features_batch_X.csv`.

## Columns
- `pdb_id`: PDB identifier
- `chain`: Chain ID
- `res_id`: Residue number
- `res_name`: Residue name (3-letter code)
- `features`: 466-dimensional feature vector
- `distance`: Minimum distance to ligand (in Angstroms, clipped at 30Å)

## Feature Vector Description (466 Dimensions)
| Index Range | Feature Group | Description | Dimensions |
| :--- | :--- | :--- | :--- |
| **0 - 19** | Residue Identity | One-Hot Encoding | 20 |
| **20 - 21** | Residue PhysChem | Charge and Hydrophobicity | 2 |
| **22 - 461** | Global Protein | AAC, DAC, AVEANF | 440 |
| **462 - 465** | Ligand Features | LogP, MW, H-Donors, H-Acceptors | 4 |
"""
    with open(os.path.join(args.output_dir, 'README.md'), 'w') as f:
        f.write(readme_content)
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
