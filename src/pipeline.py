import os
import argparse
import numpy as np
import pandas as pd
from Bio.PDB import Polypeptide
import utils
import features

def process_complex(pdb_id, pdbbind_dir):
    """
    Process a single PDBbind complex.
    pdbbind_dir: Root directory of PDBbind (e.g., .../P-L)
    """
    # Locate files
    # Structure: PDBbind usually organizes by year or directly by PDB ID in older versions
    # We'll assume the provided structure: .../P-L/<YearRange>/<PDBID>/... 
    # OR directly .../P-L/<PDBID>/ if we passed the specific parent path.
    # To be robust, let's assume `pdbbind_dir` is the parent containing the PDBID folder.
    
    complex_dir = os.path.join(pdbbind_dir, pdb_id)
    pdb_file = os.path.join(complex_dir, f"{pdb_id}_protein.pdb")
    ligand_sdf = os.path.join(complex_dir, f"{pdb_id}_ligand.sdf")
    ligand_mol2 = os.path.join(complex_dir, f"{pdb_id}_ligand.mol2")
    
    # Load Structure
    if not os.path.exists(pdb_file):
        # Fallback to .ent or other names if necessary
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
    ligand_feat_vec = features.get_ligand_features(mol) # 4 dims
    
    # Get Global Protein Attributes
    full_sequence = utils.get_sequence_from_structure(structure)
    if not full_sequence:
        print(f"Skipping {pdb_id}: No sequence found")
        return None
        
    global_prot_feat = features.get_global_protein_features(full_sequence) # 20 dims
    
    # Iterate Residues
    dataset_rows = []
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if not Polypeptide.is_aa(residue, standard=True):
                    continue
                
                res_name = residue.get_resname()
                
                # Residue Features
                # 1. One Hot
                res_onehot = features.get_residue_one_hot(res_name) # 20 dims
                # 2. PhysChem
                res_phys = features.get_residue_physchem(res_name) # 2 dims
                
                # Distance Label
                res_coords = utils.get_atom_coords(residue)
                min_dist = utils.calculate_min_distance(res_coords, ligand_coords)
                
                label = 0
                if min_dist < 5.0:
                    label = 1
                elif min_dist > 8.0:
                    label = 0
                else:
                    # Optional: Ignore margin
                    continue
                
                # Construct Feature Vector
                # [Residue(22), Global(20), Ligand(4)] = 46 dims
                feature_vector = np.concatenate([res_onehot, res_phys, global_prot_feat, ligand_feat_vec])
                
                row = {
                    'pdb_id': pdb_id,
                    'chain': chain.get_id(),
                    'res_id': residue.get_id()[1],
                    'res_name': res_name,
                    'features': feature_vector, # Alternatively store as separate columns
                    'label': label
                }
                dataset_rows.append(row)
                
    return dataset_rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdb-id', type=str, help='Single PDB ID to process')
    parser.add_argument('--data-dir', type=str, required=True, help='Path to PDBbind P-L directory')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Output directory')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    pdb_list = []
    if args.pdb_id:
        # Find the specific PDB folder
        # This is a bit tricky if we don't know the exact year structure.
        # We recursively search for it.
        for root, dirs, files in os.walk(args.data_dir):
            if args.pdb_id in dirs:
                pdb_list.append((args.pdb_id, root))
                break
    else:
        # Walk and find all potential PDB folders
        # PDBbind structure: Year/PDBID
        # PDBbind folders usually name the folder as the PDB ID
        print(f"Scanning {args.data_dir} for PDBs...")
        for root, dirs, files in os.walk(args.data_dir):
            for d in dirs:
                # Basic check: PDB IDs are 4 chars
                if len(d) == 4 and d.isalnum():
                     pdb_list.append((d, root)) # Store ID and parent path to locate it quickly if needed, or just ID.
                     # Actually process_complex assumes args.data_dir is root. 
                     # But process_complex needs the PARENT of the pdb_id folder.
                     # Let's adjust logic.
    
    print(f"Found {len(pdb_list)} complexes.")
    
    batch_size = 100
    batch_data = []
    total_processed = 0
    
    # Header for CSV
    sample_header = ['pdb_id', 'chain', 'res_id', 'res_name', 'features', 'label']
    
    # Clean/Init output file if combining, or we write separate files. 
    # Writing one huge file might be risky if script crashes.
    # Let's write batch files: batch_0.csv, batch_1.csv...
    
    batch_idx = 0
    
    for i, (pdb_id, parent_path) in enumerate(pdb_list):
        # We need to pass the parent path to process_complex so it finds parent/pdb_id
        # process_complex arg name is 'pdbbind_dir' which effectively is parent_path
        try:
            data = process_complex(pdb_id, parent_path)
            if data:
                batch_data.extend(data)
        except Exception as e:
            print(f"Error processing {pdb_id}: {e}")

        if len(batch_data) >= 5000: # Flush every 5000 residues or so (approx 15-20 proteins)
            # Actually residues are ~200-500 per protein. 10 PDBs ~ 4000 rows.
            # Let's flush by PDB count or just raw size.
            pass
            
        if (i + 1) % batch_size == 0 or (i + 1) == len(pdb_list):
             if batch_data:
                 df = pd.DataFrame(batch_data)
                 # Ensure proper column order
                 df = df[sample_header]
                 out_file = os.path.join(args.output_dir, f"features_batch_{batch_idx}.csv")
                 df.to_csv(out_file, index=False)
                 print(f"Saved batch {batch_idx} with {len(batch_data)} rows.")
                 batch_data = []
                 batch_idx += 1
    
    # Flush remaining
    if batch_data:
         df = pd.DataFrame(batch_data)
         df = df[sample_header]
         out_file = os.path.join(args.output_dir, f"features_batch_{batch_idx}.csv")
         df.to_csv(out_file, index=False)
         print(f"Saved final batch {batch_idx}.")

if __name__ == "__main__":
    main()
