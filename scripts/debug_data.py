
import os
import sys
import numpy as np
import torch
import argparse
from pathlib import Path
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pocket_dataset import PocketDataset, load_splits

def check_nan(tensor, name, pdb_id):
    if torch.isnan(tensor).any():
        print(f"[NaN] Found NaN in {name} for {pdb_id}")
        return True
    if torch.isinf(tensor).any():
        print(f"[Inf] Found Inf in {name} for {pdb_id}")
        print(f"  Range: [{tensor.min()}, {tensor.max()}]")
        return True
    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_dir', type=str, default='database/features')
    parser.add_argument('--split_file', type=str, default=None)
    args = parser.parse_args()
    
    feature_dir = Path(args.feature_dir)
    print(f"Checking features in: {feature_dir}")
    
    pdb_ids = None
    if args.split_file:
        print(f"Loading splits from: {args.split_file}")
        train_ids, val_ids, test_ids = load_splits(args.split_file)
        pdb_ids = train_ids + val_ids + test_ids
        print(f"Total samples to check: {len(pdb_ids)}")
    
    dataset = PocketDataset(
        feature_dir=str(feature_dir),
        pdb_ids=pdb_ids,
        max_seq_len=2000, 
        max_atoms=200
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    bad_samples = []
    
    # scan
    for i in tqdm(range(len(dataset))):
        try:
            sample = dataset[i]
            pdb_id = sample['pdb_id']
            
            is_bad = False
            # Check likely culprits
            if check_nan(sample['ligand_features'], 'ligand_features', pdb_id):
                is_bad = True
            elif check_nan(sample['protein_scalar'], 'protein_scalar', pdb_id):
                is_bad = True
            elif check_nan(sample['esm'], 'esm', pdb_id):
                is_bad = True
            
            if is_bad:
                bad_samples.append(pdb_id)
                
        except Exception as e:
            print(f"Error loading index {i}: {e}")
            if hasattr(dataset, 'pdb_ids'):
                print(f"PDB ID: {dataset.pdb_ids[i]}")
    
    print("\n" + "="*30)
    print("Scan Complete")
    print("="*30)
    
    if bad_samples:
        print(f"Found {len(bad_samples)} corrupted samples.")
        with open('bad_samples.txt', 'w') as f:
            for pid in bad_samples:
                f.write(f"{pid}\n")
                print(f"  - {pid}")
        print("List of corrupted PDB IDs saved to 'bad_samples.txt'")
    else:
        print("No NaN/Inf values found in the checked features.")

if __name__ == "__main__":
    main()
