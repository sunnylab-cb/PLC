"""
Dataset and data loading for Pocket-GVP-Fusion v2.

Loads pre-extracted features from .npz files and creates training batches.
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_splits(
    feature_dir: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create random train/val/test splits from feature files.
    
    Args:
        feature_dir: Directory containing *_features.npz files
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_ids, val_ids, test_ids) - lists of PDB IDs
    """
    set_seed(seed)
    
    # Find all feature files
    feature_path = Path(feature_dir)
    npz_files = list(feature_path.glob("*_features.npz"))
    
    # Extract PDB IDs
    pdb_ids = [f.stem.replace("_features", "") for f in npz_files]
    
    # Shuffle
    random.shuffle(pdb_ids)
    
    # Split
    n = len(pdb_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    
    train_ids = pdb_ids[:n_train]
    val_ids = pdb_ids[n_train:n_train + n_val]
    test_ids = pdb_ids[n_train + n_val:]
    
    return train_ids, val_ids, test_ids


def save_splits(
    train_ids: List[str],
    val_ids: List[str],
    test_ids: List[str],
    output_path: str
):
    """Save data splits to a file."""
    with open(output_path, 'w') as f:
        f.write("# Train/Val/Test splits\n")
        f.write(f"# Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}\n\n")
        
        f.write("# TRAIN\n")
        for pdb_id in train_ids:
            f.write(f"train\t{pdb_id}\n")
        
        f.write("\n# VAL\n")
        for pdb_id in val_ids:
            f.write(f"val\t{pdb_id}\n")
        
        f.write("\n# TEST\n")
        for pdb_id in test_ids:
            f.write(f"test\t{pdb_id}\n")


def load_splits(split_path: str) -> Tuple[List[str], List[str], List[str]]:
    """Load data splits from a file."""
    train_ids, val_ids, test_ids = [], [], []
    
    with open(split_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split('\t')
            if len(parts) == 2:
                split, pdb_id = parts
                if split == 'train':
                    train_ids.append(pdb_id)
                elif split == 'val':
                    val_ids.append(pdb_id)
                elif split == 'test':
                    test_ids.append(pdb_id)
    
    return train_ids, val_ids, test_ids


class PocketDataset(Dataset):
    """
    Dataset for loading pre-extracted protein-ligand features.
    
    Expects .npz files with:
        - esm: [N, 1280] ESM embeddings
        - protein_scalar: [N, 35] physicochemical features
        - gvp_coords: [N, 3] Cα coordinates
        - gvp_vectors: [N, 4, 3] GVP vector features
        - ligand_features: [M, 64] ligand atom features
        - ligand_coords: [M, 3] ligand atom coordinates
        - labels: [N] 3-class labels
        - distances: [N] raw distances
        - n_residues: int
        - n_ligand_atoms: int
        
    Args:
        feature_dir: Directory containing *_features.npz files
        pdb_ids: List of PDB IDs to include
        max_seq_len: Maximum protein sequence length (for padding)
        max_atoms: Maximum ligand atoms (for padding)
    """
    
    def __init__(
        self,
        feature_dir: str,
        pdb_ids: Optional[List[str]] = None,
        max_seq_len: int = 1000,
        max_atoms: int = 100
    ):
        super().__init__()
        
        self.feature_dir = Path(feature_dir)
        self.max_seq_len = max_seq_len
        self.max_atoms = max_atoms
        
        # Get list of available files
        if pdb_ids is not None:
            self.pdb_ids = pdb_ids
        else:
            npz_files = list(self.feature_dir.glob("*_features.npz"))
            self.pdb_ids = [f.stem.replace("_features", "") for f in npz_files]
        
        # Filter to existing files
        self.pdb_ids = [
            pid for pid in self.pdb_ids 
            if (self.feature_dir / f"{pid}_features.npz").exists()
        ]
        
    def __len__(self) -> int:
        return len(self.pdb_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pdb_id = self.pdb_ids[idx]
        npz_path = self.feature_dir / f"{pdb_id}_features.npz"
        
        # Load features
        data = np.load(npz_path, allow_pickle=True)
        
        # Extract arrays
        esm = data.get('esm', np.zeros((1, 1280), dtype=np.float32))
        protein_scalar = data.get('protein_scalar', np.zeros((1, 35), dtype=np.float32))
        gvp_coords = data.get('gvp_coords', np.zeros((1, 3), dtype=np.float32))
        gvp_vectors = data.get('gvp_vectors', np.zeros((1, 4, 3), dtype=np.float32))
        ligand_features = data.get('ligand_features', np.zeros((1, 64), dtype=np.float32))
        ligand_coords = data.get('ligand_coords', np.zeros((1, 3), dtype=np.float32))
        labels = data.get('labels', np.zeros(1, dtype=np.int64))
        
        n_residues = int(data.get('n_residues', len(esm)))
        n_atoms = int(data.get('n_ligand_atoms', len(ligand_features)))
        
        # Ensure consistent lengths
        n_residues = min(n_residues, len(esm), len(protein_scalar))
        
        # Truncate if too long
        if n_residues > self.max_seq_len:
            esm = esm[:self.max_seq_len]
            protein_scalar = protein_scalar[:self.max_seq_len]
            gvp_coords = gvp_coords[:self.max_seq_len]
            gvp_vectors = gvp_vectors[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            n_residues = self.max_seq_len
        
        if n_atoms > self.max_atoms:
            ligand_features = ligand_features[:self.max_atoms]
            ligand_coords = ligand_coords[:self.max_atoms]
            n_atoms = self.max_atoms
        
        # Convert to tensors
        return {
            'pdb_id': pdb_id,
            'esm': torch.from_numpy(esm.astype(np.float32)),
            'protein_scalar': torch.from_numpy(protein_scalar.astype(np.float32)),
            'gvp_coords': torch.from_numpy(gvp_coords.astype(np.float32)),
            'gvp_vectors': torch.from_numpy(gvp_vectors.astype(np.float32)),
            'ligand_features': torch.from_numpy(ligand_features.astype(np.float32)),
            'ligand_coords': torch.from_numpy(ligand_coords.astype(np.float32)),
            'labels': torch.from_numpy(labels.astype(np.int64)),
            'n_residues': n_residues,
            'n_atoms': n_atoms
        }


def create_dataloader(
    feature_dir: str,
    pdb_ids: List[str],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    max_seq_len: int = 1000,
    max_atoms: int = 100,
    seed: int = 42
) -> DataLoader:
    """
    Create a DataLoader for training/evaluation.
    
    Args:
        feature_dir: Directory containing features
        pdb_ids: List of PDB IDs to include
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        max_seq_len: Maximum protein length
        max_atoms: Maximum ligand atoms
        seed: Random seed
        
    Returns:
        DataLoader instance
    """
    from .collate import pocket_collate_fn
    
    set_seed(seed)
    
    dataset = PocketDataset(
        feature_dir=feature_dir,
        pdb_ids=pdb_ids,
        max_seq_len=max_seq_len,
        max_atoms=max_atoms
    )
    
    # Create generator with fixed seed for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pocket_collate_fn,
        pin_memory=True,
        generator=generator if shuffle else None
    )
    
    return dataloader
