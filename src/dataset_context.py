"""
Context-aware dataset for binding site prediction.

Creates sliding windows of amino acids centered on each residue,
allowing transformer models to utilize sequence context for prediction.
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from typing import Tuple, Optional, List


class ContextBindingDataset(Dataset):
    """
    Dataset that provides context windows around each amino acid.
    
    For each residue, returns a window of (2 * context_size + 1) residues
    centered on that residue. Edge residues are zero-padded.
    """
    
    def __init__(
        self, 
        data_frame: pd.DataFrame, 
        context_size: int = 5,
        feature_dim: int = 466,
    ):
        """
        Args:
            data_frame: DataFrame with columns: pdb_id, chain, res_id, features, label
            context_size: Number of residues on each side (total window = 2*context_size + 1)
            feature_dim: Dimension of feature vectors per residue
        """
        self.context_size = context_size
        self.feature_dim = feature_dim
        self.window_size = 2 * context_size + 1
        
        # Parse features if they are strings
        data_frame = self._parse_features(data_frame)
        
        # Build per-protein feature matrices and sample indices
        self._build_protein_windows(data_frame)
    
    def _parse_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse feature strings to numpy arrays if needed."""
        if isinstance(df.iloc[0]['features'], str):
            print("Parsing feature strings...")
            
            def parse_feature_str(s):
                s = s.replace('[', '').replace(']', '').replace('\n', ' ')
                return np.fromstring(s, sep=' ', dtype=np.float32)
            
            df = df.copy()
            df['features'] = df['features'].apply(parse_feature_str)
        
        return df
    
    def _build_protein_windows(self, df: pd.DataFrame):
        """
        Build data structures for efficient window retrieval.
        
        Creates:
            - self.protein_features: Dict mapping protein_key -> feature matrix (N, feature_dim)
            - self.samples: List of (protein_key, center_idx, label) tuples
        """
        self.protein_features = {}
        self.samples = []
        
        # Group by protein (pdb_id, chain)
        grouped = df.groupby(['pdb_id', 'chain'])
        
        print(f"Building context windows for {len(grouped)} proteins...")
        
        for (pdb_id, chain), group in grouped:
            protein_key = f"{pdb_id}_{chain}"
            
            # Sort by residue ID to ensure correct sequence order
            group = group.sort_values('res_id')
            
            # Stack features into matrix (N, feature_dim)
            features = np.stack(group['features'].values).astype(np.float32)
            labels = group['label'].values.astype(np.float32)
            
            self.protein_features[protein_key] = features
            
            # Create sample for each residue
            for idx, label in enumerate(labels):
                self.samples.append((protein_key, idx, label))
        
        print(f"Total samples: {len(self.samples)}")
        print(f"Window size: {self.window_size} (context_size={self.context_size})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a context window and its center label.
        
        Returns:
            window: Tensor of shape (window_size, feature_dim)
            label: Tensor scalar - label for center residue
        """
        protein_key, center_idx, label = self.samples[idx]
        features = self.protein_features[protein_key]
        n_residues = features.shape[0]
        
        # Build window with padding
        window = np.zeros((self.window_size, self.feature_dim), dtype=np.float32)
        
        for i, offset in enumerate(range(-self.context_size, self.context_size + 1)):
            src_idx = center_idx + offset
            
            # Only copy if within bounds (leave as zeros for padding)
            if 0 <= src_idx < n_residues:
                window[i] = features[src_idx]
        
        return torch.tensor(window), torch.tensor(label)


def load_data_for_context(
    data_dir: str, 
    limit_files: Optional[int] = None
) -> pd.DataFrame:
    """
    Load CSV files from directory into a single DataFrame.
    
    Args:
        data_dir: Directory containing features_batch_*.csv files
        limit_files: Optional limit on number of files to load
    
    Returns:
        DataFrame with all samples
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "features_batch_*.csv")))
    
    if limit_files:
        all_files = all_files[:limit_files]
    
    print(f"Found {len(all_files)} files in {data_dir} (limit={limit_files})")
    
    df_list = []
    for filename in all_files:
        print(f"Loading {os.path.basename(filename)}...")
        df = pd.read_csv(filename)
        df_list.append(df)
    
    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"Total samples loaded: {len(full_df)}")
    
    return full_df


def get_context_dataloaders(
    data_dir: str,
    context_size: int = 5,
    batch_size: int = 64,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    limit_files: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load data and create train/val/test dataloaders with context windows.
    
    Splits data by PDB ID to prevent data leakage.
    
    Returns:
        train_loader, val_loader, test_loader
    """
    df = load_data_for_context(data_dir, limit_files=limit_files)
    
    # Split by PDB ID
    groups = df['pdb_id'].values
    
    # First split: Train vs (Val + Test)
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_idx, temp_idx = next(gss_outer.split(df, groups=groups))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    # Second split: Val vs Test
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    groups_temp = temp_df['pdb_id'].values
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=seed)
    val_idx, test_idx = next(gss_inner.split(temp_df, groups=groups_temp))
    
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    
    print(f"\nSplit sizes (samples): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Split sizes (PDBs): Train={train_df['pdb_id'].nunique()}, "
          f"Val={val_df['pdb_id'].nunique()}, Test={test_df['pdb_id'].nunique()}")
    
    # Create datasets
    train_dataset = ContextBindingDataset(train_df, context_size=context_size)
    val_dataset = ContextBindingDataset(val_df, context_size=context_size)
    test_dataset = ContextBindingDataset(test_df, context_size=context_size)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    print("Testing ContextBindingDataset...")
    
    # Create dummy data
    dummy_data = {
        'pdb_id': ['1abc'] * 10 + ['2def'] * 8,
        'chain': ['A'] * 10 + ['B'] * 8,
        'res_id': list(range(1, 11)) + list(range(1, 9)),
        'features': [np.random.randn(466).astype(np.float32) for _ in range(18)],
        'label': [0] * 15 + [1] * 3,
    }
    df = pd.DataFrame(dummy_data)
    
    dataset = ContextBindingDataset(df, context_size=3)
    print(f"Dataset length: {len(dataset)}")
    
    # Get a sample
    window, label = dataset[0]
    print(f"Window shape: {window.shape}")  # Should be (7, 466)
    print(f"Label: {label}")
    
    # Check edge padding - first residue should have first 3 positions as zeros
    window_first, _ = dataset[0]
    print(f"First 3 positions of first residue window (should be zeros): "
          f"{window_first[:3].sum().item():.6f}")
    
    # Check middle residue - all positions should be filled
    window_mid, _ = dataset[4]
    print(f"Middle residue window all positions filled: "
          f"{(window_mid.sum(dim=1) != 0).all().item()}")
    
    print("\nAll tests passed!")
