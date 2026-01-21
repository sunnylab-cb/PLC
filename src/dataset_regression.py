"""
Dataset and dataloader for distance regression.
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit


class RegressionDataset(Dataset):
    def __init__(self, data_frame, log_transform=False):
        """
        Args:
            data_frame (pd.DataFrame): DataFrame containing features and distance labels.
            log_transform (bool): If True, apply log(distance + 1) transform to distances.
        """
        self.data = data_frame
        self.log_transform = log_transform
        
        # Parse features
        if isinstance(self.data.iloc[0]['features'], str):
            tqdm_available = True
            try:
                from tqdm import tqdm
            except ImportError:
                tqdm_available = False
            
            print("Parsing feature strings... this might take a while.")
            
            def parse_feature_str(s):
                s = s.replace('[', '').replace(']', '').replace('\n', ' ')
                return np.fromstring(s, sep=' ', dtype=np.float32)

            if tqdm_available:
                tqdm.pandas()
                self.features = self.data['features'].progress_apply(parse_feature_str).values
            else:
                self.features = self.data['features'].apply(parse_feature_str).values
                
            self.features = np.stack(self.features)
        else:
            self.features = np.stack(self.data['features'].values).astype(np.float32)
        
        # Distance as label (with optional log transform)
        distances = self.data['distance'].values.astype(np.float32)
        if log_transform:
            # log(distance + 1) to handle zero distances
            self.distances = np.log1p(distances)
            print(f"Applied log transform. Range: {self.distances.min():.2f} to {self.distances.max():.2f}")
        else:
            self.distances = distances

    def __len__(self):
        return len(self.distances)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.distances[idx])


def load_regression_data(data_dir, limit_files=None):
    """
    Loads all regression CSV files from the directory into a single DataFrame.
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "features_batch_*.csv")))
    if limit_files:
        all_files = all_files[:limit_files]
    print(f"Found {len(all_files)} files in {data_dir} (limit={limit_files})")
    
    df_list = []
    for filename in all_files:
        print(f"Loading {filename}...")
        df = pd.read_csv(filename)
        df_list.append(df)
    
    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"Total samples loaded: {len(full_df)}")
    
    # Print distance statistics
    distances = full_df['distance'].values
    print(f"Distance stats: min={distances.min():.2f}Å, max={distances.max():.2f}Å, "
          f"mean={distances.mean():.2f}Å, median={np.median(distances):.2f}Å")
    
    return full_df


def get_regression_dataloaders(data_dir, batch_size=64, val_ratio=0.1, test_ratio=0.1, seed=42, limit_files=None, log_transform=False):
    """
    Splits the data by PDB ID and returns train, val, test dataloaders for regression.
    
    Args:
        log_transform: If True, apply log(distance + 1) to distance targets.
    """
    df = load_regression_data(data_dir, limit_files=limit_files)
    
    # Groups
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
    
    print(f"Split sizes (by samples): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Split sizes (by PDBs): Train={train_df['pdb_id'].nunique()}, Val={val_df['pdb_id'].nunique()}, Test={test_df['pdb_id'].nunique()}")
    
    # Create Datasets (with optional log transform)
    train_dataset = RegressionDataset(train_df, log_transform=log_transform)
    val_dataset = RegressionDataset(val_df, log_transform=log_transform)
    test_dataset = RegressionDataset(test_df, log_transform=log_transform)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
