import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit

class BindingDataset(Dataset):
    def __init__(self, data_frame):
        """
        Args:
            data_frame (pd.DataFrame): DataFrame containing features and labels.
                Optionally contains 'relative_position' column to append.
        """
        self.data = data_frame
        # Convert features from string/list to float32 tensor
        # Assuming 'features' column is already parsed into numpy arrays or needs parsing
        # If it's a string from CSV: "[ 0.1  0.2 ...]"
        if isinstance(self.data.iloc[0]['features'], str):
            # Slow parsing if done item by item. 
            # Better to do it in preprocessing or vectorised.
            # But for simplicity/robustness now, let's try to do it efficiently.
            
            # Remove brackets and newlines
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
            # Assume it's already a list or array
            self.features = np.stack(self.data['features'].values).astype(np.float32)
        
        # Append relative position if available (making 466 -> 467 dims)
        if 'relative_position' in self.data.columns:
            rel_pos = self.data['relative_position'].values.astype(np.float32).reshape(-1, 1)
            self.features = np.concatenate([self.features, rel_pos], axis=1)
            print(f"Appended relative_position feature. New shape: {self.features.shape}")
            
        self.labels = self.data['label'].values.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])

def load_data(data_dir, limit_files=None, add_relative_position=True):
    """
    Loads all CSV files from the directory into a single DataFrame.
    Optionally adds relative position feature (0=N-terminus, 1=C-terminus).
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "features_batch_*.csv")))
    if limit_files:
        all_files = all_files[:limit_files]
    print(f"Found {len(all_files)} files in {data_dir} (limit={limit_files})")
    
    df_list = []
    # Read only necessary columns to save memory if needed, but we need most of them.
    # We definitely need 'pdb_id' for splitting, 'features', and 'label'.
    for filename in all_files:
        print(f"Loading {filename}...")
        df = pd.read_csv(filename)
        df_list.append(df)
    
    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"Total samples loaded: {len(full_df)}")
    
    if add_relative_position:
        print("Computing relative position feature...")
        full_df = _add_relative_position(full_df)
    
    return full_df


def _add_relative_position(df):
    """
    Computes relative position for each residue within its protein.
    Position is normalized: 0.0 = N-terminus, 1.0 = C-terminus.
    
    Uses res_id as the position indicator within each pdb_id group.
    """
    # Group by pdb_id and compute relative position
    def compute_rel_pos(group):
        res_ids = group['res_id'].values
        min_id = res_ids.min()
        max_id = res_ids.max()
        
        if max_id == min_id:
            # Single residue protein - set to 0.5 (middle)
            group['relative_position'] = 0.5
        else:
            group['relative_position'] = (res_ids - min_id) / (max_id - min_id)
        
        return group
    
    df = df.groupby('pdb_id', group_keys=False).apply(compute_rel_pos)
    return df


def get_dataloaders(data_dir, batch_size=64, val_ratio=0.1, test_ratio=0.1, seed=42, limit_files=None, add_relative_position=True):
    """
    Splits the data by PDB ID and returns train, val, test dataloaders.
    """
    df = load_data(data_dir, limit_files=limit_files, add_relative_position=add_relative_position)
    
    # Groups
    groups = df['pdb_id'].values
    
    # First split: Train vs (Val + Test)
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=(val_ratio + test_ratio), random_state=seed)
    train_idx, temp_idx = next(gss_outer.split(df, groups=groups))
    
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]
    
    # Second split: Val vs Test
    # Adjust test_size relative to the temp set
    # temp_size = val + test. We want test / (val + test) portion.
    relative_test_size = test_ratio / (val_ratio + test_ratio)
    
    groups_temp = temp_df['pdb_id'].values
    gss_inner = GroupShuffleSplit(n_splits=1, test_size=relative_test_size, random_state=seed)
    val_idx, test_idx = next(gss_inner.split(temp_df, groups=groups_temp))
    
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]
    
    print(f"Split sizes (by samples): Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"Split sizes (by PDBs): Train={train_df['pdb_id'].nunique()}, Val={val_df['pdb_id'].nunique()}, Test={test_df['pdb_id'].nunique()}")
    
    # Create Datasets
    train_dataset = BindingDataset(train_df)
    val_dataset = BindingDataset(val_df)
    test_dataset = BindingDataset(test_df)
    
    # Create DataLoaders
    # Use num_workers=0 to avoid multiprocessing issues in some envs, adjust if slow
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
