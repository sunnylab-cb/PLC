import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import ast

def parse_feature_string(feature_str):
    """
    Parses the features column which might be stored as a string representation of a list/array.
    Handles space-separated brackets or comma-separated formats.
    """
    try:
        # If it's a simple string like "[0.1, 0.2]", ast.literal_eval handles it.
        # If it's space separated "[0.1 0.2]", we need to fix it.
        feature_str = feature_str.strip()
        if feature_str.startswith('[') and feature_str.endswith(']'):
            content = feature_str[1:-1].strip()
            if ',' not in content and ' ' in content:
                 # Space separated
                 return np.fromstring(content, sep=' ')
            else:
                 # Comma separated or other python list format
                 return np.array(ast.literal_eval(feature_str), dtype=np.float32)
        else:
             # Fallback
             return np.fromstring(feature_str, sep=' ')
    except Exception as e:
        print(f"Error parsing feature string: {feature_str[:50]}... Error: {e}")
        return np.zeros(1) # Return dummy to avoid crash, will be filtered later

def load_data(data_dir, num_batches=None):
    """
    Loads features_batch_*.csv files from the directory.
    If num_batches is specified, only loads that many files (sorted by name).
    """
    all_files = sorted(glob.glob(os.path.join(data_dir, "features_batch_*.csv")))
    if not all_files:
        raise ValueError(f"No feature files found in {data_dir}")
    
    if num_batches is not None:
        print(f"Limiting to first {num_batches} batches.")
        all_files = all_files[:num_batches]
    
    print(f"Found {len(all_files)} feature files to load.")
    
    df_list = []
    for f in all_files:
        print(f"Loading {f}...")
        try:
            df_chunk = pd.read_csv(f)
            df_list.append(df_chunk)
        except Exception as e:
            print(f"Failed to load {f}: {e}")
            
    if not df_list:
        return pd.DataFrame()
        
    full_df = pd.concat(df_list, ignore_index=True)
    
    print("Parsing feature columns...")
    # Apply parsing
    # We assume 'features' column exists based on README
    try:
        # Check if first element is already a list/array (some pandas versions/formats might invoke automatic conversion)
        if isinstance(full_df['features'].iloc[0], str):
            # It's manual optimized parsing. 
            # Vectorized approach or applying lambda might be slow for huge data, 
            # but for this scale it's likely okay. 
            # To speed up, we can try to use text processing if strictly formatted, but safest is apply.
            full_df['X'] = full_df['features'].apply(parse_feature_string)
        else:
             full_df['X'] = full_df['features']

        # Convert column of arrays to 2D numpy array
        # Stack them
        X = np.stack(full_df['X'].values)
        y = full_df['label'].values
        groups = full_df['pdb_id'].values
        
        return X, y, groups, full_df
        
    except KeyError:
        print("Column 'features' or 'label' or 'pdb_id' not found.")
        return None, None, None, None

def train_rf(data_dir, output_model_path="rf_model.joblib", n_estimators=200, max_depth=None, num_batches=None):
    print("Loading data...")
    X, y, groups, df = load_data(data_dir, num_batches=num_batches)
    
    if X is None:
        print("Data loading failed.")
        return

    print(f"Data loaded. Shape: X={X.shape}, y={y.shape}, unique groups={len(np.unique(groups))}")
    
    # Split based on groups (pdb_id)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"Train/Test split: Train={len(X_train)}, Test={len(X_test)}")
    print(f"Class distribution in Train: {np.bincount(y_train.astype(int))}")
    print(f"Class distribution in Test: {np.bincount(y_test.astype(int))}")

    # Initialize Model with "appropriate" parameters
    # Balanced class weight is critical for binding site prediction
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight='balanced',
        n_jobs=-1,
        random_state=42,
        oob_score=True
    )
    
    print("Training Random Forest...")
    clf.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print(f"Saving model to {output_model_path}")
    joblib.dump(clf, output_model_path)
    
    # Also return the model and metrics for potential in-memory use
    return clf, {'accuracy': acc, 'roc_auc': auc}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Path to processed data")
    parser.add_argument("--save_path", type=str, default="rf_model.joblib", help="Path to save trained model")
    parser.add_argument("--num_batches", type=int, default=None, help="Number of batches to use (for comparison)")
    args = parser.parse_args()
    
    train_rf(args.data_dir, args.save_path, num_batches=args.num_batches)
