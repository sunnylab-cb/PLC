
import os
import argparse
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser(description="Cleanup corrupted feature files")
    parser.add_argument('--feature_dir', type=str, required=True, help="Directory containing .npz files")
    parser.add_argument('--bad_samples_file', type=str, default='bad_samples.txt', help="File containing list of bad PDB IDs")
    parser.add_argument('--backup_dir', type=str, default='database/corrupted_features', help="Where to move bad files")
    args = parser.parse_args()

    feature_dir = Path(args.feature_dir)
    backup_dir = Path(args.backup_dir)
    
    if not os.path.exists(args.bad_samples_file):
        print(f"Error: {args.bad_samples_file} not found.")
        return

    # Create backup dir
    backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"Moving corrupted files to: {backup_dir}")

    count = 0
    with open(args.bad_samples_file, 'r') as f:
        for line in f:
            pdb_id = line.strip()
            if not pdb_id:
                continue
                
            filename = f"{pdb_id}_features.npz"
            src_path = feature_dir / filename
            dst_path = backup_dir / filename
            
            if src_path.exists():
                try:
                    shutil.move(str(src_path), str(dst_path))
                    print(f"Moved: {filename}")
                    count += 1
                except Exception as e:
                    print(f"Error moving {filename}: {e}")
            else:
                print(f"Warning: File not found {filename}")

    print(f"\nCleanup complete. Moved {count} files.")

if __name__ == "__main__":
    main()
