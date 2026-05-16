"""
Dataset for Pocket-GVP-Fusion v2 with nested directory structure.

Handles data organized by year ranges:
- data/complete_features/YEAR/PDB_ID/{pdb_id}_protein.pt
- data/esm2_latents/YEAR/PDB_ID/{pdb_id}_protein.pt
- data/ligand_features/YEAR/PDB_ID/{pdb_id}_ligand.pt
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import random


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PocketDatasetV3(Dataset):
    """
    Dataset for loading protein-ligand features from nested directory structure.

    Directory structure:
        data/complete_features/YEAR/PDB_ID/{pdb_id}_protein.pt
        data/esm2_latents/YEAR/PDB_ID/{pdb_id}_protein.pt
        data/ligand_features/YEAR/PDB_ID/{pdb_id}_ligand.pt

    Features loaded:
        - ESM2 embeddings: [N, 1280]
        - Complete features: [N, 82] (42D physchem + 14D surface + 26D geometry)
        - GVP coords: [N, 3] (Cα coordinates)
        - GVP vectors: [N, 4, 3] (computed from all_atom_coords)
        - Ligand features: [M, 35]
        - Ligand coords: [M, 3]
        - Labels: [N] (3-class distance labels)
    """

    # Year ranges to scan
    YEAR_RANGES = ['1981-2000', '2001-2010', '2011-2019']

    # Blacklist of corrupted files (cleared - files have been regenerated)
    BLACKLIST = set()

    def __init__(
        self,
        data_root: str,
        pdb_ids: Optional[List[str]] = None,
        max_seq_len: int = 1000,
        max_atoms: int = 100,
        distances_file: str = None
    ):
        """
        Args:
            data_root: Root data directory (e.g., "data/")
            pdb_ids: List of PDB IDs to include (if None, uses all)
            max_seq_len: Maximum protein sequence length
            max_atoms: Maximum ligand atoms
            distances_file: Path to residue-ligand distances file
        """
        super().__init__()

        self.data_root = Path(data_root)
        self.max_seq_len = max_seq_len
        self.max_atoms = max_atoms

        # Feature directories
        self.physchem_dir = self.data_root / 'complete_features'
        self.esm_dir = self.data_root / 'esm2_latents'  # Changed from esm3_latents to esm2_latents
        self.ligand_dir = self.data_root / 'ligand_features'

        # Load distance labels
        self.distances_dict = self._load_distances(distances_file)

        # Collect valid complex IDs
        all_ids = self._collect_complex_ids()

        # Filter by provided pdb_ids if specified
        if pdb_ids is not None:
            self.pdb_ids = [pid for pid in pdb_ids if pid in all_ids]
        else:
            self.pdb_ids = all_ids

        print(f"Dataset size: {len(self.pdb_ids)} complexes")

    def _collect_complex_ids(self) -> List[str]:
        """Collect all valid complex IDs from nested directories."""
        complex_ids = []

        for year_range in self.YEAR_RANGES:
            physchem_year = self.physchem_dir / year_range

            if not physchem_year.exists():
                continue

            for complex_dir in physchem_year.iterdir():
                if not complex_dir.is_dir():
                    continue

                complex_id = complex_dir.name

                # Skip blacklisted
                if complex_id in self.BLACKLIST:
                    continue

                # Check if all required files exist
                physchem_file = complex_dir / f'{complex_id}_protein.pt'
                esm_file = self.esm_dir / year_range / complex_id / f'{complex_id}_protein.pt'
                ligand_file = self.ligand_dir / year_range / complex_id / f'{complex_id}_ligand.pt'

                if physchem_file.exists() and esm_file.exists() and ligand_file.exists():
                    complex_ids.append(complex_id)

        return sorted(complex_ids)

    def _load_distances(self, distances_file: Optional[str]) -> Dict:
        """Load residue-ligand distance labels."""
        if distances_file is None:
            distances_file = self.data_root / 'residue_ligand_distances_v2.pt'
        else:
            distances_file = Path(distances_file)

        if distances_file.exists():
            print(f"Loading distances from {distances_file}")
            return torch.load(distances_file, weights_only=False)
        else:
            print(f"Warning: Distances file not found: {distances_file}")
            return {}

    def _find_file(self, pdb_id: str, subdir: str, filename: str) -> Optional[Path]:
        """Find file in nested year directories."""
        for year_range in self.YEAR_RANGES:
            file_path = self.data_root / subdir / year_range / pdb_id / filename
            if file_path.exists():
                return file_path
        return None

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        pdb_id = self.pdb_ids[idx]

        # Find files
        physchem_file = self._find_file(pdb_id, 'complete_features', f'{pdb_id}_protein.pt')
        esm_file = self._find_file(pdb_id, 'esm2_latents', f'{pdb_id}_protein.pt')
        ligand_file = self._find_file(pdb_id, 'ligand_features', f'{pdb_id}_ligand.pt')

        if not all([physchem_file, esm_file, ligand_file]):
            raise FileNotFoundError(f"Missing files for {pdb_id}")

        # Load protein features
        physchem_data = torch.load(physchem_file, weights_only=False)
        esm_data = torch.load(esm_file, weights_only=False)
        ligand_data = torch.load(ligand_file, weights_only=False)

        # Extract ESM2 embeddings
        if isinstance(esm_data, dict):
            esm = esm_data.get('embeddings', esm_data.get('latents', None))
            if esm is None:
                raise ValueError(f"Cannot find ESM embeddings in {esm_file}")
        else:
            esm = esm_data

        # Extract complete features (82D: 42D physchem + 14D surface + 26D geometry)
        protein_scalar = physchem_data['complete_features']  # [N, 82]

        # Extract coordinates
        gvp_coords = physchem_data['ca_coords']  # [N, 3]

        # Compute GVP vectors from all_atom_coords
        all_atom_coords = physchem_data['all_atom_coords']  # List of [n_atoms, 3]
        gvp_vectors = self._compute_gvp_vectors(all_atom_coords, gvp_coords)  # [N, 4, 3]

        # Extract ligand features
        ligand_features = ligand_data['atom_features']  # [M, 35]
        ligand_coords = ligand_data['coords']  # [M, 3]

        # Get distance labels
        if pdb_id in self.distances_dict:
            distance_data = self.distances_dict[pdb_id]
            if isinstance(distance_data, dict):
                distances = distance_data['distances']  # [N]
            else:
                distances = distance_data
            labels = self._distances_to_labels(distances)
        else:
            # Fallback: all Far class
            labels = torch.full((len(protein_scalar),), 2, dtype=torch.long)

        # Ensure all features have consistent length
        n_residues = min(len(esm), len(protein_scalar), len(gvp_coords), len(gvp_vectors), len(labels))

        # Truncate all to consistent length
        esm = esm[:n_residues]
        protein_scalar = protein_scalar[:n_residues]
        gvp_coords = gvp_coords[:n_residues]
        gvp_vectors = gvp_vectors[:n_residues]
        labels = labels[:n_residues]

        n_atoms = len(ligand_features)

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

        return {
            'pdb_id': pdb_id,
            'esm': esm.float(),
            'protein_scalar': protein_scalar.float(),
            'gvp_coords': gvp_coords.float(),
            'gvp_vectors': gvp_vectors.float(),
            'ligand_features': ligand_features.float(),
            'ligand_coords': ligand_coords.float(),
            'labels': labels.long(),
            'n_residues': n_residues,
            'n_atoms': n_atoms
        }

    def _compute_gvp_vectors(self, all_atom_coords: List, ca_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute GVP vector features from all-atom coordinates.

        Returns:
            gvp_vectors: [N, 4, 3] - 4 directional vectors per residue
        """
        n_residues = len(all_atom_coords)
        gvp_vectors = torch.zeros(n_residues, 4, 3)

        for i, atom_coords in enumerate(all_atom_coords):
            if atom_coords is None or len(atom_coords) == 0:
                continue

            # Convert to tensor if needed
            if not isinstance(atom_coords, torch.Tensor):
                atom_coords = torch.tensor(atom_coords, dtype=torch.float32)

            ca = ca_coords[i]  # [3]

            # Vector 1: CA to centroid
            centroid = atom_coords.mean(dim=0)
            gvp_vectors[i, 0] = centroid - ca

            # Vector 2-4: CA to first 3 atoms (or zeros if not enough atoms)
            for j in range(min(3, len(atom_coords))):
                gvp_vectors[i, j + 1] = atom_coords[j] - ca

        return gvp_vectors

    def _distances_to_labels(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to 3-class labels.

        Args:
            distances: [N] minimum distances to ligand

        Returns:
            labels: [N] - 0: Contact (<3.5Å), 1: Near (3.5-6.0Å), 2: Far (>6.0Å)
        """
        labels = torch.zeros_like(distances, dtype=torch.long)
        labels[distances < 3.5] = 0  # Contact
        labels[(distances >= 3.5) & (distances < 6.0)] = 1  # Near
        labels[distances >= 6.0] = 2  # Far
        return labels


def create_data_splits_v3(
    data_root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create random train/val/test splits.

    Args:
        data_root: Root data directory
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed

    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    set_seed(seed)

    # Create temporary dataset to collect all IDs
    dataset = PocketDatasetV3(data_root, pdb_ids=None)
    all_ids = dataset.pdb_ids

    # Shuffle
    random.shuffle(all_ids)

    # Split
    n = len(all_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_ids = all_ids[:n_train]
    val_ids = all_ids[n_train:n_train + n_val]
    test_ids = all_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def create_dataloader_v3(
    data_root: str,
    pdb_ids: List[str],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    max_seq_len: int = 1000,
    max_atoms: int = 100,
    seed: int = 42,
    distances_file: str = None
) -> DataLoader:
    """
    Create a DataLoader for training/evaluation.

    Args:
        data_root: Root data directory
        pdb_ids: List of PDB IDs to include
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
        max_seq_len: Maximum protein length
        max_atoms: Maximum ligand atoms
        seed: Random seed
        distances_file: Path to distances file

    Returns:
        DataLoader instance
    """
    from .collate import pocket_collate_fn

    set_seed(seed)

    dataset = PocketDatasetV3(
        data_root=data_root,
        pdb_ids=pdb_ids,
        max_seq_len=max_seq_len,
        max_atoms=max_atoms,
        distances_file=distances_file
    )

    # Create generator with fixed seed
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
