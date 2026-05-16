"""
Dataset for Pocket-GVP-Fusion v5 with VDW-based binding site labels.

V5 improvements:
- Uses BioLiP-style VDW radius-based distance thresholds
- Distance threshold: d < r_vdw(P) + r_vdw(L) + 0.5 Å
- More physically accurate binding site definition

Handles data organized by year ranges:
- data/complete_features/YEAR/PDB_ID/{pdb_id}_protein.pt
- data/esm2_latents/YEAR/PDB_ID/{pdb_id}_protein.pt
- data/ligand_features_v3/YEAR/PDB_ID/{pdb_id}_ligand.pt  # 80-dim features
- data/residue_ligand_distances_vdw.pt  # VDW-based distance labels
"""

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
import random


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PocketDatasetV6(Dataset):
    """
    Dataset for loading protein-ligand features from nested directory structure.
    V5: Uses VDW-based distance labels (BioLiP standard).

    Directory structure:
        data/complete_features/YEAR/PDB_ID/{pdb_id}_protein.pt
        data/esm2_latents/YEAR/PDB_ID/{pdb_id}_protein.pt
        data/ligand_features_v3/YEAR/PDB_ID/{pdb_id}_ligand.pt

    Features loaded:
        - ESM2 embeddings: [N, 1280]
        - Complete features: [N, 82] (42D physchem + 14D surface + 26D geometry)
        - GVP coords: [N, 3] (Cα coordinates)
        - GVP vectors: [N, 4, 3] (computed from all_atom_coords)
        - Ligand features: [M, 80]  # 80-dim (35D V2 + 45D global)
        - Ligand coords: [M, 3]
        - Labels: [N] (VDW-based binding site labels)
    """

    # Year ranges to scan
    YEAR_RANGES = ['1981-2000', '2001-2010', '2011-2019']

    # Blacklist of corrupted files (cleared - files have been regenerated)
    BLACKLIST = set()

    _DISTANCES_CACHE: Dict[str, Dict] = {}

    def __init__(
        self,
        data_root: str,
        pdb_ids: Optional[List[str]] = None,
        max_seq_len: int = 1000,
        max_atoms: int = 100,
        distances_file: str = None,
        distances_file_original: str = None,  # V5: 原始固定阈值标签用于对比
        ligand_features_dir: str = 'ligand_features_v3',
        protein_features_dir: str = 'complete_features',  # V7: configurable protein features
        *,
        split: Optional[str] = None,
        verbose: bool = True,
        print_fn: Callable[..., None] = print
    ):
        """
        Args:
            data_root: Root data directory (e.g., "data/")
            pdb_ids: List of PDB IDs to include (if None, uses all)
            max_seq_len: Maximum protein sequence length
            max_atoms: Maximum ligand atoms
            distances_file: Path to VDW-based distances file (V5)
            distances_file_original: Path to original fixed-threshold distances (for comparison)
            ligand_features_dir: Ligand features directory name (default: ligand_features_v3)
            split: Optional split name for logging (e.g., "train"/"val"/"test")
            verbose: Whether to print dataset summary/logs
            print_fn: Print-like function for logging (e.g., rank-0 printer)
        """
        super().__init__()

        self.split = split
        self.verbose = verbose
        self._print = print_fn

        self.data_root = Path(data_root)
        self.max_seq_len = max_seq_len
        self.max_atoms = max_atoms

        # Feature directories
        self.physchem_dir = self.data_root / protein_features_dir  # V7: configurable
        self.esm_dir = self.data_root / 'esm2_latents'
        self.ligand_dir = self.data_root / ligand_features_dir  # V4: configurable

        # Load distance labels (VDW-based)
        self._distances_file_path = self._resolve_distances_path(distances_file)
        self.distances_dict = self._load_distances(self._distances_file_path, kind="vdw")

        # Load original distance labels for comparison (optional)
        self.distances_dict_original = None
        if distances_file_original:
            self._distances_file_original_path = self._resolve_distances_path(distances_file_original)
            self.distances_dict_original = self._load_distances(self._distances_file_original_path, kind="original")
        else:
            self._distances_file_original_path = None

        # Collect valid complex IDs
        all_ids = self._collect_complex_ids()

        # Filter by provided pdb_ids if specified
        if pdb_ids is not None:
            self.pdb_ids = [pid for pid in pdb_ids if pid in all_ids]
        else:
            self.pdb_ids = all_ids

        self._log_summary()

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

    def _resolve_distances_path(self, distances_file: Optional[str]) -> Path:
        if distances_file is None:
            return self.data_root / 'residue_ligand_distances_v2.pt'
        return Path(distances_file)

    def _load_distances(self, distances_path: Path, *, kind: str) -> Dict:
        """Load residue-ligand distance labels with per-process caching."""
        cache_key = str(distances_path.resolve())
        if cache_key in self._DISTANCES_CACHE:
            return self._DISTANCES_CACHE[cache_key]

        if distances_path.exists():
            data = torch.load(distances_path, weights_only=False)
            self._DISTANCES_CACHE[cache_key] = data
            return data

        self._warn(f"Warning: Distances file not found ({kind}): {distances_path}")
        self._DISTANCES_CACHE[cache_key] = {}
        return {}

    def _log_prefix(self) -> str:
        split = f" {self.split}" if self.split else ""
        return f"[PocketDatasetV6{split}]"

    def _log(self, msg: str) -> None:
        if self.verbose:
            self._print(msg)

    def _warn(self, msg: str) -> None:
        # Warnings are important enough to keep even when verbose=False in most callsites,
        # but keep them gated to avoid repeated spam when many datasets are created.
        # In practice, train script should pass verbose=True on rank-0 for at least one dataset.
        self._print(msg)

    def _log_summary(self) -> None:
        prefix = self._log_prefix()
        parts = [
            f"{prefix} complexes={len(self.pdb_ids)}",
            f"ligand_dir={self.ligand_dir}",
            f"labels_vdw={self._distances_file_path}",
        ]
        if self._distances_file_original_path is not None:
            parts.append(f"labels_original={self._distances_file_original_path}")
        self._log(" | ".join(parts))

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
        physchem_file = self._find_file(pdb_id, self.physchem_dir.name, f'{pdb_id}_protein.pt')
        esm_file = self._find_file(pdb_id, 'esm2_latents', f'{pdb_id}_protein.pt')

        # V4: Use ligand_dir attribute
        ligand_file = None
        for year_range in self.YEAR_RANGES:
            test_path = self.ligand_dir / year_range / pdb_id / f'{pdb_id}_ligand.pt'
            if test_path.exists():
                ligand_file = test_path
                break

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

        # Extract complete features
        # V7: Support both complete_features (82D) and complete_features_v2 (94D)
        if 'complete_features_v2' in physchem_data:
            protein_scalar = physchem_data['complete_features_v2']  # [N, 94] V7
        elif 'complete_features' in physchem_data:
            protein_scalar = physchem_data['complete_features']  # [N, 82] V6
        else:
            raise KeyError(f"Cannot find complete_features in {physchem_file}")

        # Extract coordinates
        gvp_coords = physchem_data['ca_coords']  # [N, 3]

        # Compute GVP vectors from all_atom_coords
        all_atom_coords = physchem_data['all_atom_coords']  # List of [n_atoms, 3]
        gvp_vectors = self._compute_gvp_vectors(all_atom_coords, gvp_coords)  # [N, 4, 3]

        # Extract ligand features - V4: 80-dim
        ligand_features = ligand_data['atom_features']  # [M, 80]

        # Strategy for coordinates:
        # 1. Global features (45D) are calculated with H atoms, so we need coords with H
        # 2. Original SDF files usually don't have H, so coords_original only has heavy atoms
        # 3. Use 'coords' (current coords with H) for consistency with global feature calculation
        # 4. These coords are optimized after adding H, which is acceptable for global features

        ligand_coords = ligand_data['coords']  # [M, 3] - includes H atoms

        # Verify dimensions match
        if len(ligand_features) != len(ligand_coords):
            # Fallback: if mismatch, use heavy atom coords and truncate features
            if 'coords_original' in ligand_data and ligand_data['coords_original'] is not None:
                ligand_coords = ligand_data['coords_original']
                num_heavy = len(ligand_coords)
                ligand_features = ligand_features[:num_heavy]

        # Get distance labels (VDW-based)
        if pdb_id in self.distances_dict:
            distance_data = self.distances_dict[pdb_id]
            if isinstance(distance_data, dict):
                # V5: Use VDW-adjusted distances for labels
                if 'distances_vdw_adjusted' in distance_data:
                    distances_vdw_adjusted = distance_data['distances_vdw_adjusted']  # [N] 扣除VDW半径后的距离
                    is_binding_vdw = distance_data.get('is_binding_vdw', None)  # [N] bool
                    labels = self._vdw_to_labels(distances_vdw_adjusted, is_binding_vdw)
                elif 'is_binding_vdw' in distance_data:
                    # Fallback: old format with actual distances
                    distances = distance_data['distances']  # [N]
                    is_binding_vdw = distance_data['is_binding_vdw']  # [N] bool
                    # Convert actual distances to VDW-adjusted (approximate)
                    # This is a fallback for old data files
                    labels = self._distances_to_labels(distances)
                else:
                    # Fallback to fixed threshold
                    distances = distance_data['distances']  # [N]
                    labels = self._distances_to_labels(distances)
            else:
                distances = distance_data
                labels = self._distances_to_labels(distances)
        else:
            # Fallback: all Far class
            labels = torch.full((len(protein_scalar),), 2, dtype=torch.long)

        # Get original distance labels for comparison (optional)
        labels_original = None
        if self.distances_dict_original is not None and pdb_id in self.distances_dict_original:
            distance_data_orig = self.distances_dict_original[pdb_id]
            if isinstance(distance_data_orig, dict):
                distances_orig = distance_data_orig['distances']  # [N]
            else:
                distances_orig = distance_data_orig
            # Original always uses fixed threshold
            labels_original = self._distances_to_labels(distances_orig)

        # Ensure all features have consistent length
        # V5: Include labels_original in length calculation if available
        lengths = [len(esm), len(protein_scalar), len(gvp_coords), len(gvp_vectors), len(labels)]
        if labels_original is not None:
            lengths.append(len(labels_original))
        n_residues = min(lengths)

        # Truncate all to consistent length
        esm = esm[:n_residues]
        protein_scalar = protein_scalar[:n_residues]
        gvp_coords = gvp_coords[:n_residues]
        gvp_vectors = gvp_vectors[:n_residues]
        labels = labels[:n_residues]
        if labels_original is not None:
            labels_original = labels_original[:n_residues]

        n_atoms = len(ligand_features)

        # Truncate if too long
        if n_residues > self.max_seq_len:
            esm = esm[:self.max_seq_len]
            protein_scalar = protein_scalar[:self.max_seq_len]
            gvp_coords = gvp_coords[:self.max_seq_len]
            gvp_vectors = gvp_vectors[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            if labels_original is not None:
                labels_original = labels_original[:self.max_seq_len]
            n_residues = self.max_seq_len

        if n_atoms > self.max_atoms:
            ligand_features = ligand_features[:self.max_atoms]
            ligand_coords = ligand_coords[:self.max_atoms]
            n_atoms = self.max_atoms

        result = {
            'pdb_id': pdb_id,
            'esm': esm.float(),
            'protein_scalar': protein_scalar.float(),
            'gvp_coords': gvp_coords.float(),
            'gvp_vectors': gvp_vectors.float(),
            'ligand_features': ligand_features.float(),  # V5: [M, 80]
            'ligand_coords': ligand_coords.float(),
            'labels': labels.long(),  # V5: VDW-based labels
            'n_residues': n_residues,
            'n_atoms': n_atoms
        }

        # Add original labels for comparison if available
        if labels_original is not None:
            result['labels_original'] = labels_original.long()

        return result

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

    def _vdw_to_labels(self, distances_vdw_adjusted: torch.Tensor, is_binding_vdw: torch.Tensor) -> torch.Tensor:
        """
        Convert VDW-adjusted distances to 3-class labels.

        V5 Strategy (Updated):
        - Use VDW-adjusted distances (distance - VDW_radii_sum)
        - This allows for consistent thresholds across different atom types

        Args:
            distances_vdw_adjusted: [N] VDW-adjusted distances (扣除VDW半径后的距离)
            is_binding_vdw: [N] VDW-based binding site flags

        Returns:
            labels: [N] - 0: Close (<0.5Å), 1: Medium (0.5-4.5Å), 2: Far (>4.5Å)
        """
        labels = torch.zeros_like(distances_vdw_adjusted, dtype=torch.long)

        # V5: Use VDW-adjusted distance-based classification
        # c0: <0.5Å (扣除VDW半径后)
        # c1: 0.5-4.5Å (扣除VDW半径后)
        # c2: >4.5Å (扣除VDW半径后)
        labels[distances_vdw_adjusted < 0.5] = 0   # Close: <0.5Å
        labels[(distances_vdw_adjusted >= 0.5) & (distances_vdw_adjusted < 4.5)] = 1  # Medium: 0.5-4.5Å
        labels[distances_vdw_adjusted >= 4.5] = 2  # Far: >4.5Å

        return labels

    def _distances_to_labels(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to 3-class labels (original fixed threshold method).

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


def create_data_splits_v5(
    data_root: str,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    ligand_features_dir: str = 'ligand_features_v3'
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create random train/val/test splits for V5.

    Args:
        data_root: Root data directory
        train_ratio: Fraction for training
        val_ratio: Fraction for validation
        test_ratio: Fraction for testing
        seed: Random seed
        ligand_features_dir: Ligand features directory name

    Returns:
        Tuple of (train_ids, val_ids, test_ids)
    """
    set_seed(seed)

    data_root = Path(data_root)
    physchem_dir = data_root / 'complete_features'
    esm_dir = data_root / 'esm2_latents'
    ligand_dir = data_root / ligand_features_dir

    all_ids: List[str] = []
    for year_range in PocketDatasetV6.YEAR_RANGES:
        physchem_year = physchem_dir / year_range
        if not physchem_year.exists():
            continue

        for complex_dir in physchem_year.iterdir():
            if not complex_dir.is_dir():
                continue

            complex_id = complex_dir.name
            if complex_id in PocketDatasetV6.BLACKLIST:
                continue

            physchem_file = complex_dir / f'{complex_id}_protein.pt'
            esm_file = esm_dir / year_range / complex_id / f'{complex_id}_protein.pt'
            ligand_file = ligand_dir / year_range / complex_id / f'{complex_id}_ligand.pt'

            if physchem_file.exists() and esm_file.exists() and ligand_file.exists():
                all_ids.append(complex_id)

    all_ids = sorted(all_ids)

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


def create_dataloader_v4(
    data_root: str,
    pdb_ids: List[str],
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    max_seq_len: int = 1000,
    max_atoms: int = 100,
    seed: int = 42,
    distances_file: str = None,
    ligand_features_dir: str = 'ligand_features_v3'
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
        ligand_features_dir: Ligand features directory name

    Returns:
        DataLoader instance
    """
    from .collate import pocket_collate_fn

    set_seed(seed)

    dataset = PocketDatasetV4(
        data_root=data_root,
        pdb_ids=pdb_ids,
        max_seq_len=max_seq_len,
        max_atoms=max_atoms,
        distances_file=distances_file,
        ligand_features_dir=ligand_features_dir
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
