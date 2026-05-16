"""
Dataset for Pocket-GVP-Fusion V8 with Complete Features V3.

V7 improvements:
- Loads Complete Features V3 (62D PhysChem + 72D Spatial Scalar + 6×3D Spatial Vectors)
- Separates physical-chemical features from spatial features
- Provides spatial vector features for enhanced GVP processing

Handles data organized by year ranges:
- data/complete_features_v3/YEAR/PDB_ID/{pdb_id}_protein.pt
- data/esm2_latents/YEAR/PDB_ID/{pdb_id}_protein.pt
- data/ligand_features_v4/YEAR/PDB_ID/{pdb_id}_ligand.pt
- data/residue_ligand_distances_vdw.pt
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np
import random


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PocketDatasetV7(Dataset):
    """
    Dataset for loading protein-ligand features with Complete Features V3.
    V7: Uses V3 features (62D PhysChem + 72D Spatial Scalar + 6×3D Spatial Vectors).

    Directory structure:
        data/complete_features_v3/YEAR/PDB_ID/{pdb_id}_protein.pt
        data/esm2_latents/YEAR/PDB_ID/{pdb_id}_protein.pt
        data/ligand_features_v4/YEAR/PDB_ID/{pdb_id}_ligand.pt

    Features loaded:
        - ESM2 embeddings: [N, 1280]
        - Physical-chemical features: [N, 62]
        - Spatial scalar features: [N, 72]
        - Spatial vector features: [N, 6, 3]
        - GVP coords: [N, 3] (Cα coordinates)
        - GVP vectors: [N, 4, 3] (computed from all_atom_coords)
        - Ligand features: [M, 96]  # V4: 96-dim
        - Ligand coords: [M, 3]
        - Labels: [N] (VDW-based binding site labels)
    """

    # Year ranges to scan
    YEAR_RANGES = ['1981-2000', '2001-2010', '2011-2019']

    # Blacklist of corrupted files
    BLACKLIST = set()

    _DISTANCES_CACHE: Dict[str, Dict] = {}

    def __init__(
        self,
        data_root: str,
        pdb_ids: Optional[List[str]] = None,
        max_seq_len: int = 1000,
        max_atoms: int = 100,
        distances_file: str = None,
        distances_file_original: str = None,
        ligand_features_dir: str = 'ligand_features_v4',  # V7: Use V4 ligand features
        protein_features_dir: str = 'complete_features_v3',  # V7: Use V3 protein features
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
            distances_file: Path to VDW-based distances file
            distances_file_original: Path to original fixed-threshold distances (for comparison)
            ligand_features_dir: Ligand features directory name (default: ligand_features_v4)
            protein_features_dir: Protein features directory name (default: complete_features_v3)
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
        self.protein_features_dir = self.data_root / protein_features_dir
        self.esm_dir = self.data_root / 'esm2_latents'
        self.ligand_dir = self.data_root / ligand_features_dir

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
            protein_year = self.protein_features_dir / year_range

            if not protein_year.exists():
                continue

            for complex_dir in protein_year.iterdir():
                if not complex_dir.is_dir():
                    continue

                complex_id = complex_dir.name

                # Skip blacklisted
                if complex_id in self.BLACKLIST:
                    continue

                # Check if all required files exist
                protein_file = complex_dir / f'{complex_id}_protein.pt'
                esm_file = self.esm_dir / year_range / complex_id / f'{complex_id}_protein.pt'
                ligand_file = self.ligand_dir / year_range / complex_id / f'{complex_id}_ligand.pt'

                if protein_file.exists() and esm_file.exists() and ligand_file.exists():
                    complex_ids.append(complex_id)

        return sorted(complex_ids)

    def _resolve_distances_path(self, distances_file: Optional[str]) -> Path:
        if distances_file is None:
            return self.data_root / 'residue_ligand_distances_vdw.pt'
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
        return f"[PocketDatasetV7{split}]"

    def _log(self, msg: str) -> None:
        if self.verbose:
            self._print(msg)

    def _warn(self, msg: str) -> None:
        self._print(msg)

    def _log_summary(self) -> None:
        prefix = self._log_prefix()
        parts = [
            f"{prefix} complexes={len(self.pdb_ids)}",
            f"protein_features={self.protein_features_dir.name}",
            f"ligand_features={self.ligand_dir.name}",
            f"labels_vdw={self._distances_file_path}",
        ]
        if self._distances_file_original_path is not None:
            parts.append(f"labels_original={self._distances_file_original_path}")
        self._log(" | ".join(parts))

    def __len__(self) -> int:
        return len(self.pdb_ids)

    def _find_file(self, pdb_id: str, base_dir: Path, suffix: str) -> Optional[Path]:
        """Find file in nested year directories."""
        for year_range in self.YEAR_RANGES:
            file_path = base_dir / year_range / pdb_id / f'{pdb_id}_{suffix}.pt'
            if file_path.exists():
                return file_path
        return None

    def __getitem__(self, idx: int) -> Dict:
        pdb_id = self.pdb_ids[idx]

        # Find files
        protein_file = self._find_file(pdb_id, self.protein_features_dir, 'protein')
        esm_file = self._find_file(pdb_id, self.esm_dir, 'protein')
        ligand_file = self._find_file(pdb_id, self.ligand_dir, 'ligand')

        if protein_file is None or esm_file is None or ligand_file is None:
            raise FileNotFoundError(f"Missing files for {pdb_id}")

        # Load data
        protein_data = torch.load(protein_file, weights_only=False)
        esm_data = torch.load(esm_file, weights_only=False)
        ligand_data = torch.load(ligand_file, weights_only=False)

        # Extract ESM embeddings
        esm = esm_data['embeddings']  # [N, 1280]

        # Extract V3 protein features
        if 'physchem_features' in protein_data:
            protein_physchem = protein_data['physchem_features']  # [N, 62]
            protein_spatial_scalar = protein_data['spatial_scalar_features']  # [N, 72]
            protein_spatial_vector = protein_data['spatial_vector_features']  # [N, 6, 3]
        else:
            raise KeyError(f"Cannot find V3 features in {protein_file}")

        # Extract coordinates
        gvp_coords = protein_data['ca_coords']  # [N, 3]

        # Compute GVP vectors from all_atom_coords
        all_atom_coords = protein_data['all_atom_coords']  # List of [n_atoms, 3]
        gvp_vectors = self._compute_gvp_vectors(all_atom_coords, gvp_coords)  # [N, 4, 3]

        # Extract ligand features - V4: 96-dim
        ligand_features = ligand_data['atom_features']  # [M, 96]
        ligand_coords = ligand_data['coords']  # [M, 3]

        # Verify dimensions match
        if len(ligand_features) != len(ligand_coords):
            if 'coords_original' in ligand_data and ligand_data['coords_original'] is not None:
                ligand_coords = ligand_data['coords_original']
                num_heavy = len(ligand_coords)
                ligand_features = ligand_features[:num_heavy]

        # Get distance labels (VDW-based)
        if pdb_id in self.distances_dict:
            distance_data = self.distances_dict[pdb_id]
            if isinstance(distance_data, dict):
                if 'distances_vdw_adjusted' in distance_data:
                    distances_vdw_adjusted = distance_data['distances_vdw_adjusted']
                    is_binding_vdw = distance_data.get('is_binding_vdw', None)
                    labels = self._vdw_to_labels(distances_vdw_adjusted, is_binding_vdw)
                elif 'is_binding_vdw' in distance_data:
                    distances = distance_data['distances']
                    is_binding_vdw = distance_data['is_binding_vdw']
                    labels = self._distances_to_labels(distances)
                else:
                    distances = distance_data['distances']
                    labels = self._distances_to_labels(distances)
            else:
                distances = distance_data
                labels = self._distances_to_labels(distances)
        else:
            # Fallback: all Far class
            labels = torch.full((len(protein_physchem),), 2, dtype=torch.long)

        # Get original distance labels for comparison (optional)
        labels_original = None
        if self.distances_dict_original is not None and pdb_id in self.distances_dict_original:
            distance_data_orig = self.distances_dict_original[pdb_id]
            if isinstance(distance_data_orig, dict):
                distances_orig = distance_data_orig['distances']
            else:
                distances_orig = distance_data_orig
            labels_original = self._distances_to_labels(distances_orig)

        # Ensure all features have consistent length
        lengths = [
            len(esm), len(protein_physchem), len(protein_spatial_scalar),
            len(protein_spatial_vector), len(gvp_coords), len(gvp_vectors), len(labels)
        ]
        if labels_original is not None:
            lengths.append(len(labels_original))
        n_residues = min(lengths)

        # Truncate all to consistent length
        esm = esm[:n_residues]
        protein_physchem = protein_physchem[:n_residues]
        protein_spatial_scalar = protein_spatial_scalar[:n_residues]
        protein_spatial_vector = protein_spatial_vector[:n_residues]
        gvp_coords = gvp_coords[:n_residues]
        gvp_vectors = gvp_vectors[:n_residues]
        labels = labels[:n_residues]
        if labels_original is not None:
            labels_original = labels_original[:n_residues]

        n_atoms = len(ligand_features)

        # Truncate if too long
        if n_residues > self.max_seq_len:
            esm = esm[:self.max_seq_len]
            protein_physchem = protein_physchem[:self.max_seq_len]
            protein_spatial_scalar = protein_spatial_scalar[:self.max_seq_len]
            protein_spatial_vector = protein_spatial_vector[:self.max_seq_len]
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
            'protein_physchem': protein_physchem.float(),  # V7: [N, 62]
            'protein_spatial_scalar': protein_spatial_scalar.float(),  # V7: [N, 72]
            'protein_spatial_vector': protein_spatial_vector.float(),  # V7: [N, 6, 3]
            'gvp_coords': gvp_coords.float(),
            'gvp_vectors': gvp_vectors.float(),  # [N, 4, 3]
            'ligand_features': ligand_features.float(),  # V7: [M, 96]
            'ligand_coords': ligand_coords.float(),
            'labels': labels.long(),
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

        Args:
            distances_vdw_adjusted: [N] VDW-adjusted distances
            is_binding_vdw: [N] VDW-based binding site flags

        Returns:
            labels: [N] - 0: Close (<0.5Å), 1: Medium (0.5-4.5Å), 2: Far (>4.5Å)
        """
        labels = torch.zeros_like(distances_vdw_adjusted, dtype=torch.long)

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
