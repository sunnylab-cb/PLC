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


"""
Dataset for Pocket-GVP-Fusion V9 with NMA augmentation support.

V8 improvements over V7:
- NMA augmentation: per-protein multiple conformations from complete_features_v3_nma/
- Training: each pdb_id expanded to nma_n_confs samples (conf0 always included)
- Val/Test: always uses conf0 only (no expansion)
- ESM latents and Ligand features are unchanged across conformations (loaded by pdb_id)
- Labels are looked up by pdb_id (conf suffix stripped)

Directory structure:
    data/complete_features_v3/YEAR/PDB_ID/{pdb_id}_protein.pt         (base, fallback)
    data/complete_features_v3_nma/YEAR/PDB_ID/{pdb_id}_protein_conf0.pt
    data/complete_features_v3_nma/YEAR/PDB_ID/{pdb_id}_protein_conf1.pt
    ...
    data/esm2_latents/YEAR/PDB_ID/{pdb_id}_protein.pt
    data/ligand_features_v4/YEAR/PDB_ID/{pdb_id}_ligand.pt
    data/residue_ligand_distances_vdw.pt
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


class PocketDatasetV8(Dataset):
    """
    V9 Dataset: V3 features + NMA augmentation.

    Training mode (use_nma=True):
        __len__ = len(pdb_ids) * nma_n_confs
        __getitem__(idx):
            pdb_idx  = idx // nma_n_confs   → which protein
            conf_slot = idx % nma_n_confs    → which conformation slot
            conf_slot==0 → always load conf0
            conf_slot>0  → randomly select from available conf1..confN

    Val/Test mode (use_nma=False):
        __len__ = len(pdb_ids)
        __getitem__(idx) → always loads conf0 (or base protein file)

    ESM and Ligand features are always loaded by pdb_id (no conf suffix).
    Labels are always looked up by pdb_id.
    """

    YEAR_RANGES = ['1981-2000', '2001-2010', '2011-2019']
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
        ligand_features_dir: str = 'ligand_features_v4',
        protein_features_dir: str = 'complete_features_v3',     # fallback (conf0 base)
        nma_features_dir: str = 'complete_features_v3_nma',     # V8: NMA confs dir
        nma_n_confs: int = 3,        # V8: total confs per protein (including conf0)
        nma_max_conf: int = 9,       # V8: max available conf index (conf0..conf9)
        use_nma: bool = False,       # V8: False for val/test, True for train
        label_contact_thresh: float = 3.5,   # d_min Contact threshold (Å)
        label_near_thresh: float = 6.0,      # d_min Near threshold (Å)
        *,
        split: Optional[str] = None,
        verbose: bool = True,
        print_fn: Callable[..., None] = print
    ):
        super().__init__()

        self.split = split
        self.verbose = verbose
        self._print = print_fn

        self.data_root = Path(data_root)
        self.max_seq_len = max_seq_len
        self.max_atoms = max_atoms

        # V8: NMA parameters
        self.use_nma = use_nma
        self.nma_n_confs = max(1, nma_n_confs)
        self.nma_max_conf = nma_max_conf

        # Label thresholds for d_min (minimum heavy-atom pairwise distance)
        self.label_contact_thresh = label_contact_thresh
        self.label_near_thresh = label_near_thresh

        # Feature directories
        self.protein_features_dir = self.data_root / protein_features_dir  # fallback
        self.nma_features_dir = self.data_root / nma_features_dir          # V8: NMA dir
        self.esm_dir = self.data_root / 'esm2_latents'
        self.ligand_dir = self.data_root / ligand_features_dir

        # Load distance labels
        self._distances_file_path = self._resolve_distances_path(distances_file)
        self.distances_dict = self._load_distances(self._distances_file_path, kind="vdw")

        self.distances_dict_original = None
        if distances_file_original:
            self._distances_file_original_path = self._resolve_distances_path(distances_file_original)
            self.distances_dict_original = self._load_distances(self._distances_file_original_path, kind="original")
        else:
            self._distances_file_original_path = None

        # Collect valid complex IDs
        all_ids = self._collect_complex_ids()

        if pdb_ids is not None:
            self.pdb_ids = [pid for pid in pdb_ids if pid in all_ids]
        else:
            self.pdb_ids = all_ids

        # V8: Pre-scan available NMA conf indices per pdb_id
        # Dict: pdb_id → sorted list of available conf indices (e.g. [0,1,2,...,9])
        self._nma_confs: Dict[str, List[int]] = {}
        if self.use_nma:
            self._scan_nma_confs()

        self._log_summary()

    def _scan_nma_confs(self) -> None:
        """Pre-scan available NMA conf files per pdb_id to avoid per-item filesystem calls."""
        for pdb_id in self.pdb_ids:
            available = []
            for year_range in self.YEAR_RANGES:
                nma_pdb_dir = self.nma_features_dir / year_range / pdb_id
                if not nma_pdb_dir.exists():
                    continue
                for i in range(self.nma_max_conf + 1):
                    if (nma_pdb_dir / f'{pdb_id}_protein_conf{i}.pt').exists():
                        available.append(i)
                break  # found the year_range dir; no need to continue
            # Fallback: if NMA dir not found, use conf0 from base dir
            if not available:
                available = [0]
            self._nma_confs[pdb_id] = sorted(available)

    def _collect_complex_ids(self) -> List[str]:
        """
        Collect valid complex IDs.
        Prefers NMA directory; falls back to base protein_features_dir.
        """
        complex_ids = set()

        # Scan NMA directory first
        scan_dirs = [self.nma_features_dir]
        if self.nma_features_dir != self.protein_features_dir:
            scan_dirs.append(self.protein_features_dir)

        for feat_dir in scan_dirs:
            for year_range in self.YEAR_RANGES:
                year_dir = feat_dir / year_range
                if not year_dir.exists():
                    continue
                for complex_dir in year_dir.iterdir():
                    if not complex_dir.is_dir():
                        continue
                    cid = complex_dir.name
                    if cid in self.BLACKLIST or cid in complex_ids:
                        continue
                    # Need conf0 (or base protein file), ESM, and ligand
                    has_protein = (
                        (complex_dir / f'{cid}_protein_conf0.pt').exists() or
                        (complex_dir / f'{cid}_protein.pt').exists()
                    )
                    esm_file = self.esm_dir / year_range / cid / f'{cid}_protein.pt'
                    ligand_file = self.ligand_dir / year_range / cid / f'{cid}_ligand.pt'
                    if has_protein and esm_file.exists() and ligand_file.exists():
                        complex_ids.add(cid)

        return sorted(complex_ids)

    def _resolve_distances_path(self, distances_file: Optional[str]) -> Path:
        if distances_file is None:
            return self.data_root / 'residue_ligand_distances_vdw.pt'
        return Path(distances_file)

    def _load_distances(self, distances_path: Path, *, kind: str) -> Dict:
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
        nma_info = f" nma={self.nma_n_confs}x" if self.use_nma else ""
        return f"[PocketDatasetV8{split}{nma_info}]"

    def _log(self, msg: str) -> None:
        if self.verbose:
            self._print(msg)

    def _warn(self, msg: str) -> None:
        self._print(msg)

    def _log_summary(self) -> None:
        prefix = self._log_prefix()
        n_samples = len(self)
        parts = [
            f"{prefix} complexes={len(self.pdb_ids)} samples={n_samples}",
            f"nma_dir={self.nma_features_dir.name}",
            f"use_nma={self.use_nma}",
            f"ligand_features={self.ligand_dir.name}",
            f"labels_vdw={self._distances_file_path}",
        ]
        self._log(" | ".join(parts))

        # V9: Print NMA availability statistics
        if self.use_nma and self._nma_confs:
            n_proteins_with_nma = len(self._nma_confs)
            conf_counts = [len(confs) for confs in self._nma_confs.values()]
            avg_confs = sum(conf_counts) / len(conf_counts) if conf_counts else 0
            min_confs = min(conf_counts) if conf_counts else 0
            max_confs = max(conf_counts) if conf_counts else 0

            # Count proteins with full NMA coverage (conf0-conf9)
            full_coverage = sum(1 for confs in self._nma_confs.values() if len(confs) >= 10)

            self._log(f"{prefix} NMA coverage: {n_proteins_with_nma}/{len(self.pdb_ids)} proteins")
            self._log(f"{prefix} NMA confs per protein: avg={avg_confs:.1f}, min={min_confs}, max={max_confs}")
            self._log(f"{prefix} NMA full coverage (10 confs): {full_coverage}/{len(self.pdb_ids)} ({100*full_coverage/len(self.pdb_ids):.1f}%)")

    def __len__(self) -> int:
        if self.use_nma:
            return len(self.pdb_ids) * self.nma_n_confs
        return len(self.pdb_ids)

    def _find_file_in_nma(self, pdb_id: str, conf_idx: int) -> Optional[Path]:
        """Find NMA conf file in nested year directories."""
        for year_range in self.YEAR_RANGES:
            f = self.nma_features_dir / year_range / pdb_id / f'{pdb_id}_protein_conf{conf_idx}.pt'
            if f.exists():
                return f
        return None

    def _find_base_protein_file(self, pdb_id: str) -> Optional[Path]:
        """Find base protein file (conf0 or _protein.pt) in fallback dir."""
        for year_range in self.YEAR_RANGES:
            # Try NMA conf0 first
            f = self.nma_features_dir / year_range / pdb_id / f'{pdb_id}_protein_conf0.pt'
            if f.exists():
                return f
            # Try base dir
            f = self.protein_features_dir / year_range / pdb_id / f'{pdb_id}_protein.pt'
            if f.exists():
                return f
        return None

    def _find_file(self, pdb_id: str, base_dir: Path, suffix: str) -> Optional[Path]:
        """Find file in nested year directories (ESM / ligand)."""
        for year_range in self.YEAR_RANGES:
            f = base_dir / year_range / pdb_id / f'{pdb_id}_{suffix}.pt'
            if f.exists():
                return f
        return None

    def __getitem__(self, idx: int) -> Dict:
        # ── Resolve pdb_id and conf_idx ─────────────────────────────────────
        if self.use_nma:
            pdb_idx = idx // self.nma_n_confs
            conf_slot = idx % self.nma_n_confs
            pdb_id = self.pdb_ids[pdb_idx]

            available_confs = self._nma_confs.get(pdb_id, [0])
            if conf_slot == 0:
                conf_idx = 0
            else:
                # Randomly sample from non-zero available confs
                non_zero = [c for c in available_confs if c != 0]
                conf_idx = random.choice(non_zero) if non_zero else 0
        else:
            pdb_id = self.pdb_ids[idx]
            conf_idx = 0

        # ── Load protein features ────────────────────────────────────────────
        if conf_idx == 0:
            protein_file = self._find_base_protein_file(pdb_id)
        else:
            protein_file = self._find_file_in_nma(pdb_id, conf_idx)
            if protein_file is None:
                protein_file = self._find_base_protein_file(pdb_id)

        # ── Load ESM and Ligand (always by pdb_id, no conf suffix) ───────────
        esm_file = self._find_file(pdb_id, self.esm_dir, 'protein')
        ligand_file = self._find_file(pdb_id, self.ligand_dir, 'ligand')

        if protein_file is None or esm_file is None or ligand_file is None:
            raise FileNotFoundError(
                f"Missing files for {pdb_id} conf{conf_idx}: "
                f"protein={protein_file}, esm={esm_file}, ligand={ligand_file}"
            )

        protein_data = torch.load(protein_file, weights_only=False)
        esm_data = torch.load(esm_file, weights_only=False)
        ligand_data = torch.load(ligand_file, weights_only=False)

        # ── Extract features ─────────────────────────────────────────────────
        esm = esm_data['embeddings']  # [N, 1280]

        if 'physchem_features' not in protein_data:
            raise KeyError(f"Cannot find V3 features in {protein_file}")

        protein_physchem = protein_data['physchem_features']          # [N, 62]
        protein_spatial_scalar = protein_data['spatial_scalar_features']  # [N, 72]
        protein_spatial_vector = protein_data['spatial_vector_features']  # [N, 6, 3]
        gvp_coords = protein_data['ca_coords']                         # [N, 3]

        # Raw RSA (index 1 in spatial_scalar_raw_features, range [0,1]).
        # Used by V12 SASA gate; kept as [N, 1] so shape is unambiguous.
        # protein_rsa_mask = 1 where real RSA exists, 0 for absent/old .pt files.
        # When mask=0, SASA gate reverts to identity (avoids spurious "fully buried" bias).
        _raw_feat = protein_data.get('spatial_scalar_raw_features', None)
        _N = len(protein_physchem)
        if _raw_feat is not None:
            protein_rsa_raw  = torch.as_tensor(_raw_feat[:, 1:2], dtype=torch.float32)  # [N, 1]
            protein_rsa_mask = torch.ones(_N, 1, dtype=torch.float32)
        else:
            protein_rsa_raw  = torch.zeros(_N, 1, dtype=torch.float32)
            protein_rsa_mask = torch.zeros(_N, 1, dtype=torch.float32)

        # chain_ids: list of str ('A','A','B',...) → integer indices [N]
        raw_chain_ids = protein_data.get('chain_ids', None)
        if raw_chain_ids is not None and len(raw_chain_ids) > 0:
            unique_chains = sorted(set(raw_chain_ids))
            chain_to_idx = {c: i for i, c in enumerate(unique_chains)}
            chain_ids_int = torch.tensor(
                [chain_to_idx[c] for c in raw_chain_ids], dtype=torch.long
            )
        else:
            chain_ids_int = torch.zeros(len(protein_physchem), dtype=torch.long)

        all_atom_coords = protein_data['all_atom_coords']
        gvp_vectors = self._compute_gvp_vectors(all_atom_coords, gvp_coords)  # [N, 4, 3]

        ligand_features = ligand_data['atom_features']   # [M, 96]
        ligand_coords = ligand_data['coords']             # [M, 3]
        ligand_edge_index = ligand_data.get('edge_index', None)
        ligand_edge_attr = ligand_data.get('edge_attr', None)

        if len(ligand_features) != len(ligand_coords):
            if 'coords_original' in ligand_data and ligand_data['coords_original'] is not None:
                ligand_coords = ligand_data['coords_original']
                ligand_features = ligand_features[:len(ligand_coords)]

        # ── Labels (always by pdb_id, invariant to NMA conf) ─────────────────
        raw_distances = None  # V10m8: raw distances for distance regression aux head
        if pdb_id in self.distances_dict:
            distance_data = self.distances_dict[pdb_id]
            if isinstance(distance_data, dict):
                if 'distances_vdw_adjusted' in distance_data:
                    labels = self._vdw_to_labels(
                        distance_data['distances_vdw_adjusted'],
                        distance_data.get('is_binding_vdw', None)
                    )
                    raw_distances = distance_data['distances_vdw_adjusted'].float()
                elif 'is_binding_vdw' in distance_data:
                    labels = self._distances_to_labels(distance_data['distances'])
                    raw_distances = distance_data['distances'].float()
                else:
                    labels = self._distances_to_labels(distance_data['distances'])
                    raw_distances = distance_data['distances'].float()
            else:
                labels = self._distances_to_labels(distance_data)
                raw_distances = distance_data.float()
        else:
            labels = torch.full((len(protein_physchem),), 2, dtype=torch.long)

        labels_original = None
        if self.distances_dict_original is not None and pdb_id in self.distances_dict_original:
            d = self.distances_dict_original[pdb_id]
            distances_orig = d['distances'] if isinstance(d, dict) else d
            labels_original = self._distances_to_labels(distances_orig)

        # ── Truncate to consistent length ─────────────────────────────────────
        lengths = [
            len(esm), len(protein_physchem), len(protein_spatial_scalar),
            len(protein_spatial_vector), len(gvp_coords), len(gvp_vectors),
            len(labels), len(chain_ids_int),
            len(protein_rsa_raw), len(protein_rsa_mask),
        ]
        if labels_original is not None:
            lengths.append(len(labels_original))
        if raw_distances is not None:
            lengths.append(len(raw_distances))
        n_residues = min(lengths)

        esm = esm[:n_residues]
        protein_physchem = protein_physchem[:n_residues]
        protein_spatial_scalar = protein_spatial_scalar[:n_residues]
        protein_spatial_vector = protein_spatial_vector[:n_residues]
        gvp_coords = gvp_coords[:n_residues]
        gvp_vectors = gvp_vectors[:n_residues]
        labels = labels[:n_residues]
        chain_ids_int = chain_ids_int[:n_residues]
        protein_rsa_raw  = protein_rsa_raw[:n_residues]
        protein_rsa_mask = protein_rsa_mask[:n_residues]
        if labels_original is not None:
            labels_original = labels_original[:n_residues]
        if raw_distances is not None:
            raw_distances = raw_distances[:n_residues]

        n_atoms = len(ligand_features)

        if n_residues > self.max_seq_len:
            esm = esm[:self.max_seq_len]
            protein_physchem = protein_physchem[:self.max_seq_len]
            protein_spatial_scalar = protein_spatial_scalar[:self.max_seq_len]
            protein_spatial_vector = protein_spatial_vector[:self.max_seq_len]
            gvp_coords = gvp_coords[:self.max_seq_len]
            gvp_vectors = gvp_vectors[:self.max_seq_len]
            labels = labels[:self.max_seq_len]
            chain_ids_int = chain_ids_int[:self.max_seq_len]
            protein_rsa_raw  = protein_rsa_raw[:self.max_seq_len]
            protein_rsa_mask = protein_rsa_mask[:self.max_seq_len]
            if labels_original is not None:
                labels_original = labels_original[:self.max_seq_len]
            if raw_distances is not None:
                raw_distances = raw_distances[:self.max_seq_len]
            n_residues = self.max_seq_len

        if n_atoms > self.max_atoms:
            ligand_features = ligand_features[:self.max_atoms]
            ligand_coords = ligand_coords[:self.max_atoms]
            n_atoms = self.max_atoms

        if ligand_edge_index is None or ligand_edge_attr is None:
            ligand_edge_index = torch.zeros((2, 0), dtype=torch.long)
            ligand_edge_attr = torch.zeros((0, 6), dtype=torch.float32)
        else:
            ligand_edge_index = ligand_edge_index.long()
            ligand_edge_attr = ligand_edge_attr.float()
            edge_mask = (
                (ligand_edge_index[0] < n_atoms) &
                (ligand_edge_index[1] < n_atoms)
            )
            ligand_edge_index = ligand_edge_index[:, edge_mask]
            ligand_edge_attr = ligand_edge_attr[edge_mask]

        result = {
            'pdb_id': pdb_id,
            'conf_idx': conf_idx,
            'esm': esm.float(),
            'protein_physchem': protein_physchem.float(),
            'protein_spatial_scalar': protein_spatial_scalar.float(),
            'protein_spatial_vector': protein_spatial_vector.float(),
            'protein_rsa_raw':  protein_rsa_raw,   # [N, 1] raw RSA value; 0 when absent
            'protein_rsa_mask': protein_rsa_mask,  # [N, 1] 1=real RSA, 0=absent (old .pt)
            'gvp_coords': gvp_coords.float(),
            'gvp_vectors': gvp_vectors.float(),
            'chain_ids': chain_ids_int,
            'ligand_features': ligand_features.float(),
            'ligand_coords': ligand_coords.float(),
            'ligand_edge_index': ligand_edge_index,
            'ligand_edge_attr': ligand_edge_attr,
            'labels': labels.long(),
            'n_residues': n_residues,
            'n_atoms': n_atoms
        }

        if labels_original is not None:
            result['labels_original'] = labels_original.long()

        if raw_distances is not None:
            result['raw_distances'] = raw_distances.float()

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
            labels: [N] - 0: Contact (<contact_thresh), 1: Near (<near_thresh), 2: Far
        """
        labels = torch.zeros_like(distances, dtype=torch.long)
        labels[distances < self.label_contact_thresh] = 0  # Contact
        labels[(distances >= self.label_contact_thresh) & (distances < self.label_near_thresh)] = 1  # Near
        labels[distances >= self.label_near_thresh] = 2  # Far
        return labels
