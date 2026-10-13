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

import csv

import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np
import random

from src.contact_labels import decode_contact_labels, validate_contact_axis


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atom_crop_bounds(n_atoms: int, max_atoms: int, policy: str) -> tuple[int, int]:
    """Return [start, end) atom bounds for deterministic ligand truncation."""
    n_atoms = int(n_atoms)
    max_atoms = int(max_atoms)
    if max_atoms < 0:
        return 0, n_atoms
    if max_atoms == 0:
        raise ValueError("max_atoms=0 is ambiguous; use a positive cap or -1 to disable atom truncation")
    if n_atoms <= max_atoms:
        return 0, n_atoms
    policy = str(policy or "head").lower()
    if policy in {"head", "first"}:
        return 0, max_atoms
    if policy in {"alternating_ends", "trim_ends", "center"}:
        # Remove atoms front/back/front/back until max_atoms remains.  Keeping a
        # contiguous middle segment preserves atom order and makes edge remap safe.
        excess = n_atoms - max_atoms
        drop_front = (excess + 1) // 2
        return drop_front, drop_front + max_atoms
    raise ValueError(f"unknown atom_truncation_policy={policy!r}")


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
        esm_features_dir: str = 'esm2_latents',
        esm_alignment_features_dir: Optional[str] = None,
        esm3_features_dir: Optional[str] = None,
        ligand_features_dir: str = 'ligand_features_v4',
        protein_features_dir: str = 'complete_features_v3',     # fallback (conf0 base)
        nma_features_dir: str = 'complete_features_v3_nma',     # V8: NMA confs dir
        nma_n_confs: int = 3,        # V8: total confs per protein (including conf0)
        nma_max_conf: int = 9,       # V8: max available conf index (conf0..conf9)
        use_nma: bool = False,       # V8: False for val/test, True for train
        label_contact_thresh: float = 3.5,   # d_min Contact threshold (Å)
        label_near_thresh: float = 6.0,      # d_min Near threshold (Å)
        use_spatial_clamp: bool = False,
        spatial_clamp_dims: Optional[List[int]] = None,
        spatial_clamp_range: Optional[List[float]] = None,
        atom_truncation_policy: str = "head",
        assignment_negatives_file: Optional[str] = None,
        assignment_max_negatives: int = 4,
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
        self.atom_truncation_policy = str(atom_truncation_policy or "head").lower()

        # V8: NMA parameters.  Set before optional assignment artifact logging,
        # because _log_prefix() reports these fields.
        self.use_nma = use_nma
        self.nma_n_confs = max(1, nma_n_confs)
        self.nma_max_conf = nma_max_conf

        self.assignment_max_negatives = max(0, int(assignment_max_negatives))
        self.assignment_negatives: Dict[str, List[Dict]] = {}
        if assignment_negatives_file:
            neg_path = Path(assignment_negatives_file)
            if neg_path.exists():
                neg_data = torch.load(neg_path, map_location="cpu", weights_only=False)
                self.assignment_negatives = neg_data.get("negatives", neg_data)
                self._log(
                    f"{self._log_prefix()} assignment negatives loaded: "
                    f"{len(self.assignment_negatives)} samples from {neg_path}"
                )
            else:
                self._warn(f"Warning: assignment negatives file not found: {neg_path}")

        # Label thresholds for d_min (minimum heavy-atom pairwise distance)
        self.label_contact_thresh = label_contact_thresh
        self.label_near_thresh = label_near_thresh
        self.use_spatial_clamp = bool(use_spatial_clamp)
        self.spatial_clamp_dims = [int(d) for d in (spatial_clamp_dims or [])]
        _clamp_range = spatial_clamp_range or [-5.0, 5.0]
        if len(_clamp_range) != 2:
            raise ValueError("spatial_clamp_range must be [min, max]")
        self.spatial_clamp_min = float(_clamp_range[0])
        self.spatial_clamp_max = float(_clamp_range[1])

        # Feature directories
        self.protein_features_dir = self.data_root / protein_features_dir  # fallback
        self.nma_features_dir = self.data_root / nma_features_dir          # V8: NMA dir
        self.esm_dir = self.data_root / esm_features_dir
        self.esm_alignment_dir = (
            self.data_root / esm_alignment_features_dir
            if esm_alignment_features_dir
            else None
        )
        self.esm3_dir = self.data_root / esm3_features_dir if esm3_features_dir else None
        self.ligand_dir = self.data_root / ligand_features_dir
        self._esm_manifest_index = self._load_esm_manifest_index()

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
                    esm_file = self._find_primary_esm_file(cid)
                    esm_alignment_ok = True
                    if self.esm_alignment_dir is not None:
                        esm_alignment_ok = (
                            self.esm_alignment_dir / year_range / cid / f'{cid}_protein.pt'
                        ).exists()
                    esm3_ok = True
                    if self.esm3_dir is not None:
                        esm3_ok = (self.esm3_dir / year_range / cid / f'{cid}_protein.pt').exists()
                    ligand_file = self.ligand_dir / year_range / cid / f'{cid}_ligand.pt'
                    if (
                        has_protein
                        and esm_file is not None
                        and esm_file.exists()
                        and esm_alignment_ok
                        and esm3_ok
                        and ligand_file.exists()
                    ):
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
            f"esm_features={self.esm_dir.name}",
            f"ligand_features={self.ligand_dir.name}",
            f"labels_vdw={self._distances_file_path}",
        ]
        if self.esm3_dir is not None:
            parts.append(f"esm3_features={self.esm3_dir.name}")
        if self.esm_alignment_dir is not None:
            parts.append(f"esm_alignment={self.esm_alignment_dir.name}")
        if self._esm_manifest_index:
            parts.append(f"esm_manifest={len(self._esm_manifest_index)}")
        if self.use_spatial_clamp:
            parts.append(
                f"spatial_clamp=dims{self.spatial_clamp_dims}"
                f"[{self.spatial_clamp_min},{self.spatial_clamp_max}]"
            )
        if self.atom_truncation_policy != "head":
            parts.append(f"atom_policy={self.atom_truncation_policy}")
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

    def _load_esm_manifest_index(self) -> Dict[str, Path]:
        """Load an optional PDB-to-unique-feature manifest from the ESM root."""
        manifest_path = self.esm_dir / "pdb_manifest.tsv"
        if not manifest_path.exists():
            return {}
        with manifest_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        required = {"pdb_id", "unique_id", "seq_sha256"}
        if not rows or not required.issubset(rows[0]):
            raise ValueError(
                f"{manifest_path}: expected columns {sorted(required)}, "
                f"found {sorted(rows[0]) if rows else []}"
            )
        index: Dict[str, Path] = {}
        for row in rows:
            pdb_id = row["pdb_id"].strip().lower()
            if pdb_id in index:
                raise ValueError(f"{manifest_path}: duplicate pdb_id={pdb_id}")
            index[pdb_id] = (
                self.esm_dir
                / "unique"
                / row["seq_sha256"][:2]
                / f"{row['unique_id']}.pt"
            )
        return index

    def _find_primary_esm_file(self, pdb_id: str) -> Optional[Path]:
        if self._esm_manifest_index:
            path = self._esm_manifest_index.get(pdb_id.lower())
            return path if path is not None and path.exists() else None
        return self._find_file(pdb_id, self.esm_dir, 'protein')

    def _find_file(self, pdb_id: str, base_dir: Path, suffix: str) -> Optional[Path]:
        """Find file in nested year directories (ESM / ligand)."""
        for year_range in self.YEAR_RANGES:
            f = base_dir / year_range / pdb_id / f'{pdb_id}_{suffix}.pt'
            if f.exists():
                return f
        return None

    @staticmethod
    def _residue_key_list(data: Dict) -> Optional[List[tuple]]:
        chains = data.get("chain_id", None)
        residue_idx = data.get("residue_index", data.get("residue_indices", None))
        ins_codes = data.get("insertion_code", data.get("insertion_codes", None))
        if chains is None or residue_idx is None:
            return None
        n = min(len(chains), len(residue_idx))
        if ins_codes is None:
            ins_codes = [""] * n
        n = min(n, len(ins_codes))
        return [
            (str(chains[i]), int(residue_idx[i]), str(ins_codes[i]).strip())
            for i in range(n)
        ]

    @classmethod
    def _align_aux_esm_to_reference(cls, aux_data: Dict, ref_data: Dict) -> torch.Tensor:
        """Align auxiliary residue embeddings to the primary ESM/protein residue keys.

        ESM3 caches can omit unresolved residues that are present in the legacy
        ESM2/protein feature cache.  Key-based alignment keeps the main sample
        indexing stable and fills missing ESM3 rows with zeros.
        """
        aux_emb = aux_data["embeddings"].float()
        ref_keys = cls._residue_key_list(ref_data)
        aux_keys = cls._residue_key_list(aux_data)
        if ref_keys is None or aux_keys is None:
            n = int(ref_data["embeddings"].shape[0])
            out = aux_emb.new_zeros((n, aux_emb.shape[-1]))
            out[: min(n, aux_emb.shape[0])] = aux_emb[: min(n, aux_emb.shape[0])]
            return out

        aux_map = {key: i for i, key in enumerate(aux_keys)}
        out = aux_emb.new_zeros((len(ref_keys), aux_emb.shape[-1]))
        for i, key in enumerate(ref_keys):
            j = aux_map.get(key)
            if j is not None:
                out[i] = aux_emb[j]
        return out

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
        esm_file = self._find_primary_esm_file(pdb_id)
        esm_alignment_file = (
            self._find_file(pdb_id, self.esm_alignment_dir, 'protein')
            if self.esm_alignment_dir is not None
            else None
        )
        esm3_file = self._find_file(pdb_id, self.esm3_dir, 'protein') if self.esm3_dir is not None else None
        ligand_file = self._find_file(pdb_id, self.ligand_dir, 'ligand')

        if (
            protein_file is None
            or esm_file is None
            or ligand_file is None
            or (self.esm_alignment_dir is not None and esm_alignment_file is None)
            or (self.esm3_dir is not None and esm3_file is None)
        ):
            raise FileNotFoundError(
                f"Missing files for {pdb_id} conf{conf_idx}: "
                f"protein={protein_file}, esm={esm_file}, esm_align={esm_alignment_file}, "
                f"esm3={esm3_file}, ligand={ligand_file}"
            )

        protein_data = torch.load(protein_file, weights_only=False)
        esm_data = torch.load(esm_file, weights_only=False)
        esm_alignment_data = (
            torch.load(esm_alignment_file, weights_only=False)
            if esm_alignment_file is not None
            else None
        )
        esm3_data = torch.load(esm3_file, weights_only=False) if esm3_file is not None else None
        ligand_data = torch.load(ligand_file, weights_only=False)

        # ── Extract features ─────────────────────────────────────────────────
        esm = esm_data['embeddings']  # [N, D] or [N, K, D]
        esm_reference = esm_alignment_data if esm_alignment_data is not None else esm_data
        esm3 = self._align_aux_esm_to_reference(esm3_data, esm_reference) if esm3_data is not None else None

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
                if 'contact_label_protocol' in distance_data:
                    validate_contact_axis(distance_data, protein_data, esm_reference)
                    labels, raw_distances = decode_contact_labels(distance_data)
                    if any(len(x) != len(labels) for x in (
                        esm, protein_physchem, protein_spatial_scalar, protein_spatial_vector, gvp_coords
                    )):
                        raise ValueError(f"New contact-label feature length mismatch: {pdb_id}")
                elif 'distances_vdw_adjusted' in distance_data:
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
        if esm3 is not None:
            lengths.append(len(esm3))
        if labels_original is not None:
            lengths.append(len(labels_original))
        if raw_distances is not None:
            lengths.append(len(raw_distances))
        n_residues = min(lengths)

        esm = esm[:n_residues]
        if esm3 is not None:
            esm3 = esm3[:n_residues]
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
            if esm3 is not None:
                esm3 = esm3[:self.max_seq_len]
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

        atom_start = 0
        if n_atoms > self.max_atoms:
            atom_start, atom_end = _atom_crop_bounds(n_atoms, self.max_atoms, self.atom_truncation_policy)
            ligand_features = ligand_features[atom_start:atom_end]
            ligand_coords = ligand_coords[atom_start:atom_end]
            n_atoms = atom_end - atom_start

        if self.use_spatial_clamp and self.spatial_clamp_dims:
            max_dim = max(self.spatial_clamp_dims)
            if protein_spatial_scalar.shape[-1] <= max_dim:
                raise ValueError(
                    f"spatial_clamp_dims={self.spatial_clamp_dims} exceed "
                    f"protein_spatial_scalar dim={protein_spatial_scalar.shape[-1]}"
                )
            protein_spatial_scalar = protein_spatial_scalar.clone()
            protein_spatial_scalar[..., self.spatial_clamp_dims] = (
                protein_spatial_scalar[..., self.spatial_clamp_dims]
                .clamp(self.spatial_clamp_min, self.spatial_clamp_max)
            )

        if ligand_edge_index is None or ligand_edge_attr is None:
            ligand_edge_index = torch.zeros((2, 0), dtype=torch.long)
            ligand_edge_attr = torch.zeros((0, 6), dtype=torch.float32)
        else:
            ligand_edge_index = ligand_edge_index.long()
            ligand_edge_attr = ligand_edge_attr.float()
            edge_mask = (
                (ligand_edge_index[0] >= atom_start) &
                (ligand_edge_index[0] < atom_start + n_atoms) &
                (ligand_edge_index[1] >= atom_start) &
                (ligand_edge_index[1] < atom_start + n_atoms)
            )
            ligand_edge_index = ligand_edge_index[:, edge_mask] - atom_start
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

        if esm3 is not None:
            result['esm3'] = esm3.float()

        if labels_original is not None:
            result['labels_original'] = labels_original.long()

        if raw_distances is not None:
            result['raw_distances'] = raw_distances.float()

        if self.assignment_negatives:
            neg_items = self.assignment_negatives.get(pdb_id, [])
            neg_indices = []
            neg_ids = []
            for item in neg_items[: self.assignment_max_negatives]:
                idx = item.get("contact_indices", [])
                idx = torch.as_tensor(idx, dtype=torch.long)
                if idx.numel() == 0:
                    continue
                idx = idx[(idx >= 0) & (idx < n_residues)]
                if idx.numel() == 0:
                    continue
                neg_indices.append(idx)
                neg_ids.append(str(item.get("neg_pdb_id", "")))
            if neg_indices:
                result["assignment_neg_indices"] = neg_indices
                result["assignment_neg_ids"] = neg_ids

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
