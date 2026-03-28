"""
Pocket-GVP-Fusion V10.

V10 updates the feature interfaces for:
- protein complete_features_v4
- ligand_features_v5

Architecture remains aligned with the V9m2/V9m3 family, while input dimensions
are updated for the new independent feature pipelines.
"""

from __future__ import annotations

from typing import Dict, Optional

from .pocket_gvp_fusion_v9m2 import PocketGVPFusionV9m2


class PocketGVPFusionV10(PocketGVPFusionV9m2):
    DEFAULT_CONFIG = {
        **PocketGVPFusionV9m2.DEFAULT_CONFIG,
        'd_phys': 62,
        'd_spatial_scalar': 89,
        'd_ligand_atom': 109,
        'n_gvp_vectors': 12,  # 4 runtime GVP vectors + 8 protein v4 spatial vectors
    }

    def __init__(self, config: Optional[Dict] = None):
        merged = {**self.DEFAULT_CONFIG, **(config or {})}
        super().__init__(merged)
