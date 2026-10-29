#!/usr/bin/env python3
"""Production AMP/backward, cap1800 stress, VAL and checkpoint smoke for controls."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/train"), str(Path(__file__).resolve().parent)]
from smoke_local_spatial_scan import main
from prepare_sequence_context_controls import VARIANTS, verify
import train_sequence_context_controls as trainer

if __name__ == "__main__":
    main(variants=VARIANTS, verifier=verify, trainer_module=trainer)
