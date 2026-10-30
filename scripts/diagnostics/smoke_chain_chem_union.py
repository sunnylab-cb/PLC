#!/usr/bin/env python3
"""Reuse production loss/backward/cap/VAL smoke with the opt-in union dataset."""
import argparse
from functools import partial
import inspect
from pathlib import Path
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/train"), str(Path(__file__).resolve().parent)]
from src.data.pocket_dataset_v8 import PocketDatasetV8
from src.data.chain_chem_union import ChainChemUnionDataset
import smoke_local_spatial_scan as smoke
import smoke_libre_transfer as batches
import train_chain_chem_union as trainer
from prepare_chain_chem_union import VARIANTS, verify

if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--experiment", required=True, type=Path)
    p.add_argument("--variant", required=True, choices=VARIANTS)
    args, _ = p.parse_known_args()
    cfg = yaml.safe_load((args.experiment / "prepared" / f"{args.variant}.yaml").read_text())
    factory = partial(ChainChemUnionDataset, manifest_file=cfg["chain_union_manifest"])
    factory.__signature__ = inspect.signature(PocketDatasetV8)
    smoke.PocketDatasetV8 = batches.PocketDatasetV8 = factory
    trainer.install_dataset(cfg)
    smoke.main(variants=VARIANTS, verifier=verify, trainer_module=trainer)
