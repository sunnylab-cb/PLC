#!/usr/bin/env python3
"""Scratch G0 chain/chemical-union controls. Original model/loss are unchanged.

Only this entrypoint installs the opt-in dataset view. Legacy training and
benchmark scripts retain their task definitions. No resume, TEST or HOLO.
"""
from __future__ import annotations

from functools import partial
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).resolve().parent)]
import train_local_spatial_scan as trainer
from src.data.chain_chem_union import ChainChemUnionDataset

base = trainer.base
original_load_config = base.load_config
create_model = trainer.create_model


def install_dataset(config):
    if not config.get("chain_union_manifest"):
        raise ValueError("Missing opt-in chain/chemistry union manifest")
    if config["data"]["nma_n_confs"] != 1 or config["data"]["nma_max_conf"] != 0:
        raise ValueError("Union pilot only supports conf0")
    base.PocketDatasetV8 = partial(ChainChemUnionDataset, manifest_file=config["chain_union_manifest"])
    return config


def load_config(path):
    return install_dataset(original_load_config(path))


base.load_config = load_config

if __name__ == "__main__":
    if any(arg.startswith("--resume") for arg in sys.argv):
        raise ValueError("Chain-union controls start from scratch")
    base.main()
