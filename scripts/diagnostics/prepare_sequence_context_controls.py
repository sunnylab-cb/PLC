#!/usr/bin/env python3
"""Freeze MLP/TCN/BiGRU controls: replace G0's front context encoder only.

Same expanded joint GLOBAL-SI30 inputs, seed44 scratch, F1 patience30.
No feature extraction, external benchmark selection, or TEST forward.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_libre_transfer import sha256, source_hashes
from prepare_local_spatial_scan import BASE_SHA, make_configs as local_configs, verify

VARIANTS = {"M0_pointwise": "pointwise", "C0_tcn": "tcn", "R0_bigru": "bigru"}


def make_configs(base, experiment, source):
    reference = local_configs(base, experiment, source)["G0_small_global"]
    result = {}
    for name, operator in VARIANTS.items():
        cfg = copy.deepcopy(reference)
        cfg["experiment"].update(name=f"joint_si30_{name}_scratch_seed44",
            version_suffix="seqcontext", output_dir=str(experiment / "outputs" / name))
        cfg["model"].update(context_variant=name, protein_context_operator=operator,
                             protein_context_max_ca_step=4.5)
        result[name] = cfg
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--require-smoke", action="store_true")
    args = parser.parse_args()
    root = args.experiment
    if args.verify:
        verify(root, args.require_smoke)
        print("SEQUENCE_CONTEXT_FROZEN_INPUTS_VERIFIED", flush=True)
        return
    if args.source_root is None:
        parser.error("--source-root is required to prepare configs")
    source = args.source_root
    base_path = source / "prepared/GFGF.yaml"
    if sha256(base_path) != BASE_SHA:
        raise RuntimeError("Expected the audited expanded-joint GFGF configuration")
    old_path = source / "prepared/provenance.json"
    old = json.loads(old_path.read_text())
    assets = {k: v for k, v in old["asset_hashes"].items()
              if k.startswith(str(source / "dataset")) or k.startswith(str(source / "split"))}
    assets.update({str(base_path): BASE_SHA, str(old_path): sha256(old_path)})
    for path, expected in assets.items():
        if sha256(path) != expected:
            raise RuntimeError(f"Upstream input changed: {path}")
    prepared = root / "prepared"
    prepared.mkdir(parents=True, exist_ok=False)
    configs = make_configs(yaml.safe_load(base_path.read_text()), root, source)
    for name, cfg in configs.items():
        (prepared / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    manifest = dict(asset_hashes=assets, source_hashes=source_hashes(root / "code_v1"),
        config_hashes={name: sha256(prepared / f"{name}.yaml") for name in VARIANTS},
        variants=VARIANTS, seed=44, start="scratch", test_forward_used=False,
        cohort="expanded_joint_global_si30", counts=old["loader_qc"]["counts"],
        sources=old["loader_qc"]["by_source"],
        checkpoint_policy="best_f1_c0_model.pt and best finite loss best_model.pt only",
        reference="G0_small_global (13312384_0, source snapshot 91079ad)",
        note="Only front context changes; preserve GVP, fusion, co-attention and local54 delta. "
             "CA jumps >4.5A are conservative segment breaks, not complete missing-residue alignment. "
             "BiGRU recurrent math FP32; rest uses unchanged production AMP.")
    (prepared / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")
    verify(root)
    print(json.dumps({"prepared": str(prepared), "variants": VARIANTS}), flush=True)


if __name__ == "__main__":
    main()
