#!/usr/bin/env python3
"""Freeze G0 chainwise-ESM controls; only two model flags may change."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

VARIANTS = {
    "gvp_boundary": (True, False),
    "coattn_position": (False, True),
    "both": (True, True),
}
FLAGS = ("gvp_chain_boundary_fix", "coattn_chain_position_fix")


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def source_hashes(root):
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.suffix in (".py", ".slurm")}


def configs(base, root):
    t, m = base["training"], base["model"]
    assert base["experiment"]["seed"] == 44
    assert t["early_stopping_metric"] == "f1_c0"
    assert not any(t.get(k) for k in ("init_from_checkpoint", "init_from_v11m3", "freeze_v11_base", "freeze_v11m3_base"))
    assert m["scan_variant"] == "G0_small_global" and m["d_model"] == 256
    assert m.get("protein_context_operator", "hybrid") == "hybrid"
    assert m["use_chain_features"] and m["use_v12_coattention_block"] and not m["use_compile"]
    assert m.get("use_chain_rope") is not False and m.get("use_chain_edge") is not False
    out = {}
    for name, flags in VARIANTS.items():
        c = copy.deepcopy(base)
        c["model"].update(dict(zip(FLAGS, flags)))
        c["experiment"].update(name=f"G0_chainesm_{name}_seed44", output_dir=str(root / "outputs" / name),
                               version_suffix="chainfix")
        out[name] = c
    return out


def prepare(root, upstream):
    dest = root / "prepared"
    dest.mkdir(exist_ok=True)
    if (dest / "provenance.json").exists():
        raise ValueError("Do not overwrite a frozen experiment")
    p = json.loads((upstream / "prepared/provenance.json").read_text())
    assert p["complete"] and p["no_chain_only_samples"] and p["no_label_or_split_changes"]
    basepath = upstream / "prepared/G0_chainwise_esm_complex.yaml"
    assert sha(basepath) == p["configs"]["G0_chainwise_esm_complex"]
    base = yaml.safe_load(basepath.read_text())
    assets = {k: v for k, v in p["assets"].items() if "/checkpoints/esm2/" not in k}
    assets[str(basepath)] = sha(basepath)
    assets[str(upstream / "prepared/provenance.json")] = sha(upstream / "prepared/provenance.json")
    for path, expected in assets.items():
        if sha(path) != expected:
            raise ValueError(f"Changed upstream asset: {path}")
    for name, cfg in configs(base, root).items():
        (dest / f"{name}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    report = dict(complete=True, upstream=str(upstream), assets=assets,
        configs={n: sha(dest / f"{n}.yaml") for n in VARIANTS},
        source_hashes=source_hashes(root / "code_v1"),
        model_hashes=source_hashes(root / "model_code"),
        flags=FLAGS, variants=VARIANTS, data_unchanged=True, scratch=True, no_test=True)
    (dest / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CHAIN_BOUNDARY_PREPARED", list(VARIANTS), flush=True)


def verify(root, smoke=False, variant=None):
    p = json.loads((root / "prepared/provenance.json").read_text())
    assert p["complete"] and p["scratch"] and p["data_unchanged"] and p["no_test"]
    for dirname, key in (("code_v1", "source_hashes"), ("model_code", "model_hashes")):
        if source_hashes(root / dirname) != p[key]:
            raise ValueError(f"Frozen source changed: {dirname}")
    for name, expected in p["configs"].items():
        if sha(root / "prepared" / f"{name}.yaml") != expected:
            raise ValueError(f"Changed config: {name}")
    # Large feature/label assets were checked during preparation. Recheck small
    # split, normalization and provenance files at every launch, without weights.
    for path, expected in p["assets"].items():
        if not Path(path).exists() or (Path(path).suffix != ".pt" and sha(path) != expected):
            raise ValueError(f"Changed input: {path}")
    if smoke:
        s = json.loads((root / "smoke" / f"{variant}.json").read_text())
        assert s["passed"] and not s["test_forward_used"]
        assert s["provenance_sha256"] == sha(root / "prepared/provenance.json")
    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.experiment.resolve(), args.upstream.resolve())
