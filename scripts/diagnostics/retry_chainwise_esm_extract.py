#!/usr/bin/env python3
"""Retry the frozen full-complex extractor with one scoped ESM2 safe global.

The original code_v1, prepared manifest, features and G0 recipe stay unchanged.
Only argparse.Namespace is allowlisted, matching the audited ESM2 checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import torch


def retry(experiment):
    script = experiment / "code_v1/preparation/protein/esm2_chainwise_complex.py"
    manifest = json.loads((experiment / "prepared/manifest.json").read_text())
    if hashlib.sha256(script.read_bytes()).hexdigest() != manifest["extraction_script_sha256"]:
        raise ValueError("Frozen extractor changed; do not bypass its provenance checks")
    original_argv = sys.argv
    try:
        sys.argv = [str(script), "extract", "--experiment", str(experiment)]
        with torch.serialization.safe_globals([argparse.Namespace]):
            runpy.run_path(str(script), run_name="__main__")
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", required=True, type=Path)
    args = p.parse_args()
    retry(args.experiment)
