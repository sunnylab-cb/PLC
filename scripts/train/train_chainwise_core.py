#!/usr/bin/env python3
"""Scratch G0 from a verified core bundle; no ESM weights or downloads."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/train")]
from scripts.sync.sync_chainwise_core_to_star import VARIANT, verify_core
from train_chainwise_esm_complex import install, load_smoke


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--bundle", required=True, type=Path)
    p.add_argument("--smoke", action="store_true")
    args, rest = p.parse_known_args()
    if any(a == "--resume" or a.startswith("--resume=") for a in rest):
        raise ValueError("Core bundle is scratch-only")
    verify_core(args.bundle, smoke=not args.smoke)
    trainer = install(args.bundle, training_code=args.bundle / "model_code")
    if args.smoke:
        smoke = load_smoke()
        sys.argv = [sys.argv[0], "--experiment", str(args.bundle), "--variant", VARIANT] + rest
        smoke.main(variants=(VARIANT,), verifier=verify_core, trainer_module=trainer)
    else:
        sys.argv = [sys.argv[0], "--config", str(args.bundle / "prepared" / f"{VARIANT}.yaml")] + rest
        trainer.base.main()
