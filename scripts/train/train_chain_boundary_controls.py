#!/usr/bin/env python3
"""Scratch G0 boundary controls with unchanged losses, data and F1 selection."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts/train"), str(ROOT / "scripts/diagnostics")]
from prepare_chain_boundary_controls import VARIANTS, verify


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--smoke", action="store_true")
    args, rest = parser.parse_known_args()
    if any(s == "--resume" or s.startswith("--resume=") for s in rest):
        raise ValueError("Scratch only")
    verify(args.experiment, smoke=not args.smoke, variant=args.variant)
    from train_chainwise_esm_complex import install, load_smoke
    trainer = install(args.experiment, training_code=args.experiment / "model_code")
    import src.models.gvp_layers as graph
    if not Path(graph.__file__).resolve().is_relative_to((args.experiment / "model_code").resolve()):
        raise RuntimeError("Model imported from wrong snapshot")
    print("CHAIN_BOUNDARY_MODEL_SOURCE", graph.__file__, flush=True)
    if args.smoke:
        smoke = load_smoke()
        sys.argv = [sys.argv[0], "--experiment", str(args.experiment), "--variant", args.variant] + rest
        smoke.main(variants=tuple(VARIANTS), verifier=verify, trainer_module=trainer)
    else:
        sys.argv = [sys.argv[0], "--config", str(args.experiment / "prepared" / f"{args.variant}.yaml")] + rest
        trainer.base.main()


if __name__ == "__main__":
    main()
