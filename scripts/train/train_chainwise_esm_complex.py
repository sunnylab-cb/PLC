#!/usr/bin/env python3
"""G0 scratch with chainwise ESM, but unchanged FULL-COMPLEX model inputs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from preparation.protein.esm2_chainwise_complex import VARIANT, verify


def load_smoke():
    # The frozen G0 snapshot predates the injectable smoke-test interface.
    path = ROOT / "scripts/diagnostics/smoke_local_spatial_scan.py"
    spec = importlib.util.spec_from_file_location("chainwise_smoke", path)
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    return smoke


def install(experiment):
    manifest = json.loads((experiment / "prepared/manifest.json").read_text())
    old = Path(manifest["training"]) / "code_v1"
    sys.path[:0] = [str(old), str(old / "scripts/train"), str(old / "scripts/diagnostics")]
    import train_local_spatial_scan as trainer
    original = trainer.source_metrics
    writers = []
    writer_class = trainer.base.SummaryWriter
    val_pass = 0

    def observed_writer(*args, **kwargs):
        writer = writer_class(*args, **kwargs)
        writers.append(writer)
        return writer

    trainer.base.SummaryWriter = observed_writer

    def stratified(records, gate):
        nonlocal val_pass
        result = original(records, gate)
        for name, multichain in (("single_chain", False), ("multichain", True)):
            subset = [r for r in records if (int(r[0]["n_chains"]) > 1) == multichain]
            for key, value in original(subset, gate).items():
                source, metric = key.split("_", 1)
                result[f"{source}_{name}_{metric}"] = value
        if writers:
            for key, value in result.items():
                writers[-1].add_scalar(f"ValStrata/{key}", value, val_pass)
            writers[-1].flush()
            log = Path(writers[-1].log_dir) / "fullcomplex_val_strata.jsonl"
            with log.open("a") as handle:
                handle.write(json.dumps(dict(val_pass=val_pass + 1, joint_val_gate=gate,
                                             metrics=result)) + "\n")
        val_pass += 1
        return result

    trainer.source_metrics = stratified
    return trainer


if __name__ == "__main__":
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args, rest = p.parse_known_args()
    if any(a == "--resume" or a.startswith("--resume=") for a in rest):
        raise ValueError("This is scratch training, never resume")
    verify(args.experiment, smoke=not args.smoke)
    trainer = install(args.experiment)
    if args.smoke:
        smoke = load_smoke()
        sys.argv = [sys.argv[0], "--experiment", str(args.experiment), "--variant", VARIANT] + rest
        smoke.main(variants=(VARIANT,), verifier=verify, trainer_module=trainer)
    else:
        sys.argv = [sys.argv[0], "--config", str(args.experiment / "prepared" / f"{VARIANT}.yaml")] + rest
        trainer.base.main()
