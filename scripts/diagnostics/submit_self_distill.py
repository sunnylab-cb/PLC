#!/usr/bin/env python3
"""Submit once: cache-eval -> GPU smoke -> two independent one-GPU scratch runs."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.diagnostics.prepare_self_distill import verify_prepared


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    cache = ap.add_mutually_exclusive_group(required=True)
    cache.add_argument("--cache-job", type=int)
    cache.add_argument("--cache-ready", action="store_true",
                       help="Validate an already completed cache instead of depending on a stale Slurm job")
    args = ap.parse_args()
    root = args.experiment
    verify_prepared(root, check_cache=args.cache_ready)
    (root / "logs").mkdir(exist_ok=True)
    with (root / "submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        record = root / "submission.json"
        if record.exists():
            raise RuntimeError(f"Already submitted or partially submitted; inspect {record} before retry")
        state = dict(cache_job=args.cache_job, cache_ready_verified=args.cache_ready, jobs=[])

        def persist():
            tmp = record.with_suffix(".tmp")
            tmp.write_text(json.dumps(state, indent=2) + "\n")
            tmp.replace(record)

        persist()
        parent = args.cache_job
        for stage in ("smoke", "train"):
            cmd = ["/opt/slurm/bin/sbatch", "--parsable", "--kill-on-invalid-dep=yes",
                   f"--export=ALL,EXPERIMENT={root},STAGE={stage}",
                   f"--output={root}/logs/{stage}_%A_%a.out", f"--error={root}/logs/{stage}_%A_%a.err"]
            if parent is not None:
                cmd.append(f"--dependency=afterok:{parent}")
            cmd += ["--time=02:00:00"] if stage == "smoke" else ["--array=0-1%2"]
            cmd += [str(root / "code_v1/scripts/slurm/run_self_distill.slurm")]
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            parent = int(result.stdout.strip().split(";")[0])
            state["jobs"].append(dict(stage=stage, job_id=parent, command=cmd, stderr=result.stderr))
            persist()
            print(stage, parent, flush=True)


if __name__ == "__main__":
    main()
