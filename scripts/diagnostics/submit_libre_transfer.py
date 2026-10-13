#!/usr/bin/env python3
"""Submit the approved four-arm array only after a matching GPU smoke passes."""

import argparse
import fcntl
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def validate_smoke(smoke, provenance_sha, variants):
    if smoke.get("passed") is not True or smoke.get("provenance_sha256") != provenance_sha:
        raise RuntimeError("GPU smoke missing, failed, or belongs to different source/configs")
    runs = smoke.get("runs", [])
    if len(runs) != len(variants) or {r["name"] for r in runs} != set(variants):
        raise RuntimeError("GPU smoke did not cover all four approved arms")
    for row in runs:
        expected = 82304 if "readout" in row["name"] else 0
        if row.get("extra_parameters") != expected:
            raise RuntimeError("GPU smoke parameter budget differs from the approved matrix")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", required=True, type=Path)
    ap.add_argument("--smoke-job", required=True)
    args = ap.parse_args()
    root = args.experiment.resolve()
    sys.path.insert(0, str(root / "code_v1/scripts/diagnostics"))
    from prepare_libre_transfer import VARIANTS, verify_prepared

    if not args.smoke_job.isdigit():
        raise ValueError("Expected numeric Slurm smoke job ID")
    with (root / "submission.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        verify_prepared(root)
        digest = hashlib.sha256((root / "prepared/provenance.json").read_bytes()).hexdigest()
        smoke = json.loads((root / "smoke_pass.json").read_text())
        validate_smoke(smoke, digest, VARIANTS)
        record = root / "submission.json"
        if record.exists():
            previous = json.loads(record.read_text())
            if previous["provenance_sha256"] != digest:
                raise RuntimeError("Existing submission has different provenance")
            print("ALREADY_SUBMITTED", json.dumps(previous), flush=True)
            return
        command = ["/opt/slurm/bin/sbatch", "--parsable", "--array=0-3%2",
                   "--dependency=afterok:" + args.smoke_job, "--kill-on-invalid-dep=yes",
                   "--chdir=" + str(root),
                   "--output=" + str(root / "logs/train_%A_%a.out"),
                   "--error=" + str(root / "logs/train_%A_%a.err"),
                   "--export=ALL,STAGE=train,EXPERIMENT=" + str(root),
                   str(root / "code_v1/scripts/slurm/run_libre_transfer.slurm")]
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {job!r}; inspect queue before retry")
        result = dict(job_id=job, smoke_job=args.smoke_job, variants=list(VARIANTS),
                      provenance_sha256=digest, submitted_utc=datetime.now(timezone.utc).isoformat(),
                      command=command, status="submitted_not_necessarily_running")
        tmp = record.with_suffix(".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n")
        tmp.replace(record)
        print("TRAINING_SUBMITTED", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
