#!/usr/bin/env python3
"""Submit a fail-closed CPU -> GPU smoke -> scratch training dependency chain.

One PDBbind-only run, no array and no existing job cancellation. Partial
submissions are recorded immediately and never blindly submitted twice.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess

from prepare_internal_codataset_sequences import digest, save_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    args = ap.parse_args()
    root, source, base = (p.resolve() for p in (args.experiment, args.source, args.base))
    code = root/"code_v1"
    if root == source or source in root.parents:
        raise ValueError("Use a separate root, outside the old/joint experiment")
    for p in (base, source/"pdbbind_only/pool/pool_summary.json",
              source/"pdbbind_only/split/all_pairs/summary.json", root/"deployment.json"):
        if not p.is_file():
            raise FileNotFoundError(p)
    deployment = json.loads((root/"deployment.json").read_text())
    for path, expected in deployment["source_sha256"].items():
        if digest(code/path) != expected:
            raise ValueError(f"Deployment source hash mismatch: {path}")
    binding = dict(experiment=str(root), source=str(source), base=str(base),
                   deployment_sha256=digest(root/"deployment.json"))
    (root/"logs").mkdir(exist_ok=True)
    with (root/"submission.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        record = root/"submission.json"
        receipt = json.loads(record.read_text()) if record.exists() else dict(binding=binding, jobs={})
        if receipt["binding"] != binding:
            raise ValueError("Previous submission belongs to a different deployment")
        previous = None
        for stage in ("prepare", "smoke", "train"):
            if stage in receipt["jobs"]:
                previous = receipt["jobs"][stage]["job_id"]
                continue
            env = f"ALL,EXPERIMENT={root}"
            if stage == "prepare":
                env += f",SOURCE={source},BASE={base}"
                script = code/"scripts/slurm/run_pdbbind_global_si30_prepare.slurm"
            else:
                env += f",STAGE={stage}"
                script = code/"scripts/slurm/run_pdbbind_global_si30_gfgf.slurm"
            command = ["/opt/slurm/bin/sbatch", "--parsable", "--chdir="+str(root),
                f"--job-name=pdb_gsi30_{stage}", "--export="+env,
                f"--output={root}/logs/{stage}_%j.out", f"--error={root}/logs/{stage}_%j.err"]
            if previous:
                command += ["--dependency=afterok:"+previous, "--kill-on-invalid-dep=yes"]
            if stage == "smoke":
                command += ["--time=02:00:00"]
            command.append(str(script))
            # A lost sbatch response must be resolved from the queue, not auto-retried.
            pending = root/(stage+"_submission_inflight.json")
            if pending.exists():
                raise RuntimeError(f"Unresolved sbatch attempt: {pending}; inspect queue before retry")
            save_json(pending, dict(command=command, utc=datetime.now(timezone.utc).isoformat()))
            response = subprocess.check_output(command, text=True).strip()
            job = response.split(";")[0]
            if not job.isdigit():
                raise RuntimeError(f"Unexpected sbatch response: {response}; inspect queue")
            receipt["jobs"][stage] = dict(job_id=job, command=command,
                utc=datetime.now(timezone.utc).isoformat(), status="submitted_not_necessarily_running")
            save_json(record, receipt)
            pending.unlink()
            previous = job
            print("SUBMITTED", stage, job, flush=True)
        print("PIPELINE_SUBMISSION", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
