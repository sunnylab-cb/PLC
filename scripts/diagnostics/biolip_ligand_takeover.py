#!/usr/bin/env python3
"""Replace an allocated ligand job only after a 64-CPU successor starts.

The immutable original extractor retains its manifest, atomic output writes,
skip-existing behavior and chemistry settings. QC must already wait for the
successor, so cancelling the predecessor cannot trigger premature QC.
"""

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


SLURM = Path("/opt/slurm/bin")
RUNNER = "scripts/slurm/run_biolip_feature_expansion_cpu.slurm"
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY",
            "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def fields(text):
    return dict(re.findall(r"(?:^|\s)([A-Za-z0-9_/]+)=([^\s]*)", text))


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def command(tool, *args):
    return subprocess.run([str(SLURM / tool), *args], check=True,
                          text=True, capture_output=True).stdout.strip()


def job(job_id, allow_expired=False):
    result = subprocess.run([str(SLURM / "scontrol"), "show", "job", "-o", job_id],
                            text=True, capture_output=True)
    if result.returncode:
        if allow_expired and "Invalid job id specified" in result.stderr:
            return None
        raise RuntimeError(result.stderr or result.stdout)
    return fields(result.stdout)


def validate_target(info, job_id, experiment, name, uid):
    require(info.get("JobId") == job_id, "unexpected job ID")
    require(info.get("UserId", "").endswith(f"({uid})"), "job owner mismatch")
    require(info.get("JobName") == name, "job name mismatch")
    require(info.get("WorkDir") == str(experiment / "code"), "job workdir mismatch")
    require(info.get("Command") == str(experiment / "code" / RUNNER),
            "job command mismatch")


def validate_qc(info, successor):
    require(info.get("JobState") == "PENDING", "QC is not pending")
    require(info.get("Dependency") in
            {f"afterok:{successor}", f"afterok:{successor}(unfulfilled)"},
            "QC must depend ONLY on successful completion of the successor")


def verify_source(experiment, deployment):
    for relative, expected in deployment["files"].items():
        path = experiment / "code" / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"original source changed: {relative}")


def wait_for_exit(old_id, lock_path, timeout=900):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        active = command("squeue", "--noheader", "--user", str(os.getuid()),
                         "--format=%i %T")
        if any(line.split()[0] == old_id for line in active.splitlines() if line.split()):
            time.sleep(5)
            continue
        with lock_path.open("a+") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                time.sleep(5)
                continue
            fcntl.flock(handle, fcntl.LOCK_UN)
            return
    raise RuntimeError("predecessor/lock did not exit; refusing concurrent extraction")


def main():
    experiment = Path(os.environ["EXPERIMENT"]).resolve()
    old_id, qc_id, new_id = (os.environ[k] for k in
                             ("OLD_JOB_ID", "QC_JOB_ID", "SLURM_JOB_ID"))
    require(all(re.fullmatch(r"[0-9]+", x) for x in (old_id, qc_id, new_id)),
            "job IDs must be scalar numeric IDs")
    require(len({old_id, qc_id, new_id}) == 3, "job IDs must differ")
    require(os.environ.get("SLURM_CPUS_PER_TASK") == "64", "requires 64 allocated CPUs")
    deployment = json.loads((experiment / "deployment.json").read_text())
    require(deployment["jobs"]["ligand_remaining"]["job_id"] == old_id,
            "predecessor not in the original deployment")
    require(deployment["jobs"]["qc"]["job_id"] == qc_id,
            "QC not in the original deployment")
    verify_source(experiment, deployment)
    old = job(old_id, allow_expired=True)
    if old is not None:
        validate_target(old, old_id, experiment, "bio_expand_ligand_remaining", os.getuid())
    qc = job(qc_id)
    validate_target(qc, qc_id, experiment, "bio_expand_qc", os.getuid())
    validate_qc(qc, new_id)
    current = job(new_id)
    require(current.get("JobState") == "RUNNING", "successor is not allocated")
    require(current.get("NumCPUs") == "64", "unexpected successor allocation")
    receipt = {"old_job": old_id, "new_job": new_id, "qc_job": qc_id,
               "original_source_commit": deployment["source_commit"],
               "original_source_verified": True, "old_snapshot": old,
               "new_snapshot": current, "qc_snapshot": qc,
               "workers": 64, "chemistry_changed": False,
               "overwrite": False, "training_approved": False}
    receipt_path = experiment / f"ligand_resume64_{new_id}.json"

    def record(phase, **extra):
        receipt.update(phase=phase, updated_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), **extra)
        temp = receipt_path.with_suffix(".tmp.json")
        temp.write_text(json.dumps(receipt, indent=2) + "\n")
        temp.replace(receipt_path)
        print(phase, json.dumps(extra), flush=True)

    record("preflight_passed")
    try:
        if old is not None and old.get("JobState") not in TERMINAL | {"COMPLETING"}:
            command("scancel", old_id)
            record("predecessor_cancel_requested")
        wait_for_exit(old_id, experiment / "ligand_remaining.lock")
        log = experiment / "logs/ligand_remaining_extract.log"
        with log.open("a") as handle:
            handle.write(f"\nRESUME64 job={new_id} replaces={old_id} "
                         f"at={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        record("starting_original_extractor")
        env = dict(os.environ, EXPERIMENT=str(experiment), CODE_ROOT=str(experiment / "code"),
                   STAGE="ligand_remaining", SMOKE_LIMIT="",
                   BIO_ROOT="/hpc2hdd/home/shuang886/biolip",
                   QC_POOL="/hpc2hdd/home/shuang886/PLC/experiments/"
                           "internal_si30_dual_fixed5_20261013/pdbbind_biolip_joint/pool")
        os.execve("/bin/bash", ["bash", str(experiment / "code" / RUNNER)], env)
    except Exception as exc:
        record("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
