#!/usr/bin/env python3
"""Create a separate HDD snapshot and submit smoke -> six noise trials.

Run on hpc2 after uploading this small allowlisted payload. Never pulls/cleans
remote git, copies datasets/checkpoints, or modifies an existing experiment.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

FILES = (
    "src/models/pocket_gvp_fusion_v11.py", "src/models/protein_context_noise.py",
    "scripts/train/train_context_noise.py", "scripts/diagnostics/prepare_context_noise.py",
    "scripts/diagnostics/smoke_context_noise.py", "scripts/diagnostics/deploy_context_noise.py",
    "scripts/slurm/run_context_noise.slurm", "tests/test_protein_context_noise.py",
)
EXPECTED = {
    "src/models/pocket_gvp_fusion_v11.py": "73647fa3ae631680f924053ff8891383a0404662a551be24b4d6d90a27bee0b9",
    "scripts/train/train_v11m4.py": "e2854fc8ea93693908b8a4b8310daca1810a075feee3ddab8658a5264779aa84",
    "scripts/train/train_v11m3.py": "4f6eb1bc61174b4d5329473e9eebec3649eb13bc21bcfd8966e92c15a3521164",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--experiment", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="Frozen original/libre-transfer experiment")
    ap.add_argument("--base-config", type=Path, required=True)
    ap.add_argument("--payload", type=Path, required=True)
    ap.add_argument("--python", default="/hpc2ssd/softwares/anaconda3/envs/af3/bin/python")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()
    root = args.experiment.resolve()
    for name, expected in EXPECTED.items():
        if digest(args.source / "code_v1" / name) != expected:
            raise RuntimeError(f"Frozen reference code changed: {name}")
    for name in FILES:
        if not (args.payload / name).is_file():
            raise FileNotFoundError(name)
    hook = """        # Optional training-only context augmentation, outside checkpoint replay.
        context_noise = getattr(self, '_protein_context_noise', None)
        if self.training and context_noise is not None:
            h_seq = context_noise(h_seq, protein_spatial_scalar, protein_coords, protein_mask)

"""
    model_source = (args.payload / "src/models/pocket_gvp_fusion_v11.py").read_text()
    if (model_source.count(hook) != 1 or hashlib.sha256(model_source.replace(hook, "").encode()).hexdigest()
            != EXPECTED["src/models/pocket_gvp_fusion_v11.py"]):
        raise RuntimeError("Model overlay changes more than the approved default-off noise hook")
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(args.source / "code_v1", root / "code_v1",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    for name in FILES:
        target = root / "code_v1" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.payload / name, target)
    (root / "deps").symlink_to((args.source / "deps").resolve(), target_is_directory=True)
    for name in ("logs", "outputs"):
        (root / name).mkdir()
    subprocess.run([args.python, str(root / "code_v1/scripts/diagnostics/prepare_context_noise.py"),
                    "--experiment", str(root), "--base", str(args.base_config)], check=True)
    deployment = dict(root=str(root), source=str(args.source),
                      payload_hashes={name: digest(root / "code_v1" / name) for name in FILES})
    (root / "deployment.json").write_text(json.dumps(deployment, indent=2) + "\n")
    if not args.submit:
        print("PREPARED_NOT_SUBMITTED", json.dumps(deployment), flush=True)
        return
    script = str(root / "code_v1/scripts/slurm/run_context_noise.slurm")

    def submit(stage, extra):
        command = ["/opt/slurm/bin/sbatch", "--parsable", "--chdir=" + str(root),
            "--output=" + str(root / f"logs/{stage}_%A_%a.out"),
            "--error=" + str(root / f"logs/{stage}_%A_%a.err"),
            f"--export=ALL,EXPERIMENT={root},STAGE={stage}", *extra, script]
        job = subprocess.check_output(command, text=True).strip().split(";")[0]
        if not job.isdigit():
            raise RuntimeError(f"Unexpected sbatch output: {job!r}; inspect queue before retry")
        record = dict(job_id=job, command=command, status="submitted_not_necessarily_running")
        (root / f"{stage}_submission.json").write_text(json.dumps(record, indent=2) + "\n")
        return job

    smoke = submit("smoke", ["--time=02:00:00"])
    train = submit("train", ["--array=0-5%2", "--dependency=afterok:" + smoke, "--kill-on-invalid-dep=yes"])
    print("CONTEXT_NOISE_SUBMITTED", json.dumps(dict(smoke=smoke, training_array=train, root=str(root))), flush=True)


if __name__ == "__main__":
    main()
