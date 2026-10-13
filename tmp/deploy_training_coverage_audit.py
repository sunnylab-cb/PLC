"""Deploy only the isolated coverage audit and its small provenance inputs."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

BASE = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--phase", choices=("coverage", "known-ligands"), default="coverage")
args = ap.parse_args()
ROOT = "/hpc2hdd/home/shuang886/PLC/experiments/training_coverage_audit_20261013"
SLURM = "scripts/slurm/run_training_coverage_audit.slurm"
LOCAL_RECORD = "deployment.json"
FILES = [
    "scripts/diagnostics/audit_pdbbind_coverage.py",
    "scripts/diagnostics/audit_biolip_expansion.py",
    "scripts/diagnostics/audit_biolip_expansion_tensors.py",
    "scripts/slurm/run_training_coverage_audit.slurm",
    "preparation/dataset_split/mmseqs_splits_strict.txt",
    "configs/splits/splits.txt",
    "configs/splits/excluded_samples.txt",
    "tmp/pdbbind_coverage_20261013/local_inventory.json",
]
if args.phase == "known-ligands":
    ROOT = "/hpc2hdd/home/shuang886/PLC/experiments/biolip_known_chain_expansion_20261013"
    SLURM = "scripts/slurm/run_biolip_known_chain_expansion.slurm"
    LOCAL_RECORD = "known_ligand_deployment.json"
    FILES = [
        "scripts/diagnostics/extract_biolip_known_chain_expansion.py",
        "scripts/diagnostics/audit_biolip_expansion.py",
        "scripts/diagnostics/audit_biolip_expansion_tensors.py", SLURM,
    ] + [str(p.relative_to(BASE)) for p in sorted((BASE/"preparation/ligand").glob("*.py"))]
hashes = {name: hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in FILES}
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as tar:
    for name in FILES:
        tar.add(BASE/name, arcname=name, recursive=False)
code = "import hashlib,io,json,os,subprocess,sys,tarfile\nfrom pathlib import Path\n"
code += f"root=Path({ROOT!r})\nexpected={hashes!r}\nslurm={SLURM!r}\n"
code += """
root.mkdir(parents=True,exist_ok=False)
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
    members=tar.getmembers()
    if {m.name for m in members} != set(expected) or any(not m.isfile() for m in members):
        raise ValueError('Archive mismatch')
    for member in members:
        data=tar.extractfile(member).read()
        if hashlib.sha256(data).hexdigest() != expected[member.name]:
            raise ValueError(member.name)
        path=root/member.name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
command=['/opt/slurm/bin/sbatch','--parsable','--chdir='+str(root),
         '--export=ALL,EXPERIMENT='+str(root),'--output='+str(root/'qc_%j.out'),
         '--error='+str(root/'qc_%j.err'),str(root/slurm)]
submitted=subprocess.run(command,capture_output=True,text=True,check=True)
job=submitted.stdout.strip().split(';')[0]
if not job.isdigit():
    raise ValueError(submitted.stdout)
report=dict(root=str(root),job_id=job,command=command,sha256=expected)
(root/'deployment.json').write_text(json.dumps(report,indent=2)+'\\n')
print('__COVERAGE_SUBMISSION__')
print(json.dumps(report))
"""
result = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "hpc2",
    "/hpc2ssd/softwares/anaconda3/envs/af3/bin/python -c " + shlex.quote(code)],
    input=buf.getvalue(), capture_output=True, check=True, timeout=100)
report=json.loads(result.stdout.decode().split("__COVERAGE_SUBMISSION__\n",1)[1])
(BASE/"tmp/pdbbind_coverage_20261013"/LOCAL_RECORD).write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
