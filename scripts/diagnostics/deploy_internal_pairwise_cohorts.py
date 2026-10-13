#!/usr/bin/env python3
"""Deploy an immutable, hash-verified CPU pipeline and submit two i64 jobs."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host",default="hpc2")
    ap.add_argument("--remote-root",required=True)
    ap.add_argument("--record",type=Path,required=True)
    ap.add_argument("--submit",action="store_true",help="Without this flag only print the source manifest")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    files = ["scripts/diagnostics/"+name for name in (
        "prepare_internal_codataset_sequences.py","internal_cohort_qc.py","internal_contact_geometry.py",
        "smoke_internal_contact_labels.py",
        "run_internal_pairwise_split.py","preprocess_internal_cohort.py")]
    files += ["src/contact_labels.py", "scripts/slurm/run_internal_pairwise_cohort.slurm",
              "preparation/dataset_split/mmseqs_splits_strict.txt"]
    hashes = {name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in files}
    if not args.submit:
        print(json.dumps(dict(remote_root=args.remote_root,sha256=hashes),indent=2))
        return
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive,mode="w:gz") as tar:
        for name in files:
            tar.add(repo/name,arcname=name,recursive=False)
    remote = "import hashlib,io,json,subprocess,sys,tarfile\nfrom pathlib import Path\n"
    remote += f"root=Path({args.remote_root!r})\nexpected={hashes!r}\n"
    remote += '''
root.mkdir(parents=True,exist_ok=False)
code=root/'code'
with tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode='r:gz') as tar:
    members=tar.getmembers()
    if {m.name for m in members} != set(expected) or any(not m.isfile() for m in members):
        raise ValueError('Unexpected archive member')
    for member in members:
        payload=tar.extractfile(member).read()
        if hashlib.sha256(payload).hexdigest()!=expected[member.name]:
            raise ValueError(member.name)
        path=code/member.name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(payload)
import Bio,torch,scipy,gemmi,rdkit
report=dict(root=str(root),sha256=expected,biopython=Bio.__version__,jobs={})
(root/'deployment.json').write_text(json.dumps(report,indent=2)+'\\n')
sys.path.insert(0,str(code/'scripts/diagnostics'))
from smoke_internal_contact_labels import run_smoke
report['contact_smoke']=run_smoke('/hpc2hdd/home/shuang886/jhspoolers/PLC/data',
                                '/hpc2hdd/home/shuang886/biolip',root/'contact_smoke',contact_rule='fixed5')
(root/'deployment.json').write_text(json.dumps(report,indent=2)+'\\n')
for cohort,name in [('pdbbind_only','pdb_5a_si30'),('pdbbind_biolip_joint','joint_5a_si30')]:
    out=root/cohort
    out.mkdir()
    cmd=['/opt/slurm/bin/sbatch','--parsable','--job-name='+name,'--chdir='+str(code),
         '--export=ALL,EXPERIMENT='+str(root)+',COHORT='+cohort,
         '--output='+str(out/'pipeline_%j.out'),'--error='+str(out/'pipeline_%j.err'),
         str(code/'scripts/slurm/run_internal_pairwise_cohort.slurm')]
    result=subprocess.run(cmd,capture_output=True,text=True,check=True)
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():
        raise ValueError(result.stdout)
    report['jobs'][cohort]=dict(job_id=job,command=cmd)
    (root/'deployment.json').write_text(json.dumps(report,indent=2)+'\\n')
print('__INTERNAL_SI_SUBMISSION__')
print(json.dumps(report))
'''
    result = subprocess.run(["ssh","-o","BatchMode=yes","-o","ConnectTimeout=20",args.host,
        "/hpc2ssd/softwares/anaconda3/envs/af3/bin/python -c "+shlex.quote(remote)],
        input=archive.getvalue(),capture_output=True,timeout=180)
    if result.returncode:
        raise RuntimeError(result.stdout.decode()+result.stderr.decode()+
                           "\nInspect remote deployment.json before retrying; a job may have submitted.")
    report = json.loads(result.stdout.decode().split("__INTERNAL_SI_SUBMISSION__\n",1)[1])
    args.record.parent.mkdir(parents=True,exist_ok=True)
    args.record.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
