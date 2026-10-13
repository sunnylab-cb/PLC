"""Collect small coverage/QC reports using hash-checked SSH transport."""

import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

DEST=Path(__file__).resolve().parent/"pdbbind_coverage_20261013"
ROOT="/hpc2hdd/home/shuang886/PLC/experiments/training_coverage_audit_20261013"
FILES={"pdbbind_coverage.json":ROOT+"/pdbbind_coverage.json",
       "pdbbind_coverage.candidates.tsv":ROOT+"/pdbbind_coverage.candidates.tsv",
       "biolip_tensor_qc_summary.json":ROOT+"/biolip_tensor_qc/summary.json",
       "audit.out":ROOT+"/qc_13003040.out", "audit.err":ROOT+"/qc_13003040.err"}
EXTRACT="/hpc2hdd/home/shuang886/PLC/experiments/biolip_known_chain_expansion_20261013"
FILES.update({"known_ligand_summary.json":EXTRACT+"/prepared/summary.json",
              "known_ligand_provenance.json":EXTRACT+"/prepared/provenance.json",
              "known_ligand_extract.out":EXTRACT+"/qc_13003132.out"})
code="import base64,hashlib,json,subprocess\nfrom pathlib import Path\nfiles="+repr(FILES)+"\n"
code+='''
result={}
for name,path in files.items():
    data=Path(path).read_bytes()
    if len(data)>8*1024**2:
        raise ValueError(path)
    result[name]=dict(path=path,sha256=hashlib.sha256(data).hexdigest(),data=base64.b64encode(data).decode())
acc=subprocess.run(['/opt/slurm/bin/sacct','-j','13003040,13003132','--format=JobID,State,ExitCode,Elapsed,MaxRSS','-P'],capture_output=True,text=True,check=True).stdout
print('__COVERAGE_RESULTS__')
print(json.dumps(dict(files=result,sacct=acc)))
'''
res=subprocess.run(["ssh","-o","BatchMode=yes","hpc2",
    "/hpc2ssd/softwares/anaconda3/envs/af3/bin/python -c "+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=90)
payload=json.loads(res.stdout.split("__COVERAGE_RESULTS__\n",1)[1])
DEST.mkdir(exist_ok=True)
for name,r in payload["files"].items():
    data=base64.b64decode(r.pop("data"),validate=True)
    if hashlib.sha256(data).hexdigest()!=r["sha256"]:
        raise ValueError(name)
    (DEST/name).write_bytes(data)
(DEST/"sync_manifest.json").write_text(json.dumps(payload["files"],indent=2)+"\n")
(DEST/"sacct.txt").write_text(payload["sacct"])
print(payload["sacct"])
print(json.dumps(payload["files"],indent=2))
