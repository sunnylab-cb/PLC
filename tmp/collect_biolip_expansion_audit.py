"""Fetch small BioLiP audit results and historical provenance, not tensors."""

import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

DEST = Path(__file__).resolve().parent / "biolip_expansion_20261013"
BIO = "/hpc2hdd/home/shuang886/biolip"
EXP = "/hpc2hdd/home/shuang886/PLC/experiments/biolip_expansion_audit_20261013"
FINAL = BIO + "/pdbbind_chain_si30_split_11283646_final"
FILES = {
    "summary.json": EXP + "/summary.json",
    "audit.out": EXP + "/audit_13001698.out",
    "audit.err": EXP + "/audit_13001698.err",
    "old_metadata_summary.json": BIO + "/qc_training_manifest_allclean_20260811/summary.json",
    "old_sequence_summary.json": BIO + "/qc_ligand_candidates_sequence_clean_20260811/summary.json",
    "old_ligand_tensor_qc.json": BIO + "/qc_ligand_candidates_sequence_clean_20260811/ligand_v6_extraction_qc.json",
    "old_materialized_summary.json": BIO + "/v11_allclean_route5_20260811/summary.json",
    "final_repair_summary.json": FINAL + "/repair_summary.json",
    "final_leakage_summary.json": FINAL + "/final_allowlist_audit/summary.json",
}
code = "import base64,hashlib,json,subprocess\nfrom pathlib import Path\nfiles=" + repr(FILES) + "\n"
code += """
result = {}
for name, path in files.items():
    data = Path(path).read_bytes()
    if len(data) > 2 * 1024 ** 2:
        raise ValueError(path)
    result[name] = dict(path=path, sha256=hashlib.sha256(data).hexdigest(), data=base64.b64encode(data).decode())
sacct = subprocess.run(['/opt/slurm/bin/sacct','-j','13001698','--format=JobID,Partition,State,ExitCode,Elapsed,MaxRSS','-P'],capture_output=True,text=True,check=True).stdout
print('__BIO_EXPANSION_JSON__')
print(json.dumps(dict(files=result,sacct=sacct)))
"""
result = subprocess.run(["ssh", "hpc2", "/hpc2ssd/softwares/anaconda3/envs/af3/bin/python -c " + shlex.quote(code)],
                        capture_output=True, text=True, check=True, timeout=90)
payload = json.loads(result.stdout.split("__BIO_EXPANSION_JSON__\n", 1)[1])
DEST.mkdir(exist_ok=True)
for name, entry in payload["files"].items():
    data = base64.b64decode(entry.pop("data"), validate=True)
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    (DEST / name).write_bytes(data)
(DEST / "sync_manifest.json").write_text(json.dumps(payload["files"], indent=2) + "\n")
(DEST / "sacct.txt").write_text(payload["sacct"])
print(payload["sacct"])
summary = json.loads((DEST / "summary.json").read_text())
print(json.dumps(summary, indent=2))
