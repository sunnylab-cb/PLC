"""Collect small completed-run artifacts despite remote shell stdout banners."""

import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


REMOTE = "/hpc2hdd/home/shuang886/PLC/experiments/context_noise_gfgf_20261012"
DEST = Path(__file__).resolve().parent / "context_noise_20261013"
MARKER = "__PLC_NOISE_ARTIFACTS_JSON__"
code = r'''
import base64, hashlib, json
from pathlib import Path
root = Path(ROOT)
paths = list(root.glob("*.json"))
paths += list((root / "logs").glob("*.out"))
paths += list((root / "logs").glob("*.err"))
paths += list((root / "prepared").glob("*.json"))
paths += list((root / "prepared").glob("*.yaml"))
for run in sorted((root / "outputs").iterdir()):
    if run.is_dir():
        paths += [p for p in run.iterdir() if p.suffix in (".yaml", ".json", ".log", ".txt")]
        paths += list(run.glob("runs/events.out.tfevents.*"))
paths = sorted(set(p for p in paths if p.is_file() and not p.is_symlink()))
total = sum(p.stat().st_size for p in paths)
if total > 100 * 1024 ** 2:
    raise RuntimeError(f"Unexpected small-artifact size: {total}")
files = []
for path in paths:
    data = path.read_bytes()
    files.append(dict(path=str(path.relative_to(root)), size=len(data),
                      sha256=hashlib.sha256(data).hexdigest(),
                      data=base64.b64encode(data).decode("ascii")))
print(MARKER)
print(json.dumps(dict(root=str(root), total_bytes=total, files=files)))
'''
code = f"ROOT={REMOTE!r}\nMARKER={MARKER!r}\n" + code
command = "/hpc2ssd/softwares/anaconda3/envs/af3/bin/python -c " + shlex.quote(code)
result = subprocess.run(["ssh", "hpc2", command], capture_output=True, check=True, timeout=180)
prefix, encoded = result.stdout.decode().split(MARKER + "\n", 1)
payload = json.loads(encoded)
DEST.mkdir(exist_ok=True)
manifest = []
for entry in payload["files"]:
    relative = Path(entry["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(relative)
    data = base64.b64decode(entry.pop("data"), validate=True)
    if len(data) != entry["size"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
        raise ValueError(f"Hash mismatch: {relative}")
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    manifest.append(entry)
(DEST / "sync_manifest.json").write_text(json.dumps(dict(
    remote=REMOTE, total_bytes=payload["total_bytes"], files=manifest,
    ignored_shell_banner=prefix), indent=2) + "\n")
accounting = subprocess.run(["ssh", "hpc2",
    "/opt/slurm/bin/sacct -j 12990623,12990624 "
    "--format=JobID,JobName%30,Partition,State,ExitCode,Elapsed,Start,End,NodeList,MaxRSS -P"],
    capture_output=True, check=True, text=True, timeout=60)
(DEST / "sacct.txt").write_text(accounting.stdout)
print(json.dumps(dict(destination=str(DEST), files=len(manifest), bytes=payload["total_bytes"])))
