#!/usr/bin/env python3
"""Download pinned official LiBRe assets on HPC; never execute upstream code.

Only Python source, README, example CSVs and the released checkpoint are fetched.
ESM2 is linked from the existing HPC cache. No environments or jobs are created.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from datetime import datetime, timezone


REPO = "GoldRiver0/LiBRe"
API = "https://api.github.com/repos/" + REPO


def fetch(url, destination):
    subprocess.run(
        ["curl", "-fL", "--connect-timeout", "20", "--max-time", "180",
         "--retry", "3", "--retry-delay", "3", "-sS", "-o", str(destination), url],
        check=True,
    )


def api_json(url):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "response.json"
        fetch(url, path)
        return json.loads(path.read_text())


def hashes(path):
    sha256 = hashlib.sha256()
    blob = hashlib.sha1()
    blob.update(("blob %d\0" % path.stat().st_size).encode())
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha256.update(chunk)
            blob.update(chunk)
    return sha256.hexdigest(), blob.hexdigest()


def link_existing(target, link):
    if link.is_symlink() and link.resolve() == target.resolve():
        return
    if link.exists() or link.is_symlink():
        raise RuntimeError("Refusing to replace existing path: " + str(link))
    link.symlink_to(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--esm-cache", type=Path, required=True)
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()
    root = args.root.resolve()
    for name in ("source", "checkpoints", "features", "manifests", "logs", "results"):
        (root / name).mkdir(parents=True, exist_ok=True)
    commit = api_json(API + "/commits/" + args.revision)["sha"]
    tree = api_json(API + "/git/trees/" + commit + "?recursive=1")
    if tree.get("truncated"):
        raise RuntimeError("Incomplete upstream file listing")
    source = root / "source" / commit
    checkpoint = root / "checkpoints" / commit / "model.pt"
    records = []
    print("Pinned upstream commit: " + commit, flush=True)
    for entry in tree["tree"]:
        name = entry["path"]
        if entry["type"] != "blob" or not (
            name == "README.md" or name == "models/model.pt"
            or (name.startswith(("src/", "data/"))
                and name.endswith((".py", ".csv"))
                and not any(part.startswith(".") for part in Path(name).parts))
        ):
            continue
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise RuntimeError("Unsafe upstream path")
        dest = checkpoint if name == "models/model.pt" else source / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            partial = dest.with_name(dest.name + ".part")
            fetch("https://raw.githubusercontent.com/" + REPO + "/" + commit + "/" + name, partial)
            if partial.stat().st_size != entry["size"] or hashes(partial)[1] != entry["sha"]:
                raise RuntimeError("Downloaded file failed verification: " + name)
            os.replace(partial, dest)
        sha256, blob = hashes(dest)
        if blob != entry["sha"]:
            raise RuntimeError("Existing asset differs from upstream: " + str(dest))
        records.append({"upstream_path": name, "local_path": str(dest),
                        "bytes": dest.stat().st_size, "git_blob_sha1": blob, "sha256": sha256})
        print("Verified: %s (%d bytes)" % (name, dest.stat().st_size), flush=True)
    if not checkpoint.exists():
        raise RuntimeError("No released checkpoint found")
    (source / "models").mkdir(exist_ok=True)
    link_existing(checkpoint, source / "models" / "model.pt")
    esm_records = []
    esm_dir = root / "checkpoints" / "esm2"
    esm_dir.mkdir(exist_ok=True)
    for suffix in (".pt", "-contact-regression.pt"):
        cached = args.esm_cache / ("esm2_t33_650M_UR50D" + suffix)
        if not cached.is_file():
            raise FileNotFoundError("Required existing ESM asset not found: " + str(cached))
        link_existing(cached.resolve(), esm_dir / cached.name)
        digest, _ = hashes(cached)
        esm_records.append({"path": str(cached), "bytes": cached.stat().st_size, "sha256": digest})
        print("Reused existing ESM asset: " + str(cached), flush=True)
    manifest = {"repository": "https://github.com/" + REPO, "commit": commit,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "assets": records, "esm2_reused": esm_records,
                "upstream_code_executed": False, "checkpoint_loaded": False,
                "jobs_submitted": False}
    manifest_path = root / "manifests" / (commit + ".json")
    partial = manifest_path.with_suffix(".json.part")
    partial.write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(partial, manifest_path)
    print("Manifest: " + str(manifest_path), flush=True)


if __name__ == "__main__":
    main()
