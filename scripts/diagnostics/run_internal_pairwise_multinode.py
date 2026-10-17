#!/usr/bin/env python3
"""Slurm multi-node scheduling around the immutable original SI implementation.

Keep its source hash and numerical contract unchanged so completed chunks can
be reused. Each rank owns chunk_number % world_size, including on resume.
The original reducer independently validates all chunks and audit pair counts.
"""

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid


def load_legacy(code):
    code = Path(code).resolve()
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(code / "scripts/diagnostics"))
    module = importlib.import_module("run_internal_pairwise_split")
    if Path(module.__file__).resolve() != code / "scripts/diagnostics/run_internal_pairwise_split.py":
        raise ValueError("Wrong legacy SI module imported")
    return module


def stage_contract(base, labels):
    result = dict(base, labels_sha256=hashlib.sha256(json.dumps(labels).encode()).hexdigest())
    result["hash"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    return result


def shard_tasks(n, chunk_size, rank, world):
    if not 0 <= rank < world or chunk_size < 1:
        raise ValueError("Invalid shard coordinates")
    total = n * (n - 1) // 2
    return [(k, start, min(start + chunk_size, total))
            for k, start in enumerate(range(0, total, chunk_size)) if k % world == rank]


def worker(request_path, rank, world, workers):
    request = json.loads(Path(request_path).read_text())
    core = load_legacy(request["legacy_code"])
    if (world != request["world"] or workers != request["workers_per_node"]
            or core.digest(Path(core.__file__)) != request["legacy_sha256"]
            or core.digest(Path(__file__)) != request["driver_sha256"]):
        raise ValueError("Worker code/allocation differs from controller request")
    out, contract = Path(request["out"]), request["contract"]
    if json.loads((out / "contract.json").read_text()) != contract:
        raise ValueError("Stage contract changed")
    seqs, labels = request["sequences"], request["labels"]
    assigned = shard_tasks(len(seqs), request["chunk_size"], rank, world)
    counts = dict(chunks=0, evaluated=0, same_split_skipped=0, edges=0, local_only_matches=0)

    def consume(record):
        counts["chunks"] += 1
        for key in counts:
            if key != "chunks":
                counts[key] += record.get(key, 0)

    pending = []
    for task in assigned:
        k, start, stop = task
        cached = core.check_saved_chunk(out / f"chunk_{k:07d}.json", contract["hash"], start, stop)
        if cached:
            consume(cached)
        else:
            pending.append(task)
    cached_evaluated = counts["evaluated"]
    started = time.monotonic()

    def report(done=False):
        data = dict(**counts, rank=rank, world=world, hostname=os.uname().nodename,
                    pid=os.getpid(), invocation=request["invocation"], done=done,
                    assigned_chunks=len(assigned), cached_evaluated=cached_evaluated,
                    fresh_evaluated=counts["evaluated"] - cached_evaluated,
                    wall_seconds=time.monotonic() - started)
        core.save_json(out / f"rank_{rank:02d}_progress.json", data)
        print("SI_RANK_PROGRESS", json.dumps(data), flush=True)

    report()
    with ProcessPoolExecutor(max_workers=workers, initializer=core.init_worker,
                             initargs=(seqs, labels, contract, str(out))) as pool:
        iterator = iter(pending)
        futures = set()
        for task in iterator:
            futures.add(pool.submit(core.work_chunk, task))
            if len(futures) >= 2 * workers:
                break
        last_report = time.monotonic()
        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                consume(future.result())
                task = next(iterator, None)
                if task is not None:
                    futures.add(pool.submit(core.work_chunk, task))
            if time.monotonic() - last_report >= 30:
                report()
                last_report = time.monotonic()
    if (counts["chunks"] != len(assigned)
            or counts["evaluated"] + counts["same_split_skipped"] != sum(b-a for _, a, b in assigned)):
        raise RuntimeError("Incomplete rank enumeration")
    report(done=True)


def distributed_stage(core, original, code, nodes, seqs, labels, base, out, workers, chunk_size):
    out.mkdir(parents=True, exist_ok=True)
    contract = stage_contract(base, labels)
    path = out / "contract.json"
    if path.exists() and json.loads(path.read_text()) != contract:
        raise ValueError("Cannot reuse a changed SI contract")
    core.save_json(path, contract)
    invocation = uuid.uuid4().hex
    request = dict(legacy_code=str(code), legacy_sha256=core.digest(Path(core.__file__)),
                   driver_sha256=core.digest(Path(__file__)), sequences=seqs, labels=labels,
                   out=str(out.resolve()), contract=contract, chunk_size=chunk_size,
                   world=nodes, workers_per_node=workers, invocation=invocation)
    request_path = out / "distributed_request.json"
    core.save_json(request_path, request)
    command = ["/opt/slurm/bin/srun", f"--nodes={nodes}", f"--ntasks={nodes}",
               "--ntasks-per-node=1", f"--cpus-per-task={workers}", "--cpu-bind=cores",
               "--kill-on-bad-exit=1", "--label", sys.executable, "-u", str(Path(__file__).resolve()),
               "worker", "--request", str(request_path)]
    print("DISTRIBUTED_STAGE", out.name, json.dumps(command), flush=True)
    started = time.monotonic()
    total = len(seqs) * (len(seqs) - 1) // 2

    def aggregate():
        reports = []
        for rank in range(nodes):
            p = out / f"rank_{rank:02d}_progress.json"
            if p.exists():
                r = json.loads(p.read_text())
                if r["invocation"] == invocation:
                    reports.append(r)
        wall = time.monotonic() - started
        counters = {k: sum(r[k] for r in reports) for k in
                    ("chunks", "evaluated", "same_split_skipped", "edges", "local_only_matches")}
        core.save_json(out / "progress.json", dict(stage=out.name, **counters, nodes=nodes,
            workers_per_node=workers, ranks_reporting=len(reports), invocation=invocation,
            total_pairs=total, total_chunks=(total + chunk_size - 1) // chunk_size,
            wall_seconds_this_invocation=wall,
            fresh_pairs_per_second=sum(r["fresh_evaluated"] for r in reports) / max(wall, 1e-6)))

    process = subprocess.Popen(command)
    try:
        while True:
            try:
                result = process.wait(timeout=30)
                aggregate()
                break
            except subprocess.TimeoutExpired:
                aggregate()
        if result:
            raise RuntimeError(f"Distributed SI workers failed: exit {result}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            process.wait()
        raise
    # Never let the serial reducer silently recompute a missing shard.
    for rank in range(nodes):
        report = json.loads((out / f"rank_{rank:02d}_progress.json").read_text())
        if report["invocation"] != invocation or not report["done"]:
            raise RuntimeError(f"Rank {rank} did not finish this invocation")
    for k, start, stop in shard_tasks(len(seqs), chunk_size, 0, 1):
        if core.check_saved_chunk(out / f"chunk_{k:07d}.json", contract["hash"], start, stop) is None:
            raise RuntimeError(f"Missing completed chunk {k}")
    return original(seqs, labels, base, out, workers, chunk_size)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    commands = ap.add_subparsers(dest="mode", required=True)
    w = commands.add_parser("worker")
    w.add_argument("--request", type=Path, required=True)
    c = commands.add_parser("controller")
    c.add_argument("--experiment", type=Path, required=True)
    c.add_argument("--pool-dir", type=Path, required=True)
    c.add_argument("--nodes", type=int, default=4)
    c.add_argument("--workers", type=int, default=64)
    args = ap.parse_args()
    if args.mode == "worker":
        worker(args.request, int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"]),
               int(os.environ["SLURM_CPUS_PER_TASK"]))
        return
    if (int(os.environ.get("SLURM_JOB_NUM_NODES", "0")) != args.nodes
            or int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) != args.workers):
        raise ValueError("Controller must run within the requested multi-node Slurm allocation")
    code = args.experiment.resolve() / "code_v1"
    core = load_legacy(code)
    deployment = json.loads((args.experiment / "deployment.json").read_text())
    for relative, expected in deployment["source_sha256"].items():
        if core.digest(code / relative) != expected:
            raise ValueError(f"Original source changed: {relative}")
    if core.digest(args.pool_dir / "pool_summary.json") != deployment["pool_sha256"]:
        raise ValueError("Pool changed after deployment")
    original = core.run_stage
    core.run_stage = lambda seqs, labels, base, out, workers, chunk: distributed_stage(
        core, original, code, args.nodes, seqs, labels, base, out, workers, chunk)
    sys.argv = [core.__file__, "--pool-dir", str(args.pool_dir), "--output-dir",
                str(args.experiment / "split"), "--edge-rule", "global_only",
                "--workers", str(args.workers), "--chunk-size", "20000", "--seed", "42"]
    core.main()


if __name__ == "__main__":
    main()
