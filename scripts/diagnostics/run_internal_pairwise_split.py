#!/usr/bin/env python3
"""Exhaustive Biopython SI graph -> connected split -> fresh cross-set audit.

No search prefilter, representative clustering, external dataset or old split
is consulted. Every unordered pair of exact unique input-chain sequences is
aligned. Fixed BLOSUM62 affine-gap global SI >=.30 OR local SI >=.30 with either
coverage >=.80 creates an edge. The first score-optimal alignment in canonical
SHA256 sequence order defines SI, not the maximum SI over all possible/tied
alignments. Version and scoring policy are pinned in each run's contract.

This certifies an observed-input-sequence split, not pretraining contamination,
ligand scaffold independence, or label/normalization readiness for training.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

from prepare_internal_codataset_sequences import ALPHABET, digest, save_json


SCHEMA = 1
SPLITS = ("train", "val", "test")
_STATE = None


class UnionFind:
    def __init__(self, values):
        self.parent = {x: x for x in values}
        self.size = dict.fromkeys(self.parent, 1)

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        x, y = self.find(x), self.find(y)
        if x != y:
            if self.size[x] < self.size[y]:
                x, y = y, x
            self.parent[y] = x
            self.size[x] += self.size[y]


def make_aligners():
    import numpy as np
    from Bio.Align import PairwiseAligner, substitution_matrices
    base = substitution_matrices.load("BLOSUM62")
    alphabet = "".join(sorted(ALPHABET))
    aliases = {"U": "C", "O": "K", "J": "X"}
    scores = np.array([[base[aliases.get(a, a), aliases.get(b, b)] for b in alphabet]
                       for a in alphabet])
    matrix = substitution_matrices.Array(alphabet=alphabet, dims=2, data=scores)
    result = []
    for mode in ("global", "local"):
        aligner = PairwiseAligner()
        aligner.substitution_matrix = matrix
        aligner.open_gap_score = -10.0
        aligner.extend_gap_score = -0.5
        aligner.mode = mode
        aligner.epsilon = 1e-6
        result.append(aligner)
    return result


def alignment_stats(alignment, left, right):
    # Original letters, not substitution-score aliases, define exact identity.
    coords = alignment.coordinates
    identical = columns = consumed_left = consumed_right = 0
    for k in range(coords.shape[1] - 1):
        a, b = int(coords[0, k]), int(coords[1, k])
        na, nb = int(coords[0, k + 1]), int(coords[1, k + 1])
        da, db = na-a, nb-b
        if da < 0 or db < 0 or (da and db and da != db):
            raise ValueError("invalid protein alignment coordinates")
        columns += max(da, db)
        consumed_left += da
        consumed_right += db
        if da and db:
            identical += sum(x == y for x, y in zip(left[a:na], right[b:nb]))
    if columns <= 0:
        raise ValueError("empty alignment")
    return (identical / columns, consumed_left / len(left), consumed_right / len(right))


def pair_stats(left, right, aligners):
    values = []
    for aligner in aligners:
        alignments = aligner.align(left, right)
        try:
            aln = next(iter(alignments))
        except StopIteration:
            values.append((0.0, 0.0, 0.0))
        else:
            values.append(alignment_stats(aln, left, right))
    return values[0][0], *values[1]


def is_edge(stats, identity=.30, coverage=.80):
    global_si, local_si, qcov, tcov = stats
    return global_si >= identity or (local_si >= identity and max(qcov, tcov) >= coverage)


def row_start(i, n):
    return i*(2*n-i-1)//2


def pair_at(k, n):
    if not 0 <= k < n*(n-1)//2:
        raise IndexError(k)
    # Avoid floating-point inverse triangular-number rounding for large pools.
    lo, hi = 0, n-1
    while lo+1 < hi:
        mid = (lo+hi)//2
        if row_start(mid, n) <= k:
            lo = mid
        else:
            hi = mid
    return lo, lo+1+k-row_start(lo, n)


def iter_pairs(start, stop, n):
    if start == stop:
        return
    i, j = pair_at(start, n)
    for _ in range(start, stop):
        yield i, j
        j += 1
        if j == n:
            i, j = i+1, i+2


def init_worker(sequences, labels, contract, directory):
    global _STATE
    _STATE = (sequences, labels, contract, Path(directory), make_aligners())


def check_saved_chunk(path, contract_hash, start, stop):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    if (data["contract_hash"], data["start"], data["stop"]) != (contract_hash, start, stop):
        raise ValueError(f"chunk contract mismatch: {path}")
    edge_file = path.with_suffix(".edges.tsv.gz")
    if digest(edge_file) != data["edge_sha256"]:
        raise ValueError(f"chunk edges missing/corrupt: {edge_file}")
    if data["evaluated"] + data["same_split_skipped"] != stop-start:
        raise ValueError(f"incomplete chunk: {path}")
    return data


def work_chunk(task):
    number, start, stop = task
    sequences, labels, contract, out, aligners = _STATE
    path = out / f"chunk_{number:07d}.json"
    cached = check_saved_chunk(path, contract["hash"], start, stop)
    if cached:
        return cached
    started = time.monotonic()
    edges = path.with_suffix(".edges.tsv.gz")
    tmp = edges.with_suffix(edges.suffix + ".tmp")
    evaluated = skipped = violations = 0
    histogram = [0] * 101
    max_global = max_covered_local = 0.0
    pair_counts = Counter()
    with gzip.open(tmp, "wt", compresslevel=1) as f:
        for i, j in iter_pairs(start, stop, len(sequences)):
            if labels is not None and labels[i] == labels[j]:
                skipped += 1
                continue
            stats = pair_stats(sequences[i], sequences[j], aligners)
            g, local, qc, tc = stats
            if not all(math.isfinite(x) and 0 <= x <= 1 for x in stats):
                raise ValueError(f"invalid alignment stats: {i}, {j}, {stats}")
            evaluated += 1
            if labels is not None:
                pair_counts["_vs_".join(sorted((labels[i], labels[j])))] += 1
            histogram[min(100, int(g*100))] += 1
            max_global = max(max_global, g)
            if max(qc, tc) >= contract["coverage"]:
                max_covered_local = max(max_covered_local, local)
            if is_edge(stats, contract["identity"], contract["coverage"]):
                violations += 1
                f.write("\t".join(map(str, (i, j, *stats))) + "\n")
    tmp.replace(edges)
    result = dict(contract_hash=contract["hash"], start=start, stop=stop,
        evaluated=evaluated, same_split_skipped=skipped, edges=violations,
        global_si_histogram_1pct=histogram, max_global_si=max_global,
        max_local_si_at_required_coverage=max_covered_local, by_set_pair=dict(pair_counts),
        edge_sha256=digest(edges), elapsed_seconds=time.monotonic()-started)
    save_json(path, result)
    return result


def run_stage(sequences, labels, base_contract, out, workers, chunk_size):
    out.mkdir(parents=True, exist_ok=True)
    contract = dict(base_contract, labels_sha256=hashlib.sha256(json.dumps(labels).encode()).hexdigest())
    contract["hash"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    contract_path = out / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("cannot resume with a changed alignment contract")
    save_json(contract_path, contract)
    total = len(sequences)*(len(sequences)-1)//2
    tasks = [(k, start, min(start+chunk_size, total))
             for k, start in enumerate(range(0, total, chunk_size))]
    started = time.monotonic()
    counts, pair_counts = Counter(), Counter()
    hist, maxima = [0]*101, [0.0, 0.0]
    def consume(r):
        for name in ("evaluated", "same_split_skipped", "edges"):
            counts[name] += r[name]
        counts["chunks"] += 1
        pair_counts.update(r["by_set_pair"])
        for k, value in enumerate(r["global_si_histogram_1pct"]):
            hist[k] += value
        maxima[0] = max(maxima[0], r["max_global_si"])
        maxima[1] = max(maxima[1], r["max_local_si_at_required_coverage"])
    pending = []
    for task in tasks:
        k, start, stop = task
        saved = check_saved_chunk(out/f"chunk_{k:07d}.json", contract["hash"], start, stop)
        if saved:
            consume(saved)
        else:
            pending.append(task)
    print("PAIRWISE_STAGE", out.name, "unique", len(sequences), "all_pairs", total,
          "chunks_completed", counts["chunks"], "chunks_remaining", len(pending), flush=True)
    initial_evaluated = counts["evaluated"]
    def progress():
        wall = time.monotonic()-started
        fresh = counts["evaluated"]-initial_evaluated
        report = dict(stage=out.name, **counts, total_pairs=total, total_chunks=len(tasks),
            wall_seconds_this_invocation=wall, fresh_pairs_per_second=fresh/max(wall, 1e-6))
        save_json(out/"progress.json", report)
        print("PAIRWISE_PROGRESS", json.dumps(report), flush=True)
    # Bounded scheduling avoids constructing millions of futures for large pools.
    with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
            initargs=(sequences, labels, contract, str(out))) as pool:
        iterator = iter(pending)
        futures = {}
        for task in iterator:
            futures[pool.submit(work_chunk, task)] = task
            if len(futures) >= 2*workers:
                break
        last_report = 0.0
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                futures.pop(future)
                consume(future.result())
                task = next(iterator, None)
                if task is not None:
                    futures[pool.submit(work_chunk, task)] = task
            if time.monotonic()-last_report >= 30:
                progress()
                last_report = time.monotonic()
    if counts["evaluated"]+counts["same_split_skipped"] != total or counts["chunks"] != len(tasks):
        raise RuntimeError("incomplete pair enumeration")
    if labels is not None:
        ns = Counter(labels)
        expected = {"_vs_".join(sorted((a, b))): ns[a]*ns[b]
                    for i, a in enumerate(SPLITS) for b in SPLITS[i+1:]}
        if dict(pair_counts) != expected:
            raise RuntimeError(f"cross-set audit count mismatch: {pair_counts} != {expected}")
    summary = dict(contract_hash=contract["hash"], **counts, all_unordered_pairs=total,
        by_set_pair=dict(pair_counts), global_si_histogram_1pct=hist, max_global_si=maxima[0],
        max_local_si_at_required_coverage=maxima[1])
    save_json(out/"summary.json", summary)
    progress()
    return summary, tasks


def assign_components(samples, uf, seed):
    groups = defaultdict(list)
    for key in sorted(samples):
        groups[uf.find(key)].append(key)
    components = list(groups.values())
    if len(components) < len(SPLITS):
        raise ValueError("fewer than three separable components; refusing to split a family")
    random.Random(seed).shuffle(components)
    components.sort(key=len, reverse=True)
    counts = Counter()
    targets = dict(zip(SPLITS, (len(samples)*.8, len(samples)*.1, len(samples)*.1)))
    assignment = {}
    for index, group in enumerate(components):
        empty = [s for s in SPLITS if counts[s] == 0]
        # Reserve the last components for empty sets without breaking a family.
        choices = empty if len(components)-index == len(empty) else SPLITS
        label = max(choices, key=lambda s: targets[s]-counts[s])
        assignment.update((key, label) for key in group)
        counts[label] += len(group)
    if set(counts) != set(SPLITS):
        raise ValueError("fewer than three separable components; refusing to split a family")
    return assignment, components


def load_pool(path):
    pool_summary = json.loads((path/"pool_summary.json").read_text())
    for name, sha in pool_summary["output_sha256"].items():
        if digest(path/name) != sha:
            raise ValueError(f"pool file corrupt: {name}")
    samples = {}
    for line in (path/"samples.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["key"] in samples:
            raise ValueError("duplicate sample key")
        samples[row["key"]] = row
    seqs = [json.loads(line) for line in (path/"sequences.jsonl").read_text().splitlines()]
    seqs.sort(key=lambda r: r["sequence_id"])
    ids = [r["sequence_id"] for r in seqs]
    if len(ids) != len(set(ids)) or len({r["sequence"] for r in seqs}) != len(seqs):
        raise ValueError("sequence pool is not exact deduplicated")
    if len(seqs) < 3:
        raise ValueError("sequence pool too small")
    expected_members = defaultdict(set)
    for key, row in samples.items():
        if not row["sequence_ids"]:
            raise ValueError(f"sample has no sequence: {key}")
        for sid in row["sequence_ids"]:
            expected_members[sid].add(key)
    if set(expected_members) != set(ids):
        raise ValueError("sample/sequence index differs")
    for row in seqs:
        if set(row["members"]) != expected_members[row["sequence_id"]]:
            raise ValueError("sample/sequence reverse index differs")
        seq = row["sequence"]
        if not seq or set(seq)-ALPHABET or row["sequence_id"] != "seq_"+hashlib.sha256(seq.encode()).hexdigest():
            raise ValueError("invalid sequence content/hash")
    return samples, seqs, pool_summary


def main():
    import Bio
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--chunk-size", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        ap.error("workers and chunk size must be positive")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    samples, records, pool = load_pool(args.pool_dir)
    sequences = [r["sequence"] for r in records]
    contract = dict(schema=SCHEMA, biopython_version=Bio.__version__, identity=.30, coverage=.80,
        matrix="BLOSUM62", rare_letter_score_aliases={"U":"C", "O":"K", "J":"X"},
        identity_letters="original letters including ambiguous-letter equality; no sequence remapping",
        gap_open=-10.0, gap_extend=-.5, terminal_gaps="penalized in global",
        epsilon=1e-6, tie_policy="first optimal alignment; ascending sequence SHA256 orientation",
        predicate="global_SI>=0.30 OR (local_SI>=0.30 AND max(qcov,tcov)>=0.80)",
        identity_denominator="alignment columns including internal/terminal gaps as returned per mode",
        coverage_definition="consumed original residues / full original chain length",
        prefilter="none; all unique unordered pairs", seed=args.seed, chunk_size=args.chunk_size,
        cohort=pool["cohort"], pool_sha256=digest(args.pool_dir/"pool_summary.json"),
        script_sha256=digest(Path(__file__)))
    save_json(out/"status.json", dict(stage="all_pairs", passed=False, training_approved=False))
    all_result, tasks = run_stage(sequences, None, contract, out/"all_pairs", args.workers, args.chunk_size)
    uf = UnionFind(samples)
    pdb_first = {}
    for key, sample in samples.items():
        pdb = sample["pdb_id"]
        uf.union(key, pdb_first.setdefault(pdb, key))
    for row in records:
        members = row["members"]
        for member in members[1:]:
            uf.union(members[0], member)
    edge_count = 0
    for k, _, _ in tasks:
        with gzip.open(out/"all_pairs"/f"chunk_{k:07d}.edges.tsv.gz", "rt") as f:
            for line in f:
                fields = line.split("\t")
                i, j = map(int, fields[:2])
                if not (0 <= i < j < len(records)) or not is_edge(tuple(map(float, fields[2:]))):
                    raise ValueError("invalid graph edge")
                uf.union(records[i]["members"][0], records[j]["members"][0])
                edge_count += 1
    if edge_count != all_result["edges"]:
        raise ValueError("graph edge accounting mismatch")
    assignment, components = assign_components(samples, uf, args.seed)
    labels = []
    for row in records:
        seen = {assignment[key] for key in row["members"]}
        if len(seen) != 1:
            raise ValueError("exact sequence leaked across splits")
        labels.append(next(iter(seen)))
    counts = Counter(assignment.values())
    by_source = {source: dict(Counter(assignment[key] for key, s in samples.items() if s["source"] == source))
                 for source in sorted({s["source"] for s in samples.values()})}
    component_ids = {key: index for index, group in enumerate(components) for key in group}
    with (out/"candidate_split.tsv").open("w") as f:
        f.write("split\tsample_key\tsource\tsample_id\tpdb_id\tcomponent_id\n")
        for key in sorted(samples):
            row = samples[key]
            f.write("\t".join(map(str, (assignment[key], key, row["source"], row["sample_id"],
                                      row["pdb_id"], component_ids[key])))+"\n")
    save_json(out/"status.json", dict(stage="independent_cross_set_recalculation", passed=False,
        counts=dict(counts), by_source=by_source, training_approved=False))
    audit, _ = run_stage(sequences, labels, contract, out/"cross_set_audit", args.workers, args.chunk_size)
    passed = audit["edges"] == 0
    report = dict(passed=passed, training_approved=False, stage="si_complete", cohort=pool["cohort"],
        contract=contract, counts=dict(counts), by_source=by_source, unique_sequences=len(records),
        components=len(components), largest_components=sorted(map(len, components), reverse=True)[:20],
        actual_fractions={s:counts[s]/len(samples) for s in SPLITS},
        balance_within_3_percentage_points=all(abs(counts[s]/len(samples)-f) <= .03
                                              for s, f in zip(SPLITS,(.8,.1,.1))),
        split_sha256=digest(out/"candidate_split.tsv"), all_pairs=all_result, cross_set_audit=audit,
        external_benchmarks_used=False,
        training_pending=["per-pair feature/label geometry and residue-axis QC",
            "cross-source duplicate observation resolution within each frozen component",
            "separate TRAIN-only normalization for this cohort; no old all-data stats",
            "fresh random initialization; no checkpoint trained on old split"],
        limitations=["SI is defined by the fixed optimal-score alignment policy, not all possible alignments",
            "Observed input chains, not an audit of PLM pretraining or unseen full-length sequences",
            "Two independent cohorts have different holdouts: do not attribute score differences solely to BioLiP"])
    save_json(out/"summary.json", report)
    save_json(out/"status.json", report)
    if not passed:
        raise RuntimeError("cross-set SI audit failed; do not use this split")
    print("SPLIT_SI_PASSED", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
