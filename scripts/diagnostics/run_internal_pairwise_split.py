#!/usr/bin/env python3
"""Exhaustive Biopython SI graph -> connected split -> fresh cross-set audit.

No search prefilter, representative clustering, external dataset or old split
is consulted. Every unordered pair of exact unique input-chain sequences is
aligned. The default edge is global SI >=.30 OR covered local SI >=.30.
The explicit global_only rule retains local SI as a diagnostic, not an edge.
Completed combined-rule caches can supply the global-edge subset, but the
cross-set audit always recomputes alignments. The first optimal alignment in canonical
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
PREDICATES = {
    "global_or_local": "global_SI>=0.30 OR (local_SI>=0.30 AND max(qcov,tcov)>=0.80)",
    "global_only": "global_SI>=0.30; local SI is diagnostic only",
}
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


def is_edge(stats, identity=.30, coverage=.80, rule="global_or_local"):
    global_si, local_si, qcov, tcov = stats
    if rule == "global_only":
        return global_si >= identity
    if rule != "global_or_local":
        raise ValueError(f"unknown SI edge rule: {rule}")
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
    evaluated = skipped = violations = local_only = 0
    histogram = [0] * 101
    max_global = max_covered_local = 0.0
    pair_counts = Counter()
    pair_maxima = {}
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
                pair_name = "_vs_".join(sorted((labels[i], labels[j])))
                pair_counts[pair_name] += 1
                pair_maxima[pair_name] = max(pair_maxima.get(pair_name, 0.), g)
            histogram[min(100, int(g*100))] += 1
            max_global = max(max_global, g)
            if max(qc, tc) >= contract["coverage"]:
                max_covered_local = max(max_covered_local, local)
            local_only += int(g < contract["identity"] and local >= contract["identity"]
                              and max(qc, tc) >= contract["coverage"])
            if is_edge(stats, contract["identity"], contract["coverage"],
                       contract.get("edge_rule", "global_or_local")):
                violations += 1
                f.write("\t".join(map(str, (i, j, *stats))) + "\n")
    tmp.replace(edges)
    result = dict(contract_hash=contract["hash"], start=start, stop=stop,
        evaluated=evaluated, same_split_skipped=skipped, edges=violations,
        global_si_histogram_1pct=histogram, max_global_si=max_global,
        max_local_si_at_required_coverage=max_covered_local, by_set_pair=dict(pair_counts),
        local_only_matches=local_only, by_set_pair_max_global_si=pair_maxima,
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
    pair_maxima = {}
    hist, maxima = [0]*101, [0.0, 0.0]
    def consume(r):
        for name in ("evaluated", "same_split_skipped", "edges", "local_only_matches"):
            counts[name] += r.get(name, 0)
        counts["chunks"] += 1
        pair_counts.update(r["by_set_pair"])
        for name, value in r.get("by_set_pair_max_global_si", {}).items():
            pair_maxima[name] = max(pair_maxima.get(name, 0.), value)
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
        by_set_pair_max_global_si=pair_maxima,
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


def verify_reusable_all_pairs(path, source_script, contract, n_sequences):
    """Validate an exhaustive, combined-rule superset without changing its files."""
    old = json.loads((path/"contract.json").read_text())
    hashed = {k:v for k,v in old.items() if k != "hash"}
    if hashlib.sha256(json.dumps(hashed, sort_keys=True).encode()).hexdigest() != old["hash"]:
        raise ValueError("source alignment contract hash mismatch")
    if digest(source_script) != old["script_sha256"]:
        raise ValueError("source alignment implementation hash mismatch")
    if contract["edge_rule"] != "global_only" or old["predicate"] != PREDICATES["global_or_local"]:
        raise ValueError("only combined-to-global cache reuse is supported")
    for key, value in contract.items():
        if key not in {"script_sha256", "predicate", "edge_rule"} and old.get(key) != value:
            raise ValueError(f"source alignment contract differs: {key}")
    if old["labels_sha256"] != hashlib.sha256(json.dumps(None).encode()).hexdigest():
        raise ValueError("source is not an exhaustive all-pairs cache")
    summary = json.loads((path/"summary.json").read_text())
    total = n_sequences*(n_sequences-1)//2
    tasks = [(k, start, min(start+old["chunk_size"], total))
             for k, start in enumerate(range(0, total, old["chunk_size"]))]
    if (summary["contract_hash"] != old["hash"] or summary["evaluated"] != total
            or summary["same_split_skipped"] != 0 or summary["all_unordered_pairs"] != total
            or summary["chunks"] != len(tasks) or summary["by_set_pair"]):
        raise ValueError("incomplete or incompatible source all-pairs summary")
    if len(list(path.glob("chunk_*.json"))) != len(tasks):
        raise ValueError("source chunk set differs from exhaustive enumeration")
    evaluated = edges = 0
    histogram = [0]*101
    for k, start, stop in tasks:
        chunk = check_saved_chunk(path/f"chunk_{k:07d}.json", old["hash"], start, stop)
        if chunk is None or chunk["same_split_skipped"] or chunk["by_set_pair"]:
            raise ValueError("incomplete source all-pairs chunk")
        evaluated += chunk["evaluated"]
        edges += chunk["edges"]
        if len(chunk["global_si_histogram_1pct"]) != 101:
            raise ValueError("invalid source SI histogram")
        histogram = [a+b for a,b in zip(histogram, chunk["global_si_histogram_1pct"])]
    if (evaluated != total or edges != summary["edges"] or sum(histogram) != total
            or histogram != summary["global_si_histogram_1pct"]):
        raise ValueError("source chunk accounting differs from summary")
    return summary, tasks, dict(directory=str(path.resolve()),
        contract_sha256=digest(path/"contract.json"), summary_sha256=digest(path/"summary.json"),
        source_script_sha256=digest(source_script), source_contract_hash=old["hash"],
        note="global edges filtered from complete combined-rule cache; cross-set audit recomputed")


def main():
    import Bio
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument("--chunk-size", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--edge-rule", choices=tuple(PREDICATES), default="global_or_local")
    ap.add_argument("--reuse-all-pairs", type=Path)
    ap.add_argument("--reuse-source-script", type=Path)
    args = ap.parse_args()
    if args.workers < 1 or args.chunk_size < 1:
        ap.error("workers and chunk size must be positive")
    if bool(args.reuse_all_pairs) != bool(args.reuse_source_script):
        ap.error("cache reuse requires both --reuse-all-pairs and --reuse-source-script")
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    samples, records, pool = load_pool(args.pool_dir)
    sequences = [r["sequence"] for r in records]
    contract = dict(schema=SCHEMA, biopython_version=Bio.__version__, identity=.30, coverage=.80,
        matrix="BLOSUM62", rare_letter_score_aliases={"U":"C", "O":"K", "J":"X"},
        identity_letters="original letters including ambiguous-letter equality; no sequence remapping",
        gap_open=-10.0, gap_extend=-.5, terminal_gaps="penalized in global",
        epsilon=1e-6, tie_policy="first optimal alignment; ascending sequence SHA256 orientation",
        predicate=PREDICATES[args.edge_rule], edge_rule=args.edge_rule,
        identity_denominator="alignment columns including internal/terminal gaps as returned per mode",
        coverage_definition="consumed original residues / full original chain length",
        prefilter="none; all unique unordered pairs", seed=args.seed, chunk_size=args.chunk_size,
        cohort=pool["cohort"], pool_sha256=digest(args.pool_dir/"pool_summary.json"),
        script_sha256=digest(Path(__file__)))
    save_json(out/"status.json", dict(stage="all_pairs", passed=False, training_approved=False))
    edge_dir = args.reuse_all_pairs or out/"all_pairs"
    if args.reuse_all_pairs:
        if edge_dir.resolve() == (out/"all_pairs").resolve():
            raise ValueError("use a new split directory; never overwrite the source cache")
        all_result, tasks, provenance = verify_reusable_all_pairs(
            edge_dir, args.reuse_source_script, contract, len(sequences))
        contract["all_pairs_reuse"] = provenance
        save_json(out/"all_pairs_reuse.json", provenance)
    else:
        all_result, tasks = run_stage(sequences, None, contract, edge_dir, args.workers, args.chunk_size)
    uf = UnionFind(samples)
    pdb_first = {}
    for key, sample in samples.items():
        pdb = sample["pdb_id"]
        uf.union(key, pdb_first.setdefault(pdb, key))
    for row in records:
        members = row["members"]
        for member in members[1:]:
            uf.union(members[0], member)
    edge_count = selected_edges = 0
    for k, start, stop in tasks:
        previous_pair = start-1
        chunk_edges = 0
        with gzip.open(edge_dir/f"chunk_{k:07d}.edges.tsv.gz", "rt") as f:
            for line in f:
                fields = line.split("\t")
                i, j = map(int, fields[:2])
                stats = tuple(map(float, fields[2:]))
                source_rule = "global_or_local" if args.reuse_all_pairs else args.edge_rule
                if (len(stats) != 4 or not all(math.isfinite(v) and 0 <= v <= 1 for v in stats)
                        or not (0 <= i < j < len(records)) or not is_edge(stats, rule=source_rule)):
                    raise ValueError("invalid graph edge")
                pair_index = row_start(i, len(records))+j-i-1
                if not start <= pair_index < stop or pair_index <= previous_pair:
                    raise ValueError("duplicate, unordered or out-of-chunk graph edge")
                previous_pair = pair_index
                if is_edge(stats, rule=args.edge_rule):
                    uf.union(records[i]["members"][0], records[j]["members"][0])
                    selected_edges += 1
                edge_count += 1
                chunk_edges += 1
        if chunk_edges != json.loads((edge_dir/f"chunk_{k:07d}.json").read_text())["edges"]:
            raise ValueError("chunk edge line count mismatch")
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
        graph_edges=selected_edges, source_cache_edges=edge_count,
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
            "global_only does not guarantee local/domain SI independence; local SI remains diagnostic",
            "Observed input chains, not an audit of PLM pretraining or unseen full-length sequences",
            "Two independent cohorts have different holdouts: do not attribute score differences solely to BioLiP"])
    save_json(out/"summary.json", report)
    save_json(out/"status.json", report)
    if not passed:
        raise RuntimeError("cross-set SI audit failed; do not use this split")
    print("SPLIT_SI_PASSED", json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
