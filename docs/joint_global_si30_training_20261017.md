# Expanded Joint Cohort: Strict Split to Scratch Training

## Four-node SI Handoff (2026-10-17 15:30 UTC+8)

At the user's request, SI now runs as **13146925**, on four `i64m512ue`
nodes (`cpu1-[87-90]`), 64 CPUs/160G per node, 256 CPUs total.
Both exhaustive comparisons and the fresh cross-set audit use four shards.
The old single-node job13146709 was cancelled only after preparation13146710
was held and rewired to `afterok:13146925`. Smoke13146711 and training13146712
retain their existing dependencies. GPU training has NOT started.

The last pre-handoff progress snapshot contained308 completed chunks,
6,160,000 comparisons. Valid completed chunks are reused, not deleted.
Scheduler overlay commit **1a45971** keeps original `code_v1` (33acf20) intact,
including the source hash embedded in the numerical SI contract. Chunk ownership
is `chunk_id % 4`; each rank writes separate progress. The original reducer
validates complete enumeration after all workers finish. Tests cover partition
coverage, equality with the original single-node computation, fresh cross-set
audit and cache reuse without rewriting completed chunks:40 tests passed.

Overlay directory: `si_multinode_20261017/code`, relative to the experiment root.
Overlay archive SHA256:
`07e0ebfddd21131da368fd3180088df117d8bcb899f8bf21793acc9c5ec64335`.
This changes scheduling only, not sequence identity, labels, splits or training.
The older single-node estimates below are historical, not the new ETA.

## Original Submission

Submitted **2026-10-17 15:20:39 UTC+8** after user approval to train the expanded
PDBbind + BioLiP cohort. At15:21:06 the SI job is RUNNING on `cpu1-57`; all
downstream jobs are PENDING(Dependency). **GPU training has not started.**

## Frozen Inputs and Protocol

- QC-passed pool: **173,948 samples =154,930 BioLiP +19,018 PDBbind**.
- Exact unique observed chain sequences:65,788; exhaustive unordered pairs:
  **2,163,997,578**, in108,200 resumable chunks of at most20,000.
- Input pool: `biolip_feature_expansion_20261015/qc_pool`, relative to the HPC
  experiments root. Actual sample/sequence counts and their summary SHA256s
  were independently checked after QC completed.
- Pool-summary SHA256:
  `69ae1b5aeac0bc9e4bb081e768b62c81c189ca86e18848d8ffb04a0eb605e8ab`.
- Labels stay native heavy-atom **d_min <5.0 A**, with at least3 actual contact
  residues, including within the existing1,800-residue cap. Existing ligand,
  feature-finiteness and residue-axis QC remain unchanged.

This cohort gets its OWN global-SI graph and whole-connected-component
80/10/10 target split. It does not reuse the PDBbind-only split or its stats.
Same PDB entries and exact sequences are kept together. All cross-set chain
pairs are independently realigned after splitting. The accepted criterion is
**global SI <0.30 for TRAIN/VAL, TRAIN/TEST and VAL/TEST**, matching the latest
PDBbind-only protocol. Local SI is computed as a diagnostic, not a graph edge
or a guarantee of domain-level independence. COACH420/HOLO4K are not used.

QC removed36 repeated metadata observations. Broader chemical/site duplicate
equivalence and scaffold separation are not certified. Same-PDB/sequence
grouping prevents those shared protein observations crossing splits; it does
not eliminate within-set weighting effects. The separately recovered ESM
features remain quarantined and are not silently added to this frozen pool.

## Submitted Chain

| Stage | Job | Partition | Allocation | Gate |
| --- | --- | --- | --- | --- |
| Exhaustive SI + fresh cross-set audit | 13146709 | i64m512ue | 64 CPUs/160G,7d | Whole-component split; zero global violations |
| TRAIN-only stats + materialization + loader QC | 13146710 | i64m512ue | 8 CPUs/160G,2d | Verified pool/split/labels/stats and exact loader coverage |
| GPU smoke | 13146711 | i64m1tga800ue | 1 GPU/8 CPUs/100G,2h | FLA check, real mixed-source TRAIN backward and VAL forward |
| Scratch seed44 training | 13146712 | i64m1tga800ue | 1 GPU/8 CPUs/100G,7d | Matching successful smoke and frozen provenance |

All links use `afterok` and `--kill-on-invalid-dep=yes`. Failure does not bypass
SI, normalization or smoke. No GPU is allocated while CPU preparation is
pending. The SI cache supports resuming completed chunks under the same
contract; a scheduler timeout would require inspecting/resubmitting the chain,
not treating partial SI as a certificate. Submission receipts prevent blind
duplicate jobs. No old experiment was cancelled or modified.

Old64-worker smaller-cohort SI runs achieved approximately13.5-14.7k pairs/s.
If that transferred, the new all-pairs phase alone would take roughly41-45h;
cross-set audit, preprocessing and queues are additional. This is a provisional
scale estimate, not measured throughput on the new65,788-sequence cohort.
No exact training start/completion time or final split counts are known yet.

**15:22:39 check:** the actual new stage has completed39 chunks/780,000 pairs
in100.46s of measured stage time,7,764 pairs/s, with empty stderr. At this
startup rate the all-pairs phase alone would be about77h, slower than the
old-cohort extrapolation. This first100s is not steady-state throughput; use
longer windows before committing to an ETA. The1,037 within-pool graph edges
are expected homology connections, NOT cross-set leakage violations. The
410 local-only matches remain diagnostic. All downstream jobs still wait.

## Model and Verification

Use the audited **V11m13 group_rescompat + GFGF**, seed44, random initialization,
fully unfrozen, with no resume or compile. Model/training settings are unchanged
from the PDBbind-only reference: co-attention, local54 residue-chem compatibility,
fractional protein gradient0.6, batch16/BF16, AdamW1e-4, existing loss/CNC,
max120 epochs/patience30, max_atoms256 alternating-ends crop and max_seq_len1800.
Existing best-F1/AUPRC/AUROC checkpoints and VAL gate diagnostics remain intact.
Smoke weights are discarded. No TEST forwarding or selection happens in setup.

Data/config/output paths and experiment name are the only recipe changes.
Source-aware TRAIN-key validation replaces the PDBbind-only prefix assumption.
Loader checks and GPU smoke explicitly cover both sources when available;
split counts are retained by source. Subsequent VAL/TEST reporting must retain
source-specific metrics; this is not a matched-holdout expansion comparison
with the independently split PDBbind-only run.

48 focused tests passed, including exhaustive/fresh SI audit, TRAIN-only stats,
production-loader end-to-end, source membership/collision guards, unchanged
recipe and idempotent submission. Slurm syntax checks passed. Optional local
PyG CUDA-extension warnings do not certify GPU support: the queued HPC smoke
is the actual GPU gate.

Immutable source **33acf20**,570 hash-verified files; no model/trainer changes.
The first archive was rejected before submission because it contained old
clustering-temporary symlinks. The replacement contains code/configs only;
no job was submitted from the rejected archive.

Archive SHA256:
`240bf3e5f778b7174cc279e24e39f3d2d94f5b620563a0e106d3c11dff5aa649`.
Reference-config SHA256:
`c58d7a8d4dd1a1840232e5cda9b648b62085a93da7dc14c00753e3ac5944fa69`.

Experiment root:
`/hpc2hdd/home/shuang886/PLC/experiments/joint_global_si30_gfgf_20261017`.
Retain `deployment.json`, `submission.json`, `split/`, `dataset/`, `prepared/`,
`smoke_pass.json`, `logs/` and `outputs/`. The prepared ESM paths and dependency
directory refer to existing caches; those backing assets must not be deleted.
