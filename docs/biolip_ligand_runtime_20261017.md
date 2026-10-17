# BioLiP Ligand Extraction Runtime, October 17

## Completion Verified

Checked **2026-10-17 15:08 UTC+8**. Replacement13133479 actually started
02:48:48 on `cpu1-42`, stopped its predecessor at02:48:52, waited for exit/lock
release, and invoked the original extractor at02:48:58. It COMPLETED0:0 at
12:08:01, elapsed **9h19m13s**, peak RSS **364.07/448 GiB**, without OOM.
The stage summary reports **149,860/149,870 files**,10 failures and extractor
return code1 (`partial_or_failed`). All10 are K6G SDF molecule-read failures
across7wu3/7wu4/7wu5/8srx, not scheduler or memory failures. Blind reruns are
not an input repair. The first manifest remains32,432/32,432 complete.

The resumed log includes skipped existing files, so149,870 divided by runtime
is NOT new-output throughput. Before takeover the old log had processed
117,904 rows; actual persisted-file counts at cancellation were not separately
snapshotted. The new log reached149,850 at6h57m02s but the last20 results took
another2h20m44s. This long tail explains why the earlier6-8h estimate was
optimistic. Slurm total CPU time430.49h is72.17% of the64-core allocation's
capacity over9h19m; this does not establish uniform worker efficiency or a
controlled4x speedup across different chemistry mixes.

QC13077847 ran12:08:02-12:33:04 (25m02s), COMPLETED0:0. The output is
**154,930 BioLiP +19,018 PDBbind =173,948 QC-passed pairs**,65,788 unique
sequences. Actual JSONL counts and SHA256s match the QC summary. QC failures
include17,646 non-route5 ligand features and14,146 missing protein inputs;
the independently recovered ESM features are not silently promoted.

The subsequent user-approved joint SI/preprocess/smoke/scratch-training chain
is recorded in [the new experiment ledger](joint_global_si30_training_20261017.md).
No expanded TRAIN/test certificate exists yet. Earlier queue/ETA observations
below are historical, not the current task state.

## 64-Worker Replacement Submitted

At **2026-10-17 02:16:51 UTC+8**, submitted **13133479** to **i64m512ue**:
one node,64 CPUs,448 GiB RAM,48-hour limit. The user requested64 workers;
the original16-worker allocation was conservative, not an extractor limit.
The follow-up queue check reports **PENDING (Resources)**. The16-worker
predecessor13077843 remains RUNNING while the replacement waits for a node.

The new allocation runs a guarded takeover: verify original source hashes,
old job identity and QC dependency, cancel only13077843, wait for its Slurm
exit AND released stage lock, then invoke the immutable972c36f extractor with
64 single-thread workers. Completed atomic `.pt` files are skipped; unfinished
pairs and missing failed outputs are retried. No overwrite, conformer-count,
RDKit-thread, timeout, feature-definition or admission-policy change is made.

QC13077847 was held during dependency rewiring, then released with
`afterok:13133479`. A failed takeover therefore does not launch premature QC.
Extractor-level partial outcomes remain explicitly recorded by the existing
stage summary and final QC. No expanded training set is certified yet.

Source **2cea12e** is pushed to `plc/master`;26 focused tests and Bash syntax
validation pass. Original extraction code and the remote production checkout
are untouched. The source archive SHA256 is
`b6ac430c0c0adae88faa78eaf565ca023b9713f58361f7ffb82d316f4255adb8`.
Deployment receipt: experiment-relative
`ligand_resume64_20261017/deployment.json`; takeover receipt will be
`ligand_resume64_13133479.json`. Logs are
`logs/ligand_resume64_13133479.{out,err}`; the extraction log gains a
`RESUME64` marker before the resumed progress stream.

**Provisional compute ETA after allocation:6-8h**, based on four times the
workers and the remaining approximately33k pairs. This is extrapolation, not
a measured64-worker throughput result. Queue time is additional; the old job
continues reducing the remaining work while queued. Shared I/O, memory use,
worker startup and chemistry mix can prevent linear scaling. The prior
16-worker estimate below remains historical, not the new job's completion ETA.

## Verified Progress

Check time **2026-10-17 01:54:57 UTC+8**. Job13077843 has run36h12m10s
since October15 13:42:47 on `cpu1-71`, `i64m512ue`,16 CPUs/128G. It is
RUNNING with zero restarts, not suspended or OOM. Its seven-day limit has not
been reached. The extraction log was updated three seconds before the check.

- Processed116,788/149,870 pairs (77.93%):116,785 successful,3 failed.
- Remaining33,082 pairs; output count is not final geometry/QC admission.
- First ligand job13077842 already completed32,432/32,432.
- Final QC13077847 remains PENDING Dependency, waiting for13077843.

## Runtime Diagnosis

Read-only process inspection through `srun --jobid=13077843 --overlap` found
all16 pool workers RUNNING at99.8-99.9% CPU, each accumulating about36 CPU
hours. The parent waits for worker results normally. Slurm's aggregate batch
CPU time is24d02h24m57s, about99.86% of16-core capacity over the elapsed time.
Reported memory is100,589,460 KiB (95.93 GiB), below the128 GiB allocation.
This supports CPU-bound chemistry work, not an idle allocation or I/O stall.

The immutable deployed command uses16 processes, one RDKit thread each,
50 requested conformers and a600-second Python timeout. The expensive path
includes ETKDGv3 conformer generation and
`MMFFOptimizeMoleculeConfs(..., maxIters=100000)` in
`preparation/ligand/ligand_features_v4.py`. Work is per pair, with no
cross-pair molecular/conformer cache. Existing complete files can be skipped,
but repeat ligand identities in new pairs are not automatically deduplicated.
The timeout is not evidence of a hard wall-clock bound for a native C++ call.

A deterministic approximately1% SHA256 sample of the second manifest, grouped
by output-file modification time, shows a marked change in completed work:

| Completion bucket | Sampled outputs | Heavy atoms median / p90 | PEB ligand ID |
| --- | ---: | ---: | ---: |
| First8h | 571 | 23 / 40 | 16/571 (2.8%) |
| Latest8h | 89 | 43 / 43 | 56/89 (62.9%) |
| Files not yet present | 319 | 28 / 43 | 12/319 (3.8%) |

This is consistent with a more expensive/repetitive segment of chemistry.
It does not establish per-molecule timings or attribute all extra time to PEB
or MMFF: the current multi-worker progress log does not retain successful
per-pair timing or fallback details. Three failures are counted, but their
detailed failure ledger is written when the pool completes. The unfinished
sample looks smaller than recent completed work, so throughput may recover.

## ETA

Rates are reconstructed from completed-count/elapsed-time records in the full
tqdm log, not its highly variable instantaneous display ETA.

| Window | Pairs/hour | Remaining hours if that rate persists |
| --- | ---: | ---: |
| Latest1h | 1,306 | 25.33 |
| Latest2h | 1,304 | 25.37 |
| Latest4h | 1,307 | 25.31 |
| Latest8h | 1,230 | 26.89 |
| Latest12h | 1,434 | 23.07 |
| Latest24h | 2,212 | 14.96 |
| Whole run | 3,227 | 10.25 |

The operational estimate is **another23-27h**, approximately **October18
01:00-05:00 UTC+8**, conditional on the recent slower regime persisting. This
is not a statistical confidence interval or a guaranteed upper bound. Faster
chemistry later in the queue could finish earlier; slow native calls could
finish later. The earlier12-20h estimate was optimistic after the composition
and throughput shift. QC/queue time is additional and has not been estimated.

At the first01:54 check, no running extraction code, conformer policy, job
allocation or dependency was changed. The later user-requested64-worker
resubmission is recorded above.

## ESM Recovery Is No Longer Running

Jobs13124029/30/31 all completed successfully on October16, ending17:45:26.
The audit found5,079 recoverable chain axes and298 unresolved; all5,079 ESM
outputs succeeded. Combined scheduled-chain cache coverage is110,763/111,061
(99.73%). New outputs remain quarantined pending V4/geometry/axis recovery;
they are not an expanded training certificate. See
[ESM recovery ledger](biolip_esm2_recovery_20261016.md).
