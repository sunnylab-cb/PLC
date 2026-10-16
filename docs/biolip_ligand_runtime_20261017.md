# BioLiP Ligand Extraction Runtime, October 17

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

No running extraction code, conformer policy, job allocation or dependency was
changed. No cancellation, restart or duplicate extraction task was submitted.

## ESM Recovery Is No Longer Running

Jobs13124029/30/31 all completed successfully on October16, ending17:45:26.
The audit found5,079 recoverable chain axes and298 unresolved; all5,079 ESM
outputs succeeded. Combined scheduled-chain cache coverage is110,763/111,061
(99.73%). New outputs remain quarantined pending V4/geometry/axis recovery;
they are not an expanded training certificate. See
[ESM recovery ledger](biolip_esm2_recovery_20261016.md).
