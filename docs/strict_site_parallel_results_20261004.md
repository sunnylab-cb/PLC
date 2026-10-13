# Strict site-head trials: status and audit results

2026-10-05 follow-up: targeted job 12918127 completed; see
`site_probability_repeat_results_20261005.md` for the BF16/GVP scatter diagnosis.
Submission-time descriptions below are historical, not current queue status.

Checked 2026-10-04 using `ssh hpc2`, Slurm accounting, archived logs and actual
JSON/NPZ outputs. Local evidence: `tmp/strict_site_parallel_20261004/`.
The account queue was empty at the initial inspection. No new job was submitted
during that status check. After the user's follow-up approval, a targeted
diagnostic was submitted as recorded below; no model, dataset, numerical
tolerance or frozen snapshot changed.

## Verified job states

| Job | State / exit | Elapsed | Outcome |
| --- | --- | --- | --- |
| 12868960 | COMPLETED / 0:0 | 00:01:39 | Repeated BF16/FP32 state diagnosis |
| 12869006 | COMPLETED / 0:0 | 00:09:28 | All 10317 TRAIN rows audited; no source QC errors |
| 12869013 | COMPLETED / 0:0 | 00:02:31 | All controls passed execution smoke on TRAIN 68 / VAL 38 |
| 12869014 | FAILED / 1:0 | 02:32:31 | Reference prediction mismatch at VAL sample 5ndf |
| 12869016_[0-3%3] | CANCELLED / 0:0 | 00:00:00 | Dependency cancellation; none of the four full head groups trained |

All termination dates were 2026-09-26. The shared export failed at 22:43:03
scheduler time. This was an explicit ValueError, not an OOM or scheduler timeout.
Its partial output contains 10317 TRAIN and 470 VAL NPZs, including the failing
row. Only TRAIN rows reached `rows.csv`; the final manifest and summary do not
exist. These files are NOT a completed or training-approved cache.

## Numerical diagnosis

On the audited first TRAIN sample (1a5v), identical BF16 forwards differed in
pre-cross protein states by approximately 0.0196; differences were already
present at the GVP output (up to 0.0255). Identical FP32 forwards differed by
approximately 1.4e-6. Zeroing ligand inputs changed FP32 protein states by only
1.4e-6, while changing ligand states substantially. Zeroing ESM inputs left
FP32 ligand states unchanged. The export independence audit passed its original
absolute 1e-5 FP32 bound; BF16 repeat noise is not evidence of cross-modal leakage.
This tests one sample, not a universal invariance proof or a kernel-level root
cause diagnosis.

The new failure is a DIFFERENT gate: reproducing the archived BF16 probabilities.
For 5ndf:

| Measurement | Value |
| --- | ---: |
| Full protein residues | 3426 |
| Original fixed contact gate | 0.391169935464859 |
| Maximum probability difference | 0.051260173320770264 |
| Existing maximum-difference limit | 0.05 |
| Mean absolute probability difference, full axis | 0.0000326998451783 |
| Binary prediction differences | 0 |
| F1 in both exports | 0.3870967741935484 |

At zero-based index 254, probability changed from 0.40277693 to 0.45403710;
both exceed the original gate. Labels, full CA coordinates and observed masks
match exactly. The full-axis mean includes the unobserved, zero-probability tail.

Across the 470 available VAL rows, 25 samples have any binary change (28 residue
flips total), the sample-mean flip fraction is 0.0001261628, and the sample-mean
F1 difference versus the old export is -0.0001046995. Only 5ndf exceeds the 0.05
probability bound. These are incomplete VAL statistics, not a new 647-row result.
This is consistent with small low-precision repeat variation, but repeated
BF16/FP32 predictions on 5ndf itself are still required before assigning a cause.

## TRAIN union audit

| Quantity | Count |
| --- | ---: |
| TRAIN observations | 10317 (5160 PDBbind + 5157 BioLiP) |
| Geometry-compatible groups | 9504 |
| Groups with multiple observations | 658 / 1471 observations |
| Heuristic multisite groups | 633 / 1421 observations (13.77% of TRAIN) |
| Multisite groups with >=2 known chemical identities | 142 / 325 observations |
| Groups gaining positives by union | 636 / 1427 observations |
| Summed union gain over each group's largest observation | 6163 residues |
| Median added residues in groups with positive union gain | 8 |
| Unknown-chemistry observations | 426 |

Of the 1421 multisite observations, 1418 are BioLiP and 3 are PDBbind. The
325 observations in multisite/multi-chemistry groups are all BioLiP, but this
group-level intersection does not establish which pairs are distinct-site,
different-chemistry negatives. It does NOT supersede the prior pairwise audit's
179/10317 (1.74%) strict known-different-chemistry valid-negative coverage.

The union result supports targeted protein-centric positive coverage supervision,
not treating unannotated residues as negative or making union labels into
per-ligand targets. Most geometry groups still have only one observation/site.
No training labels were rewritten. Site grouping is heuristic and is not
biological validation of distinct ligand-binding pockets.

## Smoke results, not efficacy evidence

The smoke used 68 TRAIN / 38 VAL examples for only two epochs (boundary: one
refiner epoch plus one selector epoch). The subset baseline F1 is 0.5000914.
All learned protein/ligand, gain, token and boundary modes selected the original
mask for every VAL sample at the retained checkpoint, so their F1 equals the
baseline. This demonstrates execution and fallback behavior, not improvement or
failure of the proposed architectures. The stats-only control selected worse
masks in 17/38 rows and achieved 0.2992401; it is likewise not a full-training
comparison. Logged gradients are finite, with the boundary refiner gradients
small (approximately 0.00065-0.00069 for this one-epoch smoke).

## Next gate

1. Repeat unchanged-input BF16/FP32 inference on 5ndf and near-limit 4k7o;
   compare probabilities and masks, input/code hashes and precision provenance.
2. Decide a documented reproducibility policy from that evidence. Do not merely
   widen the maximum tolerance after seeing the failing sample. If using a new
   FP32 reference or canonical archived proposal probabilities, explicitly
   version the protocol and re-establish a matched baseline for every branch.
3. Complete and validate a fresh full TRAIN/VAL cache, then resubmit the same four
   dependency-protected head groups. Preserve the failed artifacts for audit.

There is no new full-VAL, strict TEST, COACH420 or HOLO4K head result, and no
evidence here to replace the existing best model.

## Approved targeted follow-up

After the user requested further checking, job **12918127** was submitted on
2026-10-04 at 22:54:13 scheduler time. Verified state: PENDING, Priority; no
dependency, one GPU on i64m1tga800ue, 2 CPUs, 32G RAM, 15-minute time limit.
No full training retry or dependent array was submitted.

The probe repeats each of 5ndf and 4k7o ten times in each condition:
FP32, BF16, and BF16 with GVP scatter accumulation temporarily done in FP32
and cast back to the original dtype. The last condition only replaces the
scatter callable inside the diagnostic process and is always restored. It is
not a mainline inference or training change. Unchanged-input hashes, loader
repeatability, eval-mode checks, RNG changes, pre-cross/GVP/logit differences,
all-pairwise probability differences, observed-only errors, binary flips and
archived-reference differences are recorded. No diagnostic result is yet known.

Static source audit compared the original corrected export snapshot
`strict_assignment_20260919/code_label_fix` against
`strict_site_parallel_20260926/code_pilots_v1`: all 66 reference-side Python files
under src/models, src/data, src/utils plus the V11 inference loader are present
and byte-identical. This rules out drift in those source files, not historical
data/environment changes or numerical backend differences.

Remote diagnostic root:
`/hpc2hdd/home/shuang886/PLC/experiments/site_probability_repeat_20261004`.
Expected output: `run_12918127/report.json` and per-sample probability NPZs;
logs: `logs/repeat_12918127.{out,err}`. The frozen training snapshot is imported
directly rather than overwritten. Deployment SHA256 matched locally:

| File | SHA256 |
| --- | --- |
| diagnose_site_probability_repeat.py | a09a0d924118894ae5eb5c07a06499b6f46e68f0f135c102c3ab7439cea94fdc |
| run_site_probability_repeat.slurm | 42ed43a7399baf2374e78d98c9c539777215805aaf69bd30b67f153084e4a8d4 |

Four new local metric tests passed, including no-flip probability differences,
all-pairwise comparisons, unobserved-tail handling and invalid-input rejection.
Python compilation and Slurm shell syntax passed. Actual CUDA diagnosis remains
pending and should not be inferred from these unit tests.
