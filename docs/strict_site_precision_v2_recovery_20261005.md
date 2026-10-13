# Integer-aware precision approval and conditional recovery v2

Result update 2026-10-06: audit 12925293 processed all 647 rows but FAILED 2:0
on `6fac: repeatability`. All downstream jobs were cancelled without running.
See `site_precision_val_results_12925293.md`. Submission-time states below
are retained as history, not current queue status.

Submitted and verified 2026-10-05, approximately 23:56 scheduler time.
The user explicitly approved implementation and resubmission. This is an
acceptance-policy revision informed by the earlier VAL audit, not a new model,
a pre-registered pass, or evidence of improved predictive performance.

## What changed

- Numerical inference stays `bf16_gvp_fp32_v1`: BF16 forward with FP32 GVP
  scatter accumulation. Checkpoint parameters and mainline model files are
  unchanged by this revision. Full FP32 inference is not substituted.
- Default historical acceptance stays `strict_fraction_v1`.
- The new jobs explicitly select `integer_count_v2`, whose per-row rule is
  `flips <= max(1, n_full // 100)`. Integer arithmetic avoids floating-point
  floor errors. The rule applies uniformly, with no sample-ID, label, or
  F1-improvement exception. Below 100 residues it permits one flip; at 100 or
  more residues it preserves the integer budget implied by the old 1% rule.
- Maximum probability error remains 0.05; mean row disagreement remains at
  most 0.001; absolute mean F1 change remains at most 0.001; repeat probability
  error remains at most 1e-5 with zero repeated binary changes.
- Classification gate remains **0.391169935464859**. Data, label alignment,
  candidate construction, caps, normalization and head configurations stay
  unchanged. No TEST, COACH420 or HOLO4K is accessed.

The old audit 12919239 remains FAILED 2:0; its downstream jobs remain cancelled.
Its files and the incomplete older cache are not overwritten or reused.
The complete original analysis is in `site_precision_val_results_12919239.md`.

## Verification before submission

Re-evaluating the saved 647 comparison rows, without changing any measurement,
produced:

| Policy | Offline verdict on audit 12919239 |
| --- | --- |
| strict_fraction_v1 | FAIL: 2k00, one flip out of 92 residues |
| integer_count_v2 | PASS: all numerical gates |

This offline pass is NOT the outcome of the fresh GPU job. A new full 647-row
VAL audit must produce its own complete approval artifact bound to the exact
source snapshot. The audit records both policy verdicts for transparency.

Local validation: **79 tests passed** across precision approval, repeated
inference, cache loading, all head paths, strict diagnostics and benchmark
integrity. Python compilation and both Slurm syntax checks passed. Tests cover
integer budget boundaries, preservation of the other error gates, inconsistent
counts/fractions, and rejection of mismatched approval/cache policy versions.

Audit approval verifies source/checkpoint/split/overlay/reference hashes,
prediction hashes and the full population. Export rechecks each VAL forward
against both the archived reference and the newly audited probabilities. Cache
NPZs and summaries carry numerical-protocol and acceptance-policy identities;
the shared head loader rejects mixed identities.

## New deployment and job chain

HDD experiment root:
`/hpc2hdd/home/shuang886/PLC/experiments/strict_site_precision_v2_20261005`

Immutable source: `code_v2/` under that root, cloned from the v1 snapshot and
updated only with the six versioned approval/export/loader/launcher files.
No output is written to jhspoolers, and no dirty worktree commit/push was used.

| Job | Stage | Dependency | Verified submission status |
| --- | --- | --- | --- |
| 12925293 | Fresh complete 647-row VAL precision audit | None | PENDING, Priority |
| 12925308 | Small export and all head/control smoke paths | afterok:12925293 | PENDING, Dependency |
| 12925310 | Fresh full TRAIN/VAL shared token cache | afterok:12925308 | PENDING, Dependency |
| 12925312_[0-3%3] | Four matched head/control groups | afterok:12925310 | PENDING, Dependency |

All use **i64m1tga800ue**, one GPU per job/task. Audit requests 2 CPUs/32G/45min;
smoke 4 CPUs/64G/30min; cache 4 CPUs/64G/12h; heads 4 CPUs/64G/4h each.
Array concurrency is capped at three. Downstream jobs have
`--kill-on-invalid-dep=yes`; a failed audit/smoke/export blocks training.

Scheduler initially estimated audit start at 2026-10-06 16:41:00. This is only
a mutable scheduling estimate, not a reservation or completion forecast.

Expected approval: `audit_12925293/summary.json`.
Expected shared cache: `run_12925310_none/cache/`.
Logs: `logs/audit_12925293.*`, `logs/smoke_12925308.*`,
`logs/cache_12925310.*`, and `logs/heads_12925312_<task>.*`.
Head output directories use the actual `SLURM_JOB_ID` and array task ID, which
need not equal the array master job ID.

Array groups: 0 original stats/protein/ligand quality controls; 1 relative gain
with a matched absolute-quality control; 2 token interaction with pooling and
ligand-blind controls; 3 bounded boundary refinement with protein-only control.
Backbone stays frozen; only new heads train from scratch, seed44, at most
20 epochs. Smoke uses two epochs and cannot establish efficacy.

## Deployed SHA256 values

Remote hashes matched local files before submitting the audit:

| File | SHA256 |
| --- | --- |
| scripts/diagnostics/site_inference_precision.py | 2beb63820a915b888436ff5bcf2191e0e7fe0388c752c937bbfeddeb701b4820 |
| scripts/diagnostics/audit_site_precision_val.py | 8895658ac39769dca9797537085fe603d34030780325507b12ca5aa14d036174 |
| scripts/diagnostics/export_site_quality_cache.py | e0ff633ea6ef55cb1c235e378597364534a92d63498656eeb1e858253011eaa9 |
| scripts/train/train_site_quality_probe.py | 2fc3256009cef2867a4d1a48721b3c3cbb83355bf40e516bfd6456f6c6645342 |
| scripts/slurm/run_site_precision_val_audit.slurm | 1b1df37599c9322abdf6929980c00106defa8a20277abd23c97659057f752463 |
| scripts/slurm/run_strict_site_parallel.slurm | ccd7d2a49b734453407813df3cb2c1779097bf9cf22760606732d39697a14d0c |

## Next collection gate

Check fresh audit `complete/passed`, both policy verdicts, exact population and
repeatability before interpreting any smoke. Require the final complete cache
manifest before interpreting full head results. Compare every head against its
matched newly exported baseline and controls; retain contact-count, precision,
recall and replacement-rate guardrails. A passed numerical audit is not an
assignment improvement. No full training or external-test success is claimed
at submission time.
