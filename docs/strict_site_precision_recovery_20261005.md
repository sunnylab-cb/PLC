# Strict site-head recovery with audited GVP accumulation

Follow-up: the full audit completed all 647 rows but failed its original count
gate on 2k00 (one changed prediction among 92 residues). All downstream jobs were
cancelled before execution. See `site_precision_val_results_12919239.md`.
The submission states below are historical.

Date: 2026-10-05. User approved continuation after the completed two-sample
numerical diagnosis. This records implementation and submission, not a passed
full-VAL audit or training result.

## Submitted chain

| Job | Role | Verified state |
| --- | --- | --- |
| 12919239 | Full strict VAL precision audit, all 647 rows | RUNNING on gpu1-25 at final follow-up; model loaded, no initial traceback |
| 12919240 | Small token export and all head/control smoke | PENDING, afterok:12919239 |
| 12919243 | Fresh TRAIN 10317 / VAL 647 shared cache | PENDING, afterok:12919240 |
| 12919245_[0-3%3] | Original four matched head groups | PENDING, afterok:12919243 |

All use i64m1tga800ue, one GPU per job/task. Array concurrency is capped at three.
Smoke, export and array have `KillOInInvalidDependent=Yes`, verified via scontrol.
Time limits are audit 45 minutes, smoke 30 minutes, export 12 hours, heads 4 hours
each. These are caps, not measured runtime estimates. The audit requests 2 CPUs
and 32G RAM; other stages request 4 CPUs and 64G. No Star resources are used.

Array 0: stats_only / protein_site / ligand_site quality controls.
Array 1: original-relative gain / parameter-matched paired absolute quality.
Array 2: token late interaction / matched mean pooling / ligand-blind control.
Array 3: ligand-aware / protein-only boundary refinement plus quality selectors.
All downstream method, seed44, fixed gate, LR, epoch and selection settings
remain those of the approved 20260926 pilot. This is not a new architecture sweep.

## Numerical protocol

`legacy_bf16` remains the default. The explicit new protocol
`bf16_gvp_fp32_v1` promotes only low-precision inputs to the GVP `scatter` mean
reduction to FP32, then casts the result back. It uses an exception-safe context
in a single-threaded export process and rejects gradient-enabled calls. Model
source files and checkpoint parameters are unchanged. No torch.compile, new
normalization, new gate, truncation change, resume or external benchmark access.

The full audit runs a fresh legacy BF16 forward and two mixed-precision forwards
for each VAL row. It checks input fingerprints before/after inference, source
labels, full CA/residue axes, observed masks and the archived gate. The first
mixed forward also undergoes the existing strict FP32 pre-cross independence
check. Every numerical outlier is collected rather than aborting on the first.
Fatal provenance/axis/nonfinite errors still abort with a partial diagnostic.

Existing acceptance limits are unchanged:

| Gate versus archived BF16 reference | Limit |
| --- | ---: |
| Per-row maximum probability difference | <=0.05 |
| Per-row binary disagreement fraction, full axis | <=0.01 |
| Mean row disagreement fraction | <=0.001 |
| Absolute sample-mean VAL F1 difference | <=0.001 |

An additional repeatability gate requires mixed-protocol repeated probabilities
to differ by at most 1e-5 and zero binary flips. The current legacy comparison is
reported separately; its stochastic variation does not redefine the old limits.
Audit metrics include sample/protein mean F1, pooled precision/recall/F1 and
predicted-contact count quantiles for archived, fresh legacy and mixed inference.
All F1 metrics use the true aligned contact labels, not reference predictions.

Passing requires ALL 647 rows and aggregate gates, not merely the earlier two
diagnostic samples. A failed audit writes its numerical findings and exits 2;
the remaining chain must not run. Full FP32 inference is not substituted.

## Cache integrity

New-protocol export requires the passed full-audit summary, verifies matching
checkpoint/split/label-overlay/reference/gate and code hashes, rechecks numerical
acceptance, and verifies every audited prediction hash. Each VAL input/reference
must match its audit fingerprint. Re-exported mixed probabilities must reproduce
the audited mixed probabilities within 1e-5 with identical binary predictions.
The original historical-reference and aggregate checks also remain active.

Every cache NPZ and its summary records the inference protocol. The training
loader rejects missing or inconsistent row protocols whenever that metadata is
present. Old cache format remains readable for historical runs/tests. TRAIN and
VAL are regenerated together; the failed 7.5G partial legacy cache is not reused
or deleted. Only a complete manifest is accepted by full head training.

## Deployment and verification

Root: `/hpc2hdd/home/shuang886/PLC/experiments/strict_site_precision_20261005`.
Snapshot: `code_v1/`, copied from the old immutable `code_pilots_v1` without
Python bytecode caches. Relative to 105 old Python files, only
`export_site_quality_cache.py` and `train_site_quality_probe.py` changed. Added
Python files are the prior repeat diagnostic, precision helper and full-VAL
audit. Existing backbone/data/model Python sources remain byte-identical.

Output paths:

- `audit_12919239/{summary.json,comparisons.csv,comparisons.json,manifest.json,predictions/}`
- `run_12919240_none/{cache,baseline,gain,tokens,boundary}`
- `run_12919243_none/cache`
- `run_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}/...` for each full head task.
- `logs/{audit,smoke,cache}_<jobid>.{out,err}` and `logs/heads_12919245_<task>.{out,err}`.

71 local tests passed, including the three synthetic head-training paths,
precision promotion/restoration, training-mode rejection, input fingerprints,
all-outlier collection, unchanged limits, missing/mixed cache protocols and
failed/stale/modified audit rejection. Python compilation and Slurm shell syntax
passed. Local optional PyG extension warnings persist; local tests do not prove
HPC CUDA success. GPU audit/smoke remain the runtime gates.

Deployment SHA256 matched local files:

| File | SHA256 |
| --- | --- |
| site_inference_precision.py | 7caa96264b0f74231b637b04373500dd69198393f57cc63ff98fb489f0cf9768 |
| audit_site_precision_val.py | 73ee080a7f1639d1dc821f9f07419d3ae1639b46487156c7744b31e4a3833384 |
| export_site_quality_cache.py | 6386ae7f643213195d57fca73a560a0d89b4b5ec328915e7710451babe491d68 |
| train_site_quality_probe.py | 2721095c136f8b862f105346a94697f00a63aef5fb759afec5ba398bda0cc7f0 |
| run_site_precision_val_audit.slurm | 0eddf924b1a312518a416254e90bcb9c970261183448d334914b506c2b423f54 |
| run_strict_site_parallel.slurm | 0363484cc73e54ff68191ab64014b909eeacc591f5c3e220ce1e888c030d5ede |

No git commit/push, unrelated worktree rollback, jhspoolers output, mainline
training modification or performance claim was made. If any numerical gate still
fails, inspect the full outlier report before considering a new protocol.
