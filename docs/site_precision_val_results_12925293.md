# V2 full VAL audit: remaining repeatability failure

Subsequent user decision: prioritize progress over strict numerical agreement.
See `strict_site_practical_recovery_20261006.md` for explicit v3 tolerance and
the new recovery chain. The measurements and v2 failure below are unchanged.

Checked 2026-10-06 from Slurm accounting, stdout/stderr and all 647 saved
prediction files. Local evidence: `tmp/site_precision_val_12925293/`.
Remote experiment: `/hpc2hdd/home/shuang886/PLC/experiments/strict_site_precision_v2_20261005`.
Manifest/comparison SHA256 values and every prediction SHA256 were verified.

## Execution and acceptance

Audit **12925293 FAILED 2:0**, after processing all 647 VAL rows. It ran on
gpu1-41, NVIDIA A800-SXM4-80GB, from 2026-10-06 03:23:54 to 03:33:54 scheduler
time. Runtime was 10 minutes; stderr was empty. This was an explicit numerical
acceptance rejection, not OOM, timeout, or an exception traceback.

- `integer_count_v2` fails only on `6fac: repeatability`.
- The recorded v1 comparison fails on both `2k00: binary disagreement` and
  `6fac: repeatability`. The approved integer budget resolves the former only.
- Smoke 12925308, cache 12925310 and heads 12925312_[0-3%3] were automatically
  CANCELLED without running. The queue is empty at this status check.
- No new full training, TEST or external benchmark result exists. No new job
  was submitted during this check; no acceptance limit or model was changed.

## What changed and what did not

| Measurement | Value |
| --- | ---: |
| Archived BF16 VAL sample F1 | 0.5218677608 |
| New mixed-protocol VAL sample F1 | 0.5217902904 |
| Mean F1 difference from archive | -0.0000774704 |
| New mixed-protocol micro F1 | 0.5388621172 |
| Maximum probability error vs archive | 0.0483340919 |
| Mean binary disagreement vs archive | 0.0001029157 |
| Predicted contact count p50 / p90 | 20 / 30 |
| Samples with nonzero repeat probability differences | 1 / 647 |
| Repeated-forward binary flips across all samples | 0 |

The first mixed-protocol probability array in this job is bitwise identical to
the first mixed-protocol array from prior audit 12919239 on **all 647 samples**.
All input fingerprints also match across jobs. The changed audit/export/policy
source files are the expected v2 changes; model sources remain unchanged.
This argues against a changed input or broad prediction shift in this run, but
does not establish deterministic inference: the second forward for 6fac differs.

## Exact 6fac repeat event

| Property | Value |
| --- | ---: |
| Full / observed residues | 655 / 655 |
| Ground-truth contacts | 22 |
| Maximum repeated-forward probability difference | 0.0048123598 |
| Allowed repeated-forward probability difference | 0.0000100000 |
| Residues with any nonzero probability difference | 369 |
| Median / p99 probability difference | 2.0748e-11 / 0.0018419583 |
| Changed binary predictions | 0 |
| Predicted contacts / true positives | 16 / 10 |
| Sample F1, both forwards | 0.5263157895 |

At zero-based residue index 357, probability changes from 0.3821786344 to
0.3773662746. Both remain below the unchanged gate 0.3911699355. Index 53
changes from 0.1990220398 to 0.2037694454. No threshold crossing occurred
anywhere in this sample. Nevertheless, the maximum probability difference is
about 481 times the repeatability limit and cannot be called a passed audit.

## Interpretation and next diagnostic

The earlier statement that mixed repeats were identical on 647 rows was true
for audit 12919239, but is not a general guarantee. This fresh audit exposes a
residual intermittent inference variation. Neither the upstream operator nor
the GPU scheduling cause is localized by the saved probabilities alone.

The current wrapper raises GVP scatter accumulation precision, but still calls
the existing reduction and casts results back to the original dtype; the audit
records deterministic algorithms disabled. It is not an explicit deterministic
reduction implementation. Residual reduction-order effects amplified through
low-precision operations are a hypothesis to test, not a confirmed cause.

Next: repeat 6fac alongside 5ndf/4k7o with a fixed, declared repeat count and
capture layer outputs to find the first divergence. Compare existing mixed
inference against deterministic reduction alternatives and use full FP32 only
as a diagnostic control. Preserve all failed repetitions. After an identified
fix, rerun the full audit and only then regenerate TRAIN/VAL cache and train
heads. Do not loosen repeat tolerance or rerun unchanged audits until one
happens to pass. This check supplies numerical evidence, not an efficacy claim.
