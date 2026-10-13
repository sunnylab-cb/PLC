# Practical tolerance and resumed site-head trials

Status update: audit and smoke have completed successfully; production cache
is pending Priority and heads await it. See `strict_site_practical_status_20261006.md`.
The table below preserves submission-time states.

2026-10-06: the user requested that immaterial numerical differences no longer
block progress. No additional operator/layer-level diagnostic experiment is
being pursued. The existing short-audit -> smoke -> cache -> heads pipeline is
resubmitted with an explicit practical acceptance policy.

## Acceptance policy

`prediction_stable_v3` retains the v2 integer count rule and raises the maximum
repeated-forward probability difference from 1e-5 to **0.01**. Repeat binary
predictions must still be identical. Audit and subsequent cache export use the
same helper, avoiding a stale 1e-5 export rejection. Tolerated probability drift
is logged, not hidden. Finite values, population/axis/hash checks, archived
maximum probability error 0.05, mean disagreement 0.001 and absolute mean F1
change 0.001 remain enforced. No checkpoint, classification gate, inference
arithmetic, model architecture or feature cap changes.

This is a user-authorized, post-result engineering tolerance decision, not a
claim of deterministic inference. Original v1/v2 failures remain unchanged.
The complete saved 12925293 measurements pass v3 when re-evaluated; that is not
a new GPU result. The new short audit binds the deployed code to a fresh report
using the existing approval path rather than rewriting an old failed report.

Validation: **80 tests passed**, Python compilation and Slurm syntax checks
passed. Five changed deployment files match local SHA256 hashes. Existing local
optional PyG extension warnings persist; GPU checks run in HPC's af3 environment.

## Submitted jobs

Root: `/hpc2hdd/home/shuang886/PLC/experiments/strict_site_practical_20261006`
Snapshot: `code_v3/`, cloned from v2 with only approval/export/launcher updates.

| Job | Stage | Verified status |
| --- | --- | --- |
| 12930149 | Short full-VAL approval | PENDING Priority |
| 12930150 | Small cache and all head/control smoke | PENDING afterok:12930149 |
| 12930151 | Shared full TRAIN/VAL cache | PENDING afterok:12930150 |
| 12930152_[0-3%3] | quality, gain, tokens, boundary controls | PENDING afterok:12930151 |

All use i64m1tga800ue, one GPU per task, at most three concurrent head tasks.
Invalid dependencies cancel downstream jobs. Original head settings and matched
controls are unchanged. No training has started at this verification.

Approval: `audit_12930149/summary.json`.
Cache: `run_12930151_none/cache/`.
Logs: `logs/audit_12930149.*`, `logs/smoke_12930150.*`,
`logs/export_12930151.*`, `logs/heads_12930152_<task>.*`.
All new outputs stay on HDD, never jhspoolers. No TEST/COACH/HOLO selection.
