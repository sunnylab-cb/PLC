# Assignment experiment tracker

Updated 2026-09-26 after user approval, implementation, tests and Slurm submission.
See `../docs/strict_site_parallel_execution_20260926.md` for the execution ledger.

| Run / job | Stage | Purpose | Status | Next gate |
| --- | --- | --- | --- | --- |
| 12868601 sitequality_smoke | M0 | Frozen pre-cross export and small heads | FAILED 1:0, 00:01:41 | Diagnose independence check; not OOM |
| 12868608 sitequality_full | M1 | Full cache and stats/protein/ligand controls | CANCELLED, never ran | A new immutable retry must depend on a passing smoke |
| 12868960 state diagnostic | M0 | Repeat/intervention FP32 and BF16 | PENDING Priority | Root cause of earlier failure still unverified |
| 12869013 all-branch smoke | M0 | Token export and every training control | PENDING afterok:12868960 | Strict FP32 independence, reference reproduction, all heads |
| 12869014 shared cache | M1 | Full TRAIN/VAL pooled and token cache | PENDING afterok:12869013 | Complete manifest and full population |
| 12869016_0 baseline controls | M2 | stats/protein/ligand quality heads | IMPLEMENTED, PENDING dependency | Baseline for all new branches |
| 12869016_1 site_gain_rank | M2 / P0 | Candidate vs original predicted gain | IMPLEMENTED, PENDING dependency | Paired-feature absolute control |
| 12869016_2 site_lateinteraction | M2 / P0 | Atom-residue token matching | IMPLEMENTED, PENDING dependency | Matched mean-pooling and ligand-blind controls |
| 12869016_3 site_boundary_residual | M2 / P1 | Signed local boundary correction | IMPLEMENTED, PENDING dependency | Protein-only control; TRAIN-fitted refined quality scorer |
| 12869006 train_multisite_union_audit | M0-M2 / P1 | Protein-centric label coverage | RUNNING cpu1-104, 9000/10317 QC rows observed | Verified mapping/geometry and no held-out labels |
| successful head seeds43/45 | M3 | Confirm seed sensitivity | CONDITIONAL | Strict VAL gain and count/precision/recall guardrails |
| strict TEST + COACH/HOLO + ligdep | M4 | Confirm frozen protocol | CONDITIONAL | Freeze all selection, masking and calibration rules first |

The four-element GPU array uses one GPU per task and at most three concurrent
tasks. All depend on a passing smoke and complete export, with invalid-dependency
cancellation. 62 local tests passed; no new model-performance result exists yet.
