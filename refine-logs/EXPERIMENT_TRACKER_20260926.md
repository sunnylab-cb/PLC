# Assignment experiment tracker

Checked 2026-09-26 via ssh hpc2 sacct. New entries are proposals, not submissions.

| Run / job | Stage | Purpose | Status | Next gate |
| --- | --- | --- | --- | --- |
| 12868601 sitequality_smoke | M0 | Frozen pre-cross export and small heads | FAILED 1:0, 00:01:41 | Diagnose independence check; not OOM |
| 12868608 sitequality_full | M1 | Full cache and stats/protein/ligand controls | CANCELLED, never ran | A new immutable retry must depend on a passing smoke |
| site_gain_rank | M2 / P0 | Candidate vs original predicted gain | PROPOSED, not implemented/submitted | Pooled cache; paired-feature absolute control |
| site_lateinteraction | M2 / P0 | Atom-residue token matching | PROPOSED, not implemented/submitted | Token cache; matched mean-pooling and ligand-blind controls |
| site_boundary_residual | M2 / P1 | Signed local boundary correction | PROPOSED, not implemented/submitted | Token cache; protein-only control; TRAIN-fitted refined quality scorer |
| train_multisite_union_audit | M0-M2 / P1 | Protein-centric label coverage | PROPOSED CPU audit | Verified mapping/geometry and no held-out labels |
| successful head seeds43/45 | M3 | Confirm seed sensitivity | CONDITIONAL | Strict VAL gain and count/precision/recall guardrails |
| strict TEST + COACH/HOLO + ligdep | M4 | Confirm frozen protocol | CONDITIONAL | Freeze all selection, masking and calibration rules first |

No new code was deployed or jobs submitted by this planning update.
