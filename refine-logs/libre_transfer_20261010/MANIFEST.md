# LiBRe transfer planning artifacts

| Date | Artifact | Purpose |
|---|---|---|
| 2026-10-10 | EXPERIMENT_PLAN_20261010.md | Dated, unsubmitted product/additive and supervision controls |
| 2026-10-10 | EXPERIMENT_PLAN.md | Current implementation/deployment status; original dated proposal retained |
| 2026-10-10 | EXPERIMENT_TRACKER_20261010.md | Completed audits versus explicitly unsubmitted training proposals |
| 2026-10-10 | EXPERIMENT_TRACKER.md | Current run state including smoke 12964495 and continuation 12964516 |
| 2026-10-10 | ../../docs/libre_architecture_transfer_20261010.md | Paper/source audit, actual TB results and CPU loss checks |
| 2026-10-10 | ../../docs/libre_transfer_execution_20261010.md | Implemented four-arm matrix, 46 tests, immutable source/asset hashes and queued execution |
| 2026-10-10 | EXPERIMENT_PLAN_20261010_execution.md | Dated implementation/submission snapshot, separate from the initial proposal |
| 2026-10-10 | EXPERIMENT_TRACKER_20261010_execution.md | Dated queued-run state and automatic post-smoke continuation |
| 2026-10-10 late | ../../docs/libre_transfer_progress_20261010.md | First completed product result, running controls, exact VAL and gradient comparison |
| 2026-10-10 late | ../../tmp/libre_transfer_12964718/progress_20261010.json | Full synchronized scalar summaries, matched epochs, smoke and submission provenance |
| 2026-10-11 | ../../docs/libre_transfer_results_20261011.md | Completed four-arm strict-VAL table, gradient/classmass audit, limited conclusions |
| 2026-10-11 | ../../tmp/libre_transfer_12964718/final_20261011/analysis.json | Final scalar series/summaries and absolute/relative baseline differences |
| 2026-10-11 | ../../docs/libre_transfer_train_val_gap_20261011.md | Matched-epoch accuracy gaps, focal/Tversky trends, loss-definition audit and missing TRAIN F1/AUPRC |
| 2026-10-12 | ../../docs/self_distillation_feasibility_20261012.md | Submitted matched TRAIN/VAL evaluation 12977489, TRAIN teacher-cache plan, primary-source self-distillation review and unsubmitted pilot |
| 2026-10-12 follow-up | ../../docs/self_distillation_execution_20261012.md | Completed unified gap/cache, 53 tests, scratch KD implementation, GPU smoke 12977763 and dependent two-arm training 12977764 |
| 2026-10-12 recovery | ../../docs/self_distillation_smoke_recovery_20261012.md | Failed initial smoke, measured base-repeat GPU variation (12984888), corrected gate with 64 tests, isolated retry 12984984 / 12984985 |
| 2026-10-12 completed | ../../docs/self_distillation_results_20261012.md | R2 smoke 12985072 passed; 12985073_[0-1] completed e35; KD best VAL F1 .530556/.529393 below reference .535996; calibration, gradients and actual KL audited; no student TEST yet |
| 2026-10-12 completed | ../../tmp/self_distill_gfgf_20261012_r2/tb_analysis.json | Synchronized complete VAL curves, same-checkpoint ranking metrics, finite-scalar audit and logged-batch KD/gradient summaries |
