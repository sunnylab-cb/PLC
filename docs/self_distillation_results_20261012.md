# GFGF self-distillation: completed first generation

## Execution and scope

Checked on 2026-10-12 via `ssh hpc2`, Slurm accounting, complete logs and
TensorBoard. The corrected GPU smoke passed. Both students trained from
scratch and completed normally; neither ran out of memory.

| Stage | Job | State | Elapsed |
|---|---|---|---|
| GPU smoke | 12985072 | COMPLETED, exit 0 | 2m44s |
| T=2, lambda=.05 | 12985073_0 | COMPLETED, exit 0 | 4h06m46s |
| T=2, lambda=.10 | 12985073_1 | COMPLETED, exit 0 | 4h06m03s |

Training started at 12:11:29 and finished at 16:18:15 / 16:17:32, respectively
(Slurm-reported timestamps). Each task requested one A800 on `i64m1tga800ue`.
All three runs, including the original non-KD reference, stopped after epoch
35 under the unchanged VAL-loss patience rule. Minimum VAL loss occurred at
epoch 5; patience was 30. This is not proof that best-F1 training converged,
and is not a timeout or a completed 120-epoch schedule.

All results below are **strict VAL, seed44**. Student TEST, ligand-dependency,
COACH420 and HOLO4K evaluations have NOT been run. No new task was submitted
during this result collection. No model, loss, selection or stopping rule was
changed after observing the results.

## Primary comparison

F1/P/R are the existing batch-aggregated class-0 metrics used to select
`best_f1_c0_model.pt`. AUPRC and AUROC are global residue metrics, evaluated
at that same checkpoint. Epoch numbers are one-based; TB steps are zero-based.
Do not replace the reference F1 with the separate global-argmax reevaluation
value 0.535538 when reading this table.

| Run | Best F1 | Delta vs base | Epoch | P / R at best F1 | AUPRC at best F1 | AUROC at best F1 | Last F1 | Tail5 mean F1 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| GFGF reference, no KD | 0.535996 | 0 | 32 | 0.540253 / 0.533981 | 0.546287 | 0.946482 | 0.506131 | 0.520920 |
| T2, lambda .05 | 0.530556 | -0.005439 | 33 | 0.552183 / 0.512236 | 0.543971 | 0.947746 | 0.528533 | 0.526398 |
| T2, lambda .10 | 0.529393 | -0.006603 | 33 | 0.531093 / 0.530147 | 0.538321 | 0.944956 | 0.511357 | 0.522116 |

Separately selected **best AUPRC** is 0.546287/e32, 0.543971/e33 and
0.545090/e32, respectively. The .10 arm's best-AP checkpoint differs from
its best-F1 checkpoint; mixing the two would overstate the latter's results.

Selected epochs, showing F1 rather than interpreting combined TRAIN loss as
generalization:

| Epoch | Reference | KD .05 | KD .10 |
|---|---:|---:|---:|
| 5, last zero-KD epoch | 0.43516 | 0.43415 | 0.43636 |
| 10 | 0.46205 | 0.47274 | 0.48026 |
| 15 | 0.51522 | 0.51399 | 0.51131 |
| 25 | 0.51898 | 0.51909 | 0.52138 |
| 30 | 0.52046 | 0.51315 | 0.52337 |
| 32 | 0.53600 | 0.52377 | 0.52830 |
| 33 | 0.52650 | 0.53056 | 0.52939 |
| 35 | 0.50613 | 0.52853 | 0.51136 |

KD has an early advantage at e10 that is not sustained as a best-F1 gain.
The .05 arm has a higher last-five mean, but the user's primary selection is
best F1, not last epoch or tail mean. One seed and known non-bitwise GPU
repeatability do not support a claim of reproducible stabilization or harm.

## Calibration and ranking

These are global `p(class0)` threshold metrics on VAL, all at each run's
best-F1 checkpoint. The threshold is optimized on that same VAL set, so the
optimized score is a diagnostic, NOT held-out confirmation of calibration.

| Run | F1 at 0.5 | VAL-selected gate | F1 at gate | MCC at gate | Top L/20 P / R | Top L/10 P / R | Top L/5 P / R |
|---|---:|---:|---:|---:|---|---|---|
| Reference | 0.536287 | 0.343931 | 0.539223 | 0.524427 | 0.45856 / 0.56497 | 0.32746 / 0.71324 | 0.20758 / 0.83060 |
| KD .05 | 0.532774 | 0.284286 | 0.539569 | 0.524952 | 0.46085 / 0.55961 | 0.33031 / 0.71240 | 0.20947 / 0.83315 |
| KD .10 | 0.529437 | 0.391392 | 0.532087 | 0.516976 | 0.44741 / 0.54806 | 0.32294 / 0.69931 | 0.20897 / 0.82738 |

The .05 arm increases P by 0.011929 and decreases R by 0.021745 in the
legacy metrics. After threshold adjustment its global F1 differs from the
reference by only +0.000346. This is compatible with changed operating point
and small ranking changes, not a demonstrated improvement. AP is slightly
lower, AUROC slightly higher, and top-budget ranking is broadly similar.
The .10 best-F1 checkpoint remains weaker even after its own VAL gate search.

## KD and gradient health

The TB schedule is correct: zero weight through e5, ramp from e6 to e10,
then .05/.10. TRAIN-only cache alignment succeeded throughout. Ligand-dropped
rows are excluded from KD, not from the original hard-label loss. About
88-91% of rows in the listed logged epochs are KD-eligible, consistent with
the configured source-dropout rates (expected keep fraction 0.893952).

Values below are means over logged batches, not an exhaustive residue-pooled
loss or gradient decomposition.

| Diagnostic | KD .05 | KD .10 |
|---|---:|---:|
| Raw T-squared KL, e6 -> e35 | 0.92270 -> 0.12315 | 0.84293 -> 0.10745 |
| Weighted KL / original hard+aux loss, e10 | 10.31% | 18.11% |
| Weighted KL / original hard+aux loss, e35 | 7.83% | 13.60% |
| True background class2 share of KL, e6 -> e35 | 83.73% -> 71.27% | 82.19% -> 68.24% |
| True contact class0 share of KL, e35 | 11.19% | 12.78% |

KD was active and student predictions moved closer to the teacher; failure
to exceed reference F1 is not an absent-loss or zero-gradient failure. Most
KL loss mass is on background residues. This motivates, but does not prove,
the hypothesis that dense KD mainly reinforces common/background decisions
instead of adding difficult contact/site-assignment information. Loss shares
are not gradient contributions; class counts are highly imbalanced.

Follow-up denominator audit (2026-10-12): at e35 the logged eligible residue
counts are 3.57% contact, 5.11% near and 91.32% background. Their KL shares are
11.19% / 17.54% / 71.27% for .05, and 12.78% / 18.98% / 68.24% for .10.
Thus the majority-background KL observation alone does not demonstrate
preferential learning of background. Counts are raw eligible counts, whereas
KL contributions are already weighted by inverse row length and batch size;
dividing these summaries does NOT give an exact per-residue KL comparison.
Neither statistic measures whether pocket predictions depend on background
INPUT features. That requires a separate input/activation intervention.
See [the shortcut audit and literature review](background_noise_audit_20261012.md).

Gradient norms are recorded after AMP unscale and **before clipping at 1**.
Each run has 2240 logged gradient steps, not all 22575 training steps.

| Gradient statistic | Reference | KD .05 | KD .10 |
|---|---:|---:|---:|
| Median, all logged steps | 1.1861 | 1.3144 | 1.3855 |
| P95, all logged steps | 2.5884 | 2.8890 | 3.3170 |
| Maximum, all logged steps | 10.7011 | 11.9736 | 25.6959 |
| Fraction of logged steps above clip threshold | 64.91% | 70.94% | 75.63% |
| Last-five-epoch median | 0.8312 | 0.8413 | 0.9173 |
| Last-five-epoch P95 | 1.8286 | 2.4217 | 2.8860 |

All recorded scalars are finite. Logs contain no OOM/NaN failure; stderr only
contains a GradScaler deprecation warning. Larger KD weight increases
optimization pressure in these runs without improving best F1. Do not call
the pre-clipping maximum 25.7 an unbounded optimizer update or divergence.

## Decision

1. Keep the original GFGF best-F1 checkpoint as the mainline. This pilot does
   not meet the predeclared +0.003 VAL F1 screen. Do not expand KD .10 seeds.
2. KD .05 is approximately tied only after VAL threshold fitting, not a new
   best model. If one follow-up is desired, first use its already-saved
   best-F1 checkpoint and fixed VAL gate on TEST real/shuffle/zero; do not
   train another generation or tune on TEST to justify this branch.
3. If revisiting distillation, test a small change in *which* TRAIN evidence
   is distilled (e.g. contact/boundary-aware or teacher-reliable regions),
   rather than increasing uniform KL. This is an untested hypothesis and
   differs from the earlier hard-label classmass control. New site-assignment
   supervision is not supplied by merely copying the same teacher.

The students' eval-mode full TRAIN metrics have not been measured. Thus this
run cannot yet claim that self-distillation closes the unified TRAIN/VAL gap.
The baseline's training memorization and distribution shift remain possible
limits. Results apply to this T=2, two-weight, seed44 pilot, not all forms of
self-distillation.

## Artifacts

- Remote frozen experiment: `/hpc2hdd/home/shuang886/PLC/experiments/self_distill_gfgf_20261012_r2`.
- Output names: `gfgf_sd_t2_lam005_scratch_seed44_v11m13_20261012_121134` and `gfgf_sd_t2_lam010_scratch_seed44_v11m13_20261012_121134`.
- Both best-F1 weights remain under the respective `outputs/<name>/best_f1_c0_model.pt` paths.
- Local synchronized logs/configs/splits/TB/smoke: `tmp/self_distill_gfgf_20261012_r2/`; checkpoints were not downloaded.
- Full numeric extraction, including every VAL epoch and sampled KD statistics: `tmp/self_distill_gfgf_20261012_r2/tb_analysis.json`.
- Reference TB: `tmp/libre_transfer_12964718/baseline/runs/`.
- [Execution protocol](self_distillation_execution_20261012.md).
- [Smoke recovery history](self_distillation_smoke_recovery_20261012.md).
