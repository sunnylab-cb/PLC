# LiBRe transfer: completed seed44 pilot

Checked2026-10-11 about17:15 CST. All four training tasks completed with
exit0 and the unchanged loss-based early-stop rule at35epochs. This is strict
VAL evidence, not TEST/COACH/HOLO inference. No new training/evaluation was
submitted during this check.

Follow-up: [matched TRAIN/VAL gap audit](libre_transfer_train_val_gap_20261011.md)
finds a 4.75-5.25 percentage-point final accuracy gap and focal-driven VAL loss
deterioration despite improving best F1. TRAIN/VAL loss definitions differ;
TRAIN F1/AUPRC were not logged, so their exact generalization gaps are unknown.

## Final metrics

| Run | Best legacy F1 | Epoch | Delta vs base | AUPRC at best-F1 checkpoint | Best AUPRC, own epoch | Last F1 | Mean F1 e31-35 |
|---|---:|---:|---:|---:|---|---:|---:|
| GFGF seed44 baseline | .535996 | 32 | .000000 | .546287 | .546287 e32 | .506131 | .520920 |
| Product readout | .527161 | 25 | -.008834 | .530372 | .537802 e35 | .524501 | .525250 |
| Additive control | .527424 | 32 | -.008571 | .536484 | .536983 e33 | .512707 | .521562 |
| CNC-off | .535429 | 32 | -.000567 | .545365 | .545365 e32 | .514654 | .523693 |
| Residue classmass | .531508 | 32 | -.004487 | .539927 | .539927 e32 | .521543 | .523926 |

The relative best-F1 changes are -1.65%, -1.60%, -0.11% and -0.84% respectively.
All runs use seed44 and the frozen strict10317/647/647 TRAIN/VAL/TEST split.
`Val/f1_c0` is the legacy batch-averaged argmax score, not benchmark sample-mean
F1. AUPRC/AUROC are global pooled metrics. Epochs are one-based. Best AUPRC
can use a different checkpoint; do not assign it to the best-F1 model.

| Best-F1 checkpoint | Precision / recall | Global F1 at VAL-selected gate | Gate | Top L/20 precision / recall (macro) |
|---|---|---:|---:|---|
| Baseline | .540253 / .533981 | .539223 | .343931 | .458561 / .564972 |
| Product | .488442 / .576988 | .527706 | .475473 | .455366 / .555534 |
| Additive | .516804 / .540911 | .530500 | .362710 | .454742 / .553411 |
| CNC-off | .543160 / .530187 | .542162 | .257730 | .454927 / .556442 |
| Classmass | .538424 / .527047 | .536637 | .310367 | .457571 / .564026 |

## Interpretation and limits

1. No arm passes the planned +.003 best-F1 improvement screen. Product does
   not outperform its equal-parameter additive control; both are about.009
   below the reference. This does not support a product-specific gain in this
   local-readout implementation and training recipe. It does not invalidate
   LiBRe, whose architecture/training data differ, or multiplication generally.
2. CNC-off is the closest control: bestF1 -.000567 and bestAUPRC -.000923.
   That is a near tie descriptively, not demonstrated equivalence from one
   seed. The calibrated global F1 at its best-F1 checkpoint is +.002939 above
   the reference (.542162 vs .539223), but AUPRC and Top5%-budget precision/
   recall do not improve there. This is a small VAL operating-point signal,
   not established generalization or a reason to replace the mainline.
3. Classmass is active and optimizes its intended contact/far objective, yet
   bestF1 falls .004487 and bestAUPRC .006361. Its objective separates residue
   classes across input rows; it does not directly supervise which distinct
   pocket belongs to a particular ligand. The result does not support a claim
   of better ligand assignment. No ligand intervention has been run here.
4. Product's e31-35 meanF1 exceeds baseline by .004330, with descriptive
   epoch SD .001475 vs .010439. Classmass/CNC-off also have mildly higher tail
   means. These are within-run epoch statistics, not seed uncertainty. The
   baseline's final dip contributes, and additive shares some smoothing. Do
   not switch from best-F1 to tail-average selection after seeing this table.
5. The highest separately selected AUROC is product .950189, versus baseline
   .947742. It coexists with lower bestF1/AUPRC and cannot override the primary
   contact-detection metrics; AUROC and AUPRC measure different ranking aspects.

## Health and stopping

No non-finite scalar was found in the final synchronized TB of any arm.
All best-F1 and best-AUPRC checkpoints exist remotely. Stderr contains API
deprecation warnings, not CUDA OOM or training exceptions.

| Pre-clipping total gradient norm | Median | p95 | Largest logged norm |
|---|---:|---:|---:|
| Baseline | 1.1861 | 2.5884 | 10.7011 |
| Product | 1.1939 | 2.5959 | 8.4194 |
| Additive | 1.1828 | 2.6024 | 11.4103 |
| CNC-off | 1.1880 | 2.6324 | 16.9397 |
| Classmass | 1.2140 | 2.6244 | 15.1335 |

CNC-off/classmass have occasional higher logged pre-clip peaks, but similar
median/p95 norms. The configured clip is1.0. Finite logs are evidence against
a numerical failure, not proof that gradients provide useful ligand evidence.

Classmass raw loss decreases .762392 -> .025651; its weighted contribution
decreases .015248 -> .000513 (last5 logged-step mean .000909). Every logged
batch has16 eligible rows and702-866 eligible anchors (mean778.06), so the
auxiliary was neither disconnected nor mostly skipped. Mean sampled counts:
267.40 contact and510.67 far residues. These are logged batches, not an audit
of every optimization step, and row balancing is not sequence-group balancing.

All five runs have minimum validation loss at e5, then stop at e35 using
patience30; most best-F1 checkpoints are much later (e25/e32). This confirms
loss/F1 disagreement, not guaranteed full convergence. Keep this pilot's
comparison matched. Do not silently extend only a favored arm or change the
user-approved best-F1 checkpoint policy.

## Completion and next step

| Slurm task | Variant | Elapsed | End (CST, 2026) |
|---|---|---|---|
| 12964718_0 | Product | 4h23m21s | Oct10 23:49:50 |
| 12964718_1 | Additive | 4h30m33s | Oct11 00:12:49 |
| 12964718_2 | CNC-off | 4h26m50s | Oct11 04:16:42 |
| 12964718_3 | Classmass | 4h30m49s | Oct11 04:47:01 |

Training allocation total:17.859GPU-hours, excluding smoke. No job was canceled
or resumed. Keep the unchanged baseline as mainline; do not expand these
training matrices on one negative/near-tie seed. If a minimal extra evaluation
is approved, CNC-off versus baseline with each model's frozen VAL-selected
gate and identical TEST real/shuffle/zero protocol is the most defensible
remaining operating-point check. Do not search TEST thresholds or claim that
zero F1 must increase. Multi-seed confirmation would still be necessary.

Raw final TB/config/log snapshot:
`tmp/libre_transfer_12964718/final_20261011/`.
Complete scalar series, summaries and numerical differences: `analysis.json`
inside that directory. Baseline TB is in the sibling `baseline/` directory.
The Oct10 partial snapshot and report are retained, not overwritten.
