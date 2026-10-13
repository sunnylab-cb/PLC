# LiBRe transfer: first training results

Snapshot: approximately2026-10-10 23:56 CST. Source: synchronized TB, Slurm
accounting and stdout/stderr, not model inference. Epochs below are one-based.
Running experiments can advance beyond this saved snapshot.

## Status and strict-VAL results

| Run | Slurm | Status | Best legacy F1 | Best epoch | AUPRC at best F1 | Best AUPRC, own epoch | Last F1 |
|---|---|---|---:|---:|---:|---|---:|
| Frozen GFGF seed44 | Historical reference | Completed35epochs | .535996 | 32 | .546287 | .546287 at e32 | .506131 |
| Product readout | 12964718_0 | Completed35epochs, exit0 | .527161 | 25 | .530372 | .537802 at e35 | .524501 |
| Additive control | 12964718_1 | Running e33;32completed in TB snapshot | .527424 | 32 | .536484 | .536484 at e32 | .527424 |
| CNC-off | 12964718_2 | Running e1 | N/A | N/A | N/A | N/A | N/A |
| Residue classmass | 12964718_3 | Pending JobArrayTaskLimit | N/A | N/A | N/A | N/A | N/A |

Legacy `Val/f1_c0` is the existing batch-averaged argmax class0 F1 used for
checkpoint selection, NOT sample-mean per-protein F1. AUPRC is pooled/global;
Top-L precision is macro over input protein/pair rows. Do not compare these
numbers directly with benchmark sample F1. All results here are strict VAL,
not TEST/COACH/HOLO. All variants share the strict10317/647/647 split and seed44.

GPU smoke12964495 completed18:13 with exit0 in4m47s; CPU dispatcher12964516
completed in7s and submitted the array. The smoke used B16 x1800 residues on
A800-80GB, peak allocated51.38-51.58GiB across four arms. This is smoke VRAM,
not an all-training-batch maximum. Product took4h23m21s and saved its best-F1
and best-AUPRC checkpoints. Stderr contains a GradScaler deprecation warning,
not an OOM/traceback. CNC-off has started emitting batch losses.

## Trends and interpretation

Product bestF1 is -.008834 below the reference, bestAUPRC -.008486. Additive
bestF1 is -.008571 below the reference at this snapshot. There is currently no
evidence that multiplication beats its equal-parameter additive control or
the original baseline under the predeclared best-F1 selection rule.

This is not uniformly worse across the curve. At matched epochs31-35,
product/reference meanF1=.525250/.520920 and meanAUPRC=.534613/.531958.
Product Top-L/10 macro precision over these epochs is .329183 versus .323563.
Its tail is less volatile, but the reference's last-epoch dip contributes to
this difference. Additive also has a small matched-window advantage: at e28-32,
meanF1=.522559 versus reference .519505. These descriptive tail statistics do
not justify changing the checkpoint-selection rule after seeing results.

At each best-F1 checkpoint, P/R is .540253/.533981 (reference),
.488442/.576988 (product), and .516804/.540911 (additive). Product's selected
checkpoint favors recall at a precision cost. Val-selected global gate F1 at
these same checkpoints is .539223/.527706/.530500 respectively; sweeping the
threshold has not recovered the baseline in this snapshot. Do not infer a
ligand-awareness gain without a real/shuffle intervention.

The source of completion is explicit: `Early stopping triggered after35epochs`,
not OOM. The reference and product share loss-based patience30; both reach
minimum val loss around e5 while their F1 improves later. Product train loss
ends .073989 and val loss1.364614 (minimum .641128). This is loss/F1 divergence
and an overconfidence/generalization warning, not proof that35epochs is full
optimization convergence. Keep this round's stopping policy matched; do not
silently resume only a favored variant.

## Gradient health

| TRAIN logged statistic | Reference | Product | Additive, running |
|---|---:|---:|---:|
| Total pre-clip norm median | 1.1861 | 1.1939 | 1.2105 |
| Total pre-clip norm p95 | 2.5884 | 2.5959 | 2.6314 |
| Conditional readout norm median | N/A | .06274 | .00706 |
| Conditional readout norm last5 logged steps | N/A | .05944 | .00318 |

No non-finite scalar was found in the synchronized TB files (110/113/113 tags
for reference/product/additive). The product readout has nonzero gradients and
valid activation magnitude around .58 late in training, so disconnection is
not the evident failure mode. A larger gradient does not prove useful ligand
information or causal contribution. The two running-arm gradient windows are
not matched-duration causal comparisons. Late CNC activity remains small:
roughly .72% of valid TRAIN residues for product/reference.

## Next action and evidence

Allow additive, CNC-off and classmass to finish without changing their recipe.
Do not expand product capacity or launch seed replication on this evidence.
Compare the two supervision-only arms next; any selected survivor still needs
paired-ligand sensitivity checks and seed confirmation before a stronger claim.
No new jobs, cancellations, restarts or training-source edits were made here.

Machine-readable full scalar audit and matched-epoch comparisons:
`tmp/libre_transfer_12964718/progress_20261010.json`.
Raw synchronized assets are in its `experiment/` and `baseline/` subdirectories;
all checkpoints were excluded from transfer. Remote training submission record:
`/hpc2hdd/home/shuang886/PLC/experiments/libre_transfer_20261010/submission.json`.
