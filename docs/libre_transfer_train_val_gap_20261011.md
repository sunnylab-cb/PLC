# LiBRe transfer: matched-epoch TRAIN/VAL gap

Analyzed 2026-10-11 from the completed scalar snapshot collected at
09:13:44 UTC. Scope: GFGF seed44 reference plus the four completed
`12964718` arms, strict merged PDBbind/BioLiP split (10317 TRAIN, 647 VAL).
This is not TEST or benchmark evidence. No training settings were changed
and no additional GPU evaluation was submitted.

## Matched final epoch

All runs ended at epoch 35. Accuracy is three-class argmax accuracy over valid
residues within each batch, then an unweighted mean over batches. TRAIN is
online training-mode accuracy, not evaluation of a fixed checkpoint on TRAIN.
The gap is TRAIN minus VAL, in percentage points (pp).

| Run | TRAIN accuracy | VAL accuracy | Gap (pp) | TRAIN total loss | VAL total loss | VAL minus TRAIN loss |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 96.0461% | 91.2238% | 4.8222 | .075019 | 1.340522 | 1.265502 |
| Product readout | 96.0670% | 90.8132% | 5.2538 | .073989 | 1.364614 | 1.290625 |
| Additive control | 96.0265% | 91.2601% | 4.7664 | .074311 | 1.411335 | 1.337024 |
| CNC-off | 96.1417% | 91.3380% | 4.8037 | .088684 | 1.387976 | 1.299292 |
| Residue classmass | 95.9788% | 91.2321% | 4.7467 | .076771 | 1.327954 | 1.251183 |

The total losses do NOT have identical definitions; their difference is only
a raw logged difference. Their 15.65-18.99x ratios must not be interpreted as
a corresponding multiplier of generalization error. Accuracy is sensitive
to class imbalance and does not establish contact-class F1.

The pattern persists beyond the final epoch:

| Run | Best VAL F1 epoch | TRAIN / VAL accuracy there | Gap there (pp) | Mean accuracy gap e31-35 (pp) |
|---|---:|---|---:|---:|
| Baseline | 32 | 95.8929% / 91.7514% | 4.1415 | 4.5001 |
| Product | 25 | 95.2161% / 90.6737% | 4.5425 | 4.7076 |
| Additive | 32 | 95.9518% / 91.3602% | 4.5916 | 4.4500 |
| CNC-off | 32 | 96.0045% / 91.5489% | 4.4556 | 4.8267 |
| Classmass | 32 | 95.8202% / 91.2794% | 4.5407 | 4.6618 |

These are descriptive one-seed and within-run statistics, not confidence
intervals. The small final-gap differences do not establish a regularization
benefit: classmass, for example, has a larger tail gap than the reference.

## Why rising validation loss is not declining F1

Reference trajectory, all metrics at the SAME epoch:

| Epoch | TRAIN loss | VAL loss | Accuracy gap (pp) | VAL contact F1 | VAL global AUPRC |
|---|---:|---:|---:|---:|---:|
| 5 | .346304 | .640627 | .8223 | .435161 | .433999 |
| 10 | .182677 | .820465 | 2.6962 | .462054 | .445603 |
| 20 | .112473 | 1.080651 | 4.8101 | .508853 | .508508 |
| 32 | .078567 | 1.454572 | 4.1415 | .535996 | .546287 |
| 35 | .075019 | 1.340522 | 4.8222 | .506131 | .515299 |

From e5 to e32, VAL F1 gains .100834 and global AUPRC gains .112289,
although VAL loss increases .813944. The loss decomposition is informative:

- VAL focal loss: .654833 -> 1.879083; weighted change +.856975.
- VAL contact Tversky loss: .607480 -> .464046; weighted change -.043030.
- The two changes sum to +.813944 with weights .7 and .3.
- At e35, the focal term accounts for 89.11-89.74% of VAL total loss across
  the five runs. All five show rising focal but improving Tversky from e5 to e32.

Thus the loss deterioration comes from focal penalties, not from every
classification/overlap metric deteriorating. This is consistent with growing
penalties on confidently wrong validation residues while ranking/overlap
improves. It does not prove probability miscalibration: per-residue error
confidence, NLL/Brier, class/source breakdowns are needed to distinguish
confidence effects, hard examples and distribution differences.

All runs have minimum VAL loss at e5 and loss-based early stopping at e35;
the best-F1 epochs are e25/e32. Preserve the existing best-F1 checkpoints.
Neither e5 as best-F1 nor full convergence at e35 follows from these logs.

## Loss and measurement audit

The actual VAL objective reconstructs as `.7 * focal + .3 * Tversky` for
all 175 run/epoch pairs, within absolute tolerance 1e-6.

TRAIN differs from VAL in these ways:

1. TRAIN focal uses ordinal label smoothing .05; VAL uses hard targets.
2. TRAIN uses cluster sample weights; the four arm logs report all weights
   .9102. The implementation multiplies per-row losses and then takes their
   mean, without renormalizing weights. This is about a 9% scale change,
   not an explanation for the whole raw loss gap.
3. Except CNC-off, TRAIN downweights far-labeled residues whose predicted
   contact-plus-near probability is at least .85, using multiplier .1 after
   five warmup epochs. VAL does not clip. This is not easy-negative clipping.
   CNC-off still has a 4.8037 pp final accuracy gap and 1.387976 VAL loss.
4. TRAIN adds `.1 * distance_aux_loss` (about .0047-.0048 at e35). Classmass
   additionally adds its own auxiliary objective. VAL includes neither.
5. TRAIN uses training-mode dropout/augmentation and updates parameters
   between batches; VAL uses one fixed end-of-epoch model in eval mode.
   TRAIN is configured BF16; the reused validation autocast defaults to FP16.

`Train/epoch_dice_loss` and `Val/dice_loss` are backward-compatible names for
Tversky LOSS, not measured Dice/F1. Do not turn `1 - loss` into TRAIN F1.
The training loop computes batch class metrics but returns/logs only accuracy;
there is no TRAIN F1 or AUPRC series in any of these five runs. An exact
contact-F1 train/validation gap cannot be recovered from these scalars.

Code audited: `scripts/train/train_v11m3.py` (`compute_combined_loss_v4`,
`train_epoch`, `evaluate`, `compute_metrics`, `ClusterWeightedLoss`),
`scripts/train/train_v11m4.py` (training-only classmass insertion), and
`src/losses/confident_negative_clip.py`. See the final results report for
the frozen deployment provenance and numerical health checks.

## Decision and minimal next measurement

All four probes fit TRAIN to about the same level and none demonstrates a
better best-VAL F1 than the reference. The observed gap is compatible with a
generalization limitation, not evidence that merely enlarging the readout
will help. CNC alone cannot explain it. Numerical failure was not observed
in the synchronized scalars, but the exact cause of the gap is not identified.

The useful next diagnostic is a fixed-checkpoint TRAIN/VAL evaluation using
the existing best-F1 checkpoint, identical eval-mode precision, preprocessing,
hard-label loss and aggregation, without CNC/sample reweighting/auxiliaries.
Use a fixed source-stratified TRAIN subset if full TRAIN evaluation is costly.
Report contact F1, AUPRC, MCC, NLL/Brier and confidence of incorrect residues;
use a VAL-selected threshold unchanged on TRAIN, not separate split-optimal
thresholds. No TEST threshold search or retraining is needed for this check.

Source: `tmp/libre_transfer_12964718/final_20261011/analysis.json` (full scalar
series); raw TB/config/logs beside it and reference TB in the sibling
`baseline/`. Each series uses unique zero-based TB steps; tables add one.
