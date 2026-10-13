# Original-model context noise: completed seed44 screen

Checked 2026-10-13. This is the original GFGF / V11m13 group_rescompat
experiment, NOT self-distillation. All numbers below are clean strict VAL,
not TEST or COACH420/HOLO4K. No evaluation-time corruption was applied.

## Completion and provenance

- Smoke **12990623** completed, exit 0, 5m46s.
- All six tasks **12990624_[0-5]** completed, exit 0, in 4h05m-4h29m each.
  Total training allocation time was 26h14m52s, one GPU per task.
- All runs ended by the original loss-based early stop at epoch 35. Their
  minimum VAL loss was at epoch 5, with patience 30. This is NOT evidence that
  F1 converged or that the configured 120-epoch schedule was completed.
- Confirmed all six `best_f1_c0_model.pt` files remain on remote HDD.
- Synced 44 small artifacts, 82,062,997 bytes, with per-file SHA256 checks;
  no checkpoints or training data downloaded. Slurm accounting, logs, configs,
  smoke/QC and TB are in `tmp/context_noise_20261013/`.
- Config comparison against the original GFGF reference found differences
  only in `context_noise.*`, experiment name and output directory. The model,
  losses, data, split, seed, schedule and checkpoint selection are unchanged.
- No logged scalar was non-finite, and no conflicting repeated TB steps were
  found. Stderr contains deprecation warnings, not runtime failures.

Remote root:
`/hpc2hdd/home/shuang886/PLC/experiments/context_noise_gfgf_20261012`.

## Clean validation results

Epochs are one-based. F1/P/R in this table use the original batch-aggregated
class0 metric used to save best-F1 checkpoints. AUPRC is the global residue
metric evaluated at THAT best-F1 checkpoint, not an independently selected max.
Tail5 is mean F1 over epochs 31-35, not a multi-seed statistic.

| Run | Best F1 | Epoch | Delta vs baseline | Precision | Recall | AUPRC at best F1 | Last F1 | Tail5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original GFGF, no added noise | .535996 | 32 | .000000 | .540253 | .533981 | .546287 | .506131 | .520920 |
| Random .05 | .537633 | 32 | +.001637 | .535584 | .541819 | .550313 | .516151 | .526351 |
| Background .05 | .531978 | 33 | -.004017 | .529693 | .536726 | .543739 | .513500 | .524117 |
| Random .10 | .531351 | 32 | -.004644 | .527344 | .537811 | .538171 | .514065 | .525605 |
| Background .10 | .529397 | 32 | -.006599 | .515413 | .547184 | .537363 | .521377 | .522887 |
| Random .20 | .531116 | 32 | -.004879 | .513200 | .553140 | .539036 | .510631 | .521084 |
| Background .20 | .532950 | 32 | -.003045 | .551094 | .517873 | .546723 | .523790 | .524643 |

Background-minus-random best-F1 differences are -.005654, -.001955 and
+.001834 for sigma .05, .10 and .20. Thus there is no consistent advantage
from noise placement. Background .20 beats Random .20 at the selected peak
and in tail5, but remains below baseline and its epoch21-35 average is .006932
below Random .20. A late peak alone does not establish a better trajectory.

Random .05 is the only peak above baseline: +.001637 absolute, about +0.31%
relative, with AUPRC +.004026 and tail5 +.005431. These are weak single-seed
signals, below the plan's +.003 absolute F1 screen. No arm meets that screen.
No new zero-noise training replica was run; the reference is the existing
seed44 experiment with matched configuration, not a determinism guarantee.

## Aggregation and calibration caveats

At each run's best-F1 checkpoint:

| Run | Global F1 at .5 | MCC at .5 | VAL-selected gate | Global F1 at own gate | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | .536287 | .521754 | .343931 | .539223 | .946482 |
| Random .05 | .536561 | .521800 | .317112 | .539378 | .947562 |
| Background .20 | .531980 | .517914 | .250530 | .542332 | .948590 |

Random .05's global F1 at .5 only improves by .000274; own-gate F1 improves by
.000155. Its larger legacy-F1 gain depends partly on aggregation, so do not
present .001637 as a universal F1 improvement.

Background .20 has higher precision and lower recall at the original operating
point. Its gate-tuned F1 is +.003109 above baseline, but each gate was optimized
on the same VAL used for assessment. This is a calibration lead, not a held-out
gain or evidence that targeted noise works. AUPRC .546723 is effectively close
to baseline .546287. Best AUPRC and best F1 occur at the same epoch in all runs
except Background .10, whose AUPRC max is .539160 at epoch28.

Top-L ranking does not show a large step-change. For Baseline -> Random .05,
L/20 precision/recall is .4586/.5650 -> .4587/.5688; L/10 is
.3275/.7132 -> .3281/.7182. Complete AUROC, gate, MCC and Top-L series are saved
in `tmp/context_noise_20261013/tb_analysis.json`.

## Did the intervention work as specified?

The private noise stream and TRAIN sample ordering produced the same logged
application frequency across all arms: 25.3257% by the last logged step.
Conditional on augmented, logged batches:

| Sigma | Random relative RMS | Background relative RMS | Mean expected relative RMS |
| --- | ---: | ---: | ---: |
| .05 | .043236 | .043235 | .043236 |
| .10 | .086471 | .086471 | .086472 |
| .20 | .172943 | .172942 | .172943 |

Noise was not accidentally disabled and its relative strength scales as
intended. These are means of batch-level logged RMS values, not a pooled
energy estimate over every training token. Absolute RMS is also recorded;
relative matching is not a promise of identical absolute perturbations.

The 128-row TRAIN-only mask QC passed nondegeneracy for all rows:

| True residue class, used only for QC | Residues | Mean protection | Fraction with protection >= .5 |
| --- | ---: | ---: | ---: |
| Contact, <5 A | 2251 | .5053 | .5162 |
| Near, 5-8 A | 3182 | .5159 | .5377 |
| Far background, >=8 A | 52797 | .3325 | .2961 |

The geometry heuristic is enriched for contact/near residues, but misses many
contacts at the illustrative .5 cutoff and also protects considerable
background. This is a soft candidate union, NOT a verified pocket mask.

Across logged augmented TRAIN batches, count-weighted mean amplitudes were:
contact .7433 vs far .8332 for background placement; contact .7790 vs far
.8294 for random placement. The targeted intervention therefore reduces mean
contact amplitude only about 4.6% relative to random. Within-protein random
permutation does not imply globally identical class means, because proteins
have different amplitude distributions and class compositions. These logged
observations repeat residues over epochs and are not independent samples.

The geometry-mask strength/coverage and clean bypass paths limit the experiment:
it perturbs one hidden context path, while b0/GVP/physchem skips remain clean.
It cannot establish that all background information was suppressed, that
attention moved to true pockets, or that the old model was reward hacking.

## Gradient and optimization checks

Baseline total pre-clip gradient norm has median 1.186 and p95 2.588. Noise
arms have medians 1.199-1.219 and p95 2.554-2.650. Protein-context encoder
medians are .550-.559, co-attention .080-.082, local-delta head .133-.136.
These modules receive gradients, without a sustained distributional explosion.

There ARE isolated pre-clip spikes: Background .05 reaches 26.665 vs baseline
10.701. They are finite and the unchanged global clip is 1.0; do not describe
the runs as having no gradient spikes or as demonstrably more stable.
Smoke peak allocated CUDA memory was 51.38 GiB on the smoke B16 batch, not a
measurement of full-training peak memory. Slurm MaxRSS is host RAM, not VRAM.

Training loss falls from approximately .346-.348 at epoch5 to .075 at epoch35;
VAL loss rises from .633-.647 to 1.326-1.414. Noise has not visibly eliminated
the late validation-loss increase. Because F1 usually peaks at epoch32-33,
loss-based stopping does not establish best-F1 convergence. The stopping rule
was intentionally held fixed for this comparison and has NOT been changed.

## Decision

1. Keep the original baseline. Current evidence does NOT support a foreground-
   selection benefit or a new mainline model from this matrix.
2. Retain Random .05 as a weak generic-regularization candidate, not a confirmed
   improvement. Do not expand sigma/rho/model sweeps automatically.
3. If one follow-up is desired, confirm baseline vs Random .05 on independent
   seeds with clean VAL first. A fixed best-F1 TEST real/shuffle/zero evaluation
   can assess transfer, but must not become another TEST-selected sweep.
4. Background .20's own-VAL gate result is secondary. If evaluated, freeze its
   VAL-derived gate and report both original and calibrated operating points;
   do not switch the primary selection rule retrospectively.

No new training, TEST ligand-dependency or external benchmark was submitted
during this check. No running code, frozen experiment config or model was edited.

Reproduction: `tmp/collect_context_noise_results.py` collects the small,
hash-verified artifacts; `tmp/analyze_context_noise_results.py` rebuilds the
comparison and all scalar summaries from TB. Historical implementation details
are in [the execution ledger](context_noise_execution_20261012.md).
