# ESM3 Global-Prior Parameter Review (2026-07-14)

## Current best point

The current best single checkpoint is `esm3_global_prior_scratch_seed44`:

- Test real F1: 0.6286
- Test shuffle F1: 0.5547
- Real-shuffle gap: 0.0739
- Test zero F1: 0.5943
- COACH420 sample F1: 0.4960
- HOLO4K sample F1: 0.3568
- HOLO4K BP F1: 0.7807

Its effective recipe is V11m13 group-rescompat plus a bounded ESM3 global
prior, trained from scratch with LR 2e-4, batch size 16, 3 warmup epochs,
AdamW weight decay 0.05, BF16 AMP, and an 80-epoch budget. The ESM3 residual
starts at alpha 0.02 with a 0.20 upper bound and no ESM3 source dropout.

## Raw seed evidence

| mode | seed | best val F1 | best epoch | learned ESM3 alpha | COACH F1 | HOLO F1 |
|---|---:|---:|---:|---:|---:|---:|
| full-finetune | 43 | 0.6028 | 43 | 0.0221 | 0.4867 | 0.3400 |
| full-finetune | 44 | 0.6088 | 44 | 0.0221 | 0.4880 | 0.3395 |
| full-finetune | 45 | 0.6023 | 51 | 0.0227 | 0.4659 | 0.3467 |
| full-finetune | 46 | 0.6034 | 38 | 0.0218 | 0.4868 | 0.3379 |
| scratch | 43 | 0.5886 | 17 | 0.0463 | 0.4742 | 0.3458 |
| scratch | 44 | 0.6056 | 36 | 0.0702 | 0.4960 | 0.3568 |
| scratch | 45 | 0.5905 | 33 | 0.0678 | 0.4739 | 0.3461 |
| scratch | 46 | 0.5944 | 29 | 0.0622 | 0.4607 | 0.3441 |

| aggregate | val F1 mean +/- sd | learned alpha mean +/- sd | COACH F1 mean +/- sd | HOLO F1 mean +/- sd |
|---|---:|---:|---:|---:|
| full-finetune | 0.6043 +/- 0.0030 | 0.0222 +/- 0.0004 | 0.4819 +/- 0.0106 | 0.3410 +/- 0.0039 |
| scratch | 0.5948 +/- 0.0076 | 0.0616 +/- 0.0108 | 0.4762 +/- 0.0147 | 0.3482 +/- 0.0058 |

Seed44 ligand-dependency evidence:

| mode | real F1 | shuffle F1 | zero F1 | real-shuffle |
|---|---:|---:|---:|---:|
| full-finetune | 0.6262 | 0.5526 | 0.4586 | 0.0736 |
| scratch | 0.6286 | 0.5547 | 0.5943 | 0.0739 |

## Interpretation

1. Full-finetune barely moves ESM3 alpha away from 0.02. It has the stronger
   validation mean, but weaker HOLO4K transfer and slightly lower seed44 test
   F1. ESM3 acts as a weak correction to the warm-start trunk in this mode.
2. Scratch training consistently raises alpha into the 0.046-0.070 range.
   This indicates meaningful ESM3 use, but seed44 is an optimistic outlier in
   both validation and external benchmarks.
3. The alpha ceiling of 0.20 is not active. Raising the ceiling is therefore
   unsupported. Directly initializing alpha at 0.06 is also risky because the
   ESM3 projection is random at scratch initialization.
4. Existing epoch selection is noisy: scratch runs peak at epochs 17-36 and
   later F1 fluctuates substantially while early stopping is driven by val
   loss. Threshold-independent checkpoint selection is the immediate priority.
5. Legacy `best_val_f1_c0` is a mean of batch-level F1 values, whereas the
   ligand-dependency test reports one global residue-level F1. The values are
   not directly comparable. Legacy `dice_c0` is the closer global argmax-F1
   analogue; new metric names explicitly encode their aggregation scope.

## Gradient-path audit

The optimizer state of the best seed44 scratch checkpoint exposed a silent
gradient-path defect: `local_delta_logit_scale` had no Adam state and remained
at its initialization value 0.5. The V11m4t no-gate wrapper reconstructed
`logits` from the detached diagnostic tensor `pair_delta_scale`, disconnecting
the scale parameter while the rest of the local-delta branch continued to
train. The historical effective scale was therefore fixed at
`tanh(0.5) = 0.4621`.

The wrapper now returns the parent's differentiable no-gate logits directly;
the gated path recomputes the scale through `_delta_scale()`. Training health
logs also report gradient presence and norm for the local-delta scale, ESM3
alpha/projection, and the active local54 interaction controls. This fix changes
optimization rather than the forward value at initialization, so it requires
an exact fixed-versus-trainable control before promotion.

## Implemented evaluation protocol

New training runs now report global residue-level AUPRC/AUROC, F1 and MCC at
0.5, the exact val-selected contact gate with F1/MCC at that gate, and
sample-macro Top-L/20, Top-L/10, and Top-L/5 precision/recall/hit rate.
`best_auprc_c0_model.pt` and `best_auroc_c0_model.pt` are saved as lean
inference checkpoints and carry `val_contact_gate` in their payload.

## Recommended next probes

1. Re-run seed44 scratch as an exact fixed-scale control and a trainable-scale
   variant under the new selectors. Compare best-F1, best-AUPRC, and best-AUROC
   checkpoints on test real/shuffle/zero. This isolates the gradient-path fix.
2. If trainable scale wins, replicate it on seeds 43/45/46 and report global
   val AUPRC/AUROC, F1/MCC at 0.5 and the val-selected gate, plus Top-L metrics.
3. Only then probe whole-source ESM3 dropout at 0.05. This targets the scratch
   line's seed variance without weakening the validated ligand local54/graph
   path or confounding the scale-fix comparison.
4. Keep ESM3 alpha init at 0.02. The learned alpha does not approach its 0.20
   cap, so neither a higher cap nor a larger initial alpha is currently
   supported.
5. Defer attention pooling until the scale control is resolved. Promote a
   variant only if val AUPRC and Top-L precision improve together, test real F1
   rises, and the real-shuffle gap remains near 0.074 or higher.
