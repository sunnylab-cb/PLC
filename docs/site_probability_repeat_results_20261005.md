# Targeted probability repeatability: completed diagnosis

Approved continuation: the new full-VAL audit and conditional recovery chain are
recorded in `strict_site_precision_recovery_20261005.md`. The status below is the
earlier diagnostic inspection, before that subsequent submission.

Checked 2026-10-05. Job **12918127** completed with exit 0:0 in 00:01:23,
from 00:47:11 to 00:48:34 scheduler time. Error log is empty. No formal training
has restarted; the account queue was empty at this inspection.

Evidence: `tmp/site_probability_repeat_20261005/report.json`,
`5ndf_probabilities.npz`, `4k7o_probabilities.npz`. Remote origin:
`/hpc2hdd/home/shuang886/PLC/experiments/site_probability_repeat_20261004/run_12918127`.

## Controlled repeatability result

Each sample has ten identical-input forwards per condition. The last condition
keeps BF16 autocast but casts GVP scatter inputs to FP32 for mean reduction,
then casts the reduction output back to its original dtype. This was a
process-local intervention, not a change to the archived model source.

| Sample | Full / observed residues | BF16 max pairwise probability difference | FP32 max pairwise difference | BF16 + FP32 GVP scatter max pairwise difference |
| --- | ---: | ---: | ---: | ---: |
| 5ndf | 3426 / 1800 | 0.03856844 | 0.0000026822 | 0 |
| 4k7o | 336 / 336 | 0.03931046 | 0.0000013113 | 0 |

All three conditions had zero binary prediction flips *within their own ten
repetitions*. This does not mean the three precision conditions have the same
binary predictions. In the mixed scatter condition, captured GVP, pre-cross
protein/ligand and logit outputs were also identical across the ten repetitions.
The unchanged-input BF16 GVP state difference reached 0.03417 (5ndf) and 0.02457
(4k7o); logit differences reached 0.125 and 0.13672 respectively.

For both samples, reloading produced exactly matching input tensor hashes,
forward did not mutate inputs, and neither CPU nor CUDA RNG state changed.
All model modules were in eval mode. Archived labels, CA coordinates and
observed masks matched the partial cache and current loader checks. Prior static
audit found all 66 reference-side model/data/utility/inference files unchanged.

These controls strongly localize the observed repeated-forward instability to
low-precision GVP scatter reduction. They do not prove bitwise determinism on
every sample, GPU or software version, nor establish a specific CUDA kernel's
implementation details.

## Why the old reproduction gate can fail

| Sample | BF16 max probability error vs archived reference, range over ten runs | Runs above original 0.05 limit | Mixed scatter error vs reference |
| --- | ---: | ---: | ---: |
| 5ndf | 0.00861-0.04718 | 0/10 | 0.02490 |
| 4k7o | 0.02760-0.06691 | 1/10 | 0.04833 |

Thus the same class of gate failure was reproduced on 4k7o without changing
weights or inputs. The exact prior 5ndf maximum 0.05126 was not reproduced in
these ten trials; repeated error varies and these trials do not exhaust its
distribution. Do not present this as an exact replay of the old failure.

Both mixed-scatter samples pass the original per-row probability <=0.05 and
binary-disagreement <=0.01 limits. The 4k7o result remains close to the probability
limit. This is NOT evidence that all 647 VAL rows pass, or that whole-VAL mean
disagreement / F1 reproduction gates pass.

## Full FP32 is not a drop-in replacement

| Sample | Archived BF16 F1 | Current BF16 F1 | BF16 + FP32 GVP scatter F1 | Full FP32 F1 |
| --- | ---: | ---: | ---: | ---: |
| 5ndf | 0.387097 | 0.387097 | 0.387097 | 0.322581 |
| 4k7o | 0.240000 | 0.250000 | 0.250000 | 0.173913 |

Full FP32 is internally repeatable but differs from the archived BF16 reference
by up to 0.79529 / 0.56537 in per-residue probability and 4 / 2 binary predictions.
Changing full-model arithmetic is therefore a different inference protocol, not
merely removing stochastic error. These two selected diagnostics cannot establish
that FP32 is globally worse, but rule out assuming equivalence to the old baseline.
Likewise, mixed-scatter repeatability is a numerical benefit, NOT a demonstrated
F1 improvement.

Environment: torch 2.9.1+cu128, CUDA 12.8, PyG 2.7.0, A800-SXM4-80GB;
matmul precision highest, matmul TF32 off, cuDNN TF32 on, deterministic algorithms
off. These are current diagnostic settings, not a reconstruction of every
historical training environment.

## Recommended next gate

1. Use an explicit, versioned inference-only BF16 + FP32 GVP accumulation option
   on all 647 strict VAL rows, with matched unmodified inference comparisons and
   the same checkpoint, fixed gate, labels, split and source inputs. Retain all
   original acceptance thresholds, report every outlier instead of stopping at
   the first one, and check aggregate F1 and selected-contact counts as well.
2. If the full audit passes, regenerate BOTH TRAIN and VAL shared caches under
   that one protocol. Do not mix old BF16 training tokens with new stable VAL
   tokens or silently accept the failed partial manifest.
3. Run the original four matched head groups only after a complete cache passes.
   The new numerical protocol needs a matched base result; it is not a new
   architecture claim or a reason to replace historical benchmark numbers.

This check only collected and analyzed the diagnostic. It did not submit that
full-VAL audit, relax any tolerance, alter mainline model code or restart heads.
