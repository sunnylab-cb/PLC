# Self-distillation smoke failure and recovery

Latest follow-up, 2026-10-12: **recovery succeeded**. Smoke 12985072 passed
in 2m44s, and both 12985073 students completed normally after 35 epochs.
Best VAL F1 is .530556 / .529393; no student TEST result yet. See the
[completed result analysis](self_distillation_results_20261012.md).
The pending/failed states below are retained as historical observations.

## Observed failure

On the 2026-10-12 follow-up, `12977763` was FAILED after 3m02s (exit 1),
not OOM. It reached both BF16 optimizer steps, then failed an allclose check
on `physchem_encoder.0.weight` at lambda zero. `12977764_[0-1]` was cancelled
automatically by `--kill-on-invalid-dep=yes`; neither student started training.
There are no student VAL scores or evidence against self-distillation from
these jobs. The failure was in the pre-training equivalence gate.

The original script required all post-Adam tensors to agree at rtol=3e-4,
atol=2e-6, but did not measure whether the baseline itself met that condition.

## Measured cause

One-GPU diagnostic `12984888` completed successfully in 3m10s on gpu1-12.
It repeated the actual B16 production training step with the same seed and
batch: baseline cold, baseline warm1, baseline warm2, and lambda-zero KD.

All four initial state hashes are exactly identical. Their CPU and CUDA RNG
states are exactly identical both before and after the steps. The comparison
therefore does not indicate changed initialization or consumed random draws.

| Control | Exact loss | Logit RMS difference vs cold base | Max post-step weight difference vs cold base |
|---|---:|---:|---:|
| Base cold | 1.092094779 | 0 | 0 |
| Base warm1 | 1.091983914 | 0.00127190 | 0.000199994 |
| Base warm2 | 1.091965199 | 0.00127413 | 0.000199994 |
| KD lambda zero | 1.092097878 | 0.00128016 | 0.000199993 |

Warm-base vs warm-base logit RMS was 0.00126512. KD0 vs either warm base was
0.00127269-0.00127419. These are the same scale. Base-repeat argmax disagreement
was 0.152-0.181% of valid residues, compared with 0.152-0.162% for KD0 comparisons.

The original strict post-update tolerance fails for the unchanged model too.
This is empirical non-bitwise GPU repeatability in the BF16/custom-kernel/graph
path. This audit does not identify the precise nondeterministic kernel. The
max approximately 2*LR Adam update difference is not a measure of task-quality
degradation and must not be confused with an F1 change.

Evidence: `tmp/self_distill_control_audit_20261012/control_12984888.json`,
including complete comparisons, weight/RNG hashes and gradient norms. Original
failure logs remain in `tmp/self_distill_gfgf_20261012/logs/` and on HPC.

## Corrected gate, unchanged experiment

The smoke now requires all of the following:

1. Identical initial parameter hashes and pre/post CPU/CUDA RNG states across
   repeated controls and KD0.
2. An actual GPU training-hook assertion that lambda zero returns the same
   hard-loss tensor object, preserving the original autograd graph.
3. Bitwise-identical dropout outputs and identical RNG draws on GPU for FP32,
   FP16 and BF16, including the recorded eligible-row mask.
4. Full-model logit, loss, prediction and post-step weight differences within
   twice the measured base-repeat variation, with small numerical floors and
   independent absolute caps. A large base drift also fails; the gate does not
   accept an arbitrarily noisy reference. Detailed controls are always saved.
5. The original positive-KL BF16 backward, FP16 inference, standard checkpoint
   roundtrip, finite gradients, parameter-count and teacher-alignment checks.

64 local tests passed, including rejection tests for changed initialization,
RNG, loss-graph identity, excessive KD drift, and unstable base repeats. Only
smoke/preflight/submission code and tests changed. All `src/` hashes and the
training entrypoint are identical to the failed snapshot. Configs are identical
except output directories. Temperature, lambda, seed, data, teacher, optimizer,
precision and stopping remain unchanged.

## Retry

Fresh root: `/hpc2hdd/home/shuang886/PLC/experiments/self_distill_gfgf_20261012_r1`.
The failed original snapshot was not overwritten. The completed TRAIN cache
is validated and reused via `--cache-ready`, avoiding dependence on an old
completed job that may have left the live Slurm database.

| Stage | Job | State at first retry check |
|---|---|---|
| Corrected GPU smoke | 12984984 | PENDING, Priority |
| T=2, lambda=.05 scratch | 12984985_0 | PENDING, afterok smoke |
| T=2, lambda=.10 scratch | 12984985_1 | PENDING, afterok smoke |

Retry 1 also stopped at the numerical gate (1m52s, not OOM). Initial weights,
RNG, exact dropout arithmetic and loss-object identity passed. Logit RMS for
KD0 was 0.00127675 versus base-repeat maximum 0.00128384. The sole failing
aggregate was loss difference 0.00017297268 versus bound 0.00017283989, a
1.33e-7 excess. The relative noise estimate from three baseline repeats was
too tight for the nonlinear reduction of BF16 logits.

Retry 2 keeps the independent logit/state caps and exact structural checks,
but gives aggregate loss a 1e-4 noise floor and argmax fraction a 0.001 floor.
The absolute loss cap remains 0.002; argmax cap remains 0.01. Regression tests
still reject a real loss change and excessive logit drift. **65 tests pass**.
The r1 numerical report passes the corrected rule; a fresh GPU smoke is
required as well, rather than manufacturing a pass artifact from old logs.

Current root: `/hpc2hdd/home/shuang886/PLC/experiments/self_distill_gfgf_20261012_r2`.

| Stage | Job | State at r2 submission check |
|---|---|---|
| Fresh GPU smoke | 12985072 | PENDING, Priority |
| T=2, lambda=.05 scratch | 12985073_0 | PENDING, afterok smoke |
| T=2, lambda=.10 scratch | 12985073_1 | PENDING, afterok smoke |

All tasks use one GPU on `i64m1tga800ue`. No unrelated CPU/GPU jobs were
cancelled. Both previous dependent arrays were automatically cancelled after
their respective smoke failed. Training remains blocked until the new smoke
really passes. This is an execution repair, not an experimental result.
