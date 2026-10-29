# LocalScan Chunk And Statistics Speed Probe

## Motivation And Scope

The Oct29 live audit found mean GPU busy time of30.6%/25.8%/29.9% for
S0/S1/T1 on A800 80GiB, with only1.60/1.77/1.83GiB device memory in the
sampled windows. All tasks already have8 CPUs and8 loader workers. CPU
dispatch, synchronization and per-chunk recomputation are candidates, not
proven exclusive bottlenecks. See the LocalScan experiment tracker.

The user approved testing larger center chunks and batched host/device
diagnostics. This is a speed-only branch, not an accuracy experiment or a
restart of the existing long runs. MLP/TCN/BiGRU queued snapshots are unchanged.

## Matrix

Each S0_local_gvp / S1_cond_gvp / T1_cond_triangle runs all six cases:

| Chunk | Statistics | Comparison |
| ---: | --- | --- |
| 128 | legacy | Original execution control |
| 128 | batched | Isolate statistics synchronization |
| 512 | legacy | Isolate larger chunks |
| 512 | batched | Combined optimization |
| 1024 | legacy | More aggressive chunks |
| 1024 | batched | Combined optimization |

One Slurm task, one GPU,8 CPUs,100G host RAM,4h limit on `i64m1tga800u`.
All18 cases run sequentially on that same GPU. Each case has3 repeats,
10 warmup steps and40 timed optimizer steps per repeat. Order is shuffled
reproducibly between repeats. Initialization/RNG resets to seed44 each case.

Retain batch16, accumulation1, K32, radius12A, two spatial blocks, checkpointing,
BF16, GradScaler, original loss including CNC/distance supervision, AdamW/LR,
data membership, TRAIN normalization and<5A labels. No torch.compile.
The three fixed CPU-cached TRAIN batches contain both data sources; the first
reuses the original smoke cohort and two are selected reproducibly from TRAIN.
There is no checkpoint initialization, no TEST/VAL forward, and no saved model.

## Implementation

`training.batched_local_stats` defaults false. It is supported only for the
three LocalScan models; G0 is explicitly rejected. Classification statistics
combine integer counts into one transfer and keep the existing batch-macro
definitions/epsilon. Total and module gradient diagnostics share one packed
transfer, preserving missing-gradient keys and nonfinite total-norm reporting.
The optimization does not mutate gradients, AMP unscaling, clipping or updates.
Gradient/metric log frequency remains unchanged. Health/CNC/loss logging is
not removed or advertised as fully asynchronous.

Chunk already exists as `model.scan_chunk`. Only its value changes; default128
is retained. Larger chunks do change dropout RNG assignment in general, so
identical stochastic trajectories across chunk sizes are NOT promised.

## Gates And Measurements

Before timing each variant, deterministic FP32 outputs and gradients are
compared across chunks on a real TRAIN batch with dropout/coordinate noise
disabled solely for the parity check. The production stochastic recipe is
then used for10 legacy-versus-batched-statistics updates, checking all weights,
reported metrics, logged values and successful optimizer-update counts.

Each timed run retains normal stochastic training and reports wall time,
pairs/s, allocated/reserved GPU peaks and update counts. A synthetic16x1800
TRAIN stress update tests every case for finite metrics and OOM. Partial/OOM
results are persisted, but the overall job does not report success if a gate
fails. Existing reference code/config/asset hashes are verified read-only;
the new profile source hashes and actual case flags are recorded in JSON.

The timing includes H2D, forward, production loss, backward, optimizer and
statistics. It excludes filesystem loading and TensorBoard file I/O, using
an in-memory scalar recorder for both arms. Thus it is a kernel/loop speed
comparison, NOT a full-epoch ETA or proof that storage cannot bottleneck.

Adopt a configuration only after complete repeated GPU results and cap-stress
passes. Follow with a short end-to-end loader benchmark if gains are credible.
Do not edit immutable/running training snapshots or claim a measured speedup
before these results exist.

## Local Verification

182 related CPU tests passed, including exact metric dictionaries, missing/
zero/NaN/Inf gradient behavior, unchanged RNG/gradient buffers, chunk forward/
backward parity, and bitwise identical weights after real CPU trainer updates
with the statistics switch off/on. Existing training/context/self-distillation
regression tests also passed. Shell syntax and Python compilation passed.
These are not substitutes for the pending real-GPU gates.

Entrypoints:

- `scripts/diagnostics/profile_local_scan_speed.py`
- `scripts/slurm/run_local_scan_speed.slurm`
- `src/utils/batched_training_stats.py`
