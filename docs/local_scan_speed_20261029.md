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

## Deployment

Submitted2026-10-29 23:29 CST as job **13402625**, `localscan_speed`, on
`i64m1tga800u`. Confirmed one GPU,8 CPUs,100G and4h limit. At23:29 it is
PENDING(Priority); no GPU timing results exist yet. The af3 environment's
entrypoint import/argument dry run passed.

- Root: `/hpc2hdd/home/shuang886/PLC/experiments/local_scan_speed_u_20261029`
- Immutable source: `code_v1`, commit `9e9f56b`, pushed to `plc/master`.
- Archive SHA256: `0d216501127823b027932d24683feae00b3c72af89d58d853131be70bc815bc7`
- Outputs: `speed_profile.json`, `logs/profile_13402625.out`, `logs/profile_13402625.err`.
- Original LocalScan and queued sequence-context task snapshots were not changed.

## 2026-10-30 Result: GPU Gate Failed Before Timing

Checked at about21:58 CST. Job13402625 ran09:35:33-09:38:17 and exited
FAILED(1:0), not OUT_OF_MEMORY. The failure is in the S0 ten-step
legacy-versus-batched statistics replay, at the full weight comparison:

```text
rtol=1e-5, atol=1e-6
mismatched elements in the reported tensor: 42351 / 163840 (25.8%)
maximum absolute difference: 0.0003909748047590256
```

`speed_profile.json` contains `passed:false`, empty `parity`, `runs`,
`cap_stress` and `summary`. Therefore there is NO valid throughput, speedup,
or stress-test VRAM result. The6.5s/6.3s printed ten-step replay durations
are not benchmark measurements. Earlier parity subchecks did not persist
their individual results before this exception; do not claim a complete GPU pass.

GPU aggregation nondeterminism is a hypothesis, not a confirmed cause.
The GVP path uses `torch_geometric.utils.scatter(..., reduce="mean")`;
independent stochastic optimizer trajectories are not sufficient to isolate
whether read-only diagnostics caused the difference. In particular, CPU
bitwise tests do not establish GPU trajectory equality. No tolerance has
been relaxed and no optimization has been enabled in existing training runs.

### Independent Diagnostic

Added `scripts/diagnostics/audit_local_scan_parity.py` and
`scripts/slurm/run_local_scan_parity_audit.slurm`. This uses one GPU,8 CPUs,
30min maximum on `i64m1tga800u`, with the same original TRAIN batch:

1. Four ten-update replays: legacy A, legacy B, batched A, legacy C. Require
   bitwise identical initial states and identical initial RNG states. Record
   every pairwise final-weight difference using the ORIGINAL tolerances.
2. On one shared set of finite tensors/gradients, compare metric dictionaries
   and gradient statistics and check that weights, gradients, inputs and RNG
   are not mutated. This separates helper equivalence from replay variance.
3. Persist intermediate results. Neither legacy self-variance nor a completed
   diagnostic automatically approves the speed optimization.

Local verification:61 related CPU tests passed, including a negative test
that deliberately mutates a gradient and must be rejected. Python compilation
and Slurm shell syntax pass. No TEST/VAL forward or checkpoint is involved.

At22:12 CST, follow-up GPU submission is NOT complete: repeated SSH attempts
were reset by the remote host after the successful status/log collection.
No follow-up job ID is claimed. The original speed matrix remains gated.
