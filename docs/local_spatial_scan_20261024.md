# Local spatial scanning pilots

User-approved scope: G0/S0/S1/T1, seed44 from scratch. MoE deferred.
Architecture: [experiment plan](../refine-logs/local_spatial_scan_20261024/EXPERIMENT_PLAN.md).
Deployment: [tracker](../refine-logs/local_spatial_scan_20261024/EXPERIMENT_TRACKER.md).

Latest check, 2026-10-29 around20:00 CST: all four jobs `13312384_[0-3]`
remain RUNNING on U after110-111h. G0 is in epoch81, S0 epoch43, S1 epoch34,
T1 epoch42. Best VAL F1 G0/S0/S1/T1 = 0.602434/0.535962/0.549606/0.505269,
selected at epoch52/23/24/18 respectively. At G0's best-F1 checkpoint,
pooled AUPRC=0.633018 and MCC@0.5=0.601489. Local arms remain below G0
with no sustained catch-up. G0 is a promising width-reduction candidate
versus the historical full-width F1=0.588257, not yet a multi-seed or
fixed-denominator causal claim. F1 patience counters are28/19/9/23 of30;
G0 could stop in2-3h without another best. S1 may reach the seven-day wall
limit before its patience expires. Best-F1 checkpoints exist for all arms.
Sampled nonfinite gradient counts are now18/9/2/6; the latest200 samples
contain0/1/0/0 respectively. No OOM or nonfinite-valid-logit failure was
found, but no arm is wholly numerically clean. Details, source-stratified
scores and conditional timing are in the tracker. No TEST or external
inference was run, and no active training configuration was modified.
Historical deployment states below are retained rather than overwritten.

| Arm | Operator | Protein mixing | Ligand dependence |
| --- | --- | --- | --- |
| G0_small_global | V11m13 GFGF, d_model256 | Original global stack | Original group_rescompat path |
| S0_local_gvp | Two local GVP blocks | 12A, center + up to31 neighbors | Shared atom readout only |
| S1_cond_gvp | Two conditioned GVP blocks | Same as S0 | Bounded message FiLM + atom readout |
| T1_cond_triangle | Two local pair-attention blocks | Same neighbors, j-k distance bias | Bounded pair FiLM + atom readout |

Common local stem: ESM1280->128, phys62->32, spatial89->96, 12 vectors->16.
Ligand: three geometric graph layers plus direct local54 projection, both128D.
Readout: residue query over ligand atoms, concatenate residue256/context128/
pooled-ligand128, classify contact/near/far. Distance auxiliary loss retained.
There are no true protein-ligand pose distances, target-based crops or pocket
selection. Cross-chain spatial contacts are allowed. Two hops are not a strict
12A total receptive field. ESM/cached descriptors still contain original context.

Implementation details and limits:

- Neighbor ties use stable ascending input row order. Sequence offsets are
  within-chain row offsets, not PDB numbering; they are zero across chains.
- Original vector inputs are global-frame displacement/direction features.
  The local graph uses only global relative directions, not the original
  graph builder's mixed global/local-frame vector channels. Operator rotation
  tests do not claim that upstream preprocessing is rotation-equivariant.
- Center chunks128 and non-reentrant activation checkpointing bound activation
  memory. Neighbor search remains O(N^2) time with chunked distance storage.
- T1 uses vector norms, learned i-j/i-k states and j-k RBF bias. It is not a
  full AlphaFold triangle trunk and does not predict ligand pose.
- Source dropout rates are retained. In local arms geometric modality dropout
  masks the pointwise spatial/vector stem; unlike G0 it cannot mask a separate
  protein GVP branch, because the local architecture has one decision path.
- S0/S1 are a cleaner operator comparison; G0/local comparisons also remove
  deep fusion, late delta and virtual-token machinery. Not parameter-matched.
- All new runs use F1 early stopping, patience30, batch16/accum1, checkpointing.
  Save best-F1 and best-finite-loss only; log AUPRC/AUROC without extra CKPTs.
- VAL reports include source-separated pooled F1/MCC at0.5 and the combined
  VAL gate, sample-macro F1/MCC, PDB-ID-balanced F1, Top20 and Top-L ranking.
  PDB-ID balancing is not sequence/UniProt balancing. Existing best-F1 selection
  remains legacy batch-averaged argmax F1 for historical compatibility.
- Frozen expanded joint GLOBAL-SI30 inputs and TRAIN-only normalization are
  reused. TEST, COACH420 and HOLO4K are not used for pilot decisions.

Validation before submission: CPU forward/backward, checkpoint round-trip,
checkpoint/plain equality, rigid motions, ligand atom permutations, padded
residues, missing coordinates, empty ligand, disconnected far-component test,
cross-chain edge masks, singleton neighborhoods, unchanged data config.
The real GPU smoke additionally uses the production BF16 loss and FP16 VAL,
100 cached TRAIN steps, and a synthetic 1800-residue cap stress batch. Its
timing excludes data loading and must not be reported as a full-epoch ETA.

## Submission

Implementation snapshot `eaa5100` is pushed to `plc/master`; 61 CPU tests pass.
HPC root: `/hpc2hdd/home/shuang886/PLC/experiments/local_spatial_scan_20261024`.
Smoke array `13289651_[0-3]`; scratch training array `13289792_[0-3]` depends
on the entire smoke array succeeding. Both use one GPU/task on
`i64m1tga800ue`, at most four concurrent tasks. The training script additionally
checks all four successful smoke manifests against code/config/input hashes.
At submission reporting G0 smoke was running, the other smoke tasks queued,
and all full training tasks waiting on dependency. No GPU peak or accuracy
result is yet claimed. Follow the tracker for job-to-arm mapping.

## Move To U

The user requested resubmission on `i64m1tga800u`. Before migration G0 smoke
failed at backward: the V12 co-attention adapter's `lig_to_prot` consumed a
ligand-output cache that was already cleared, and could not independently
replay under checkpointing. This was not an OOM or a training-score result.
The remaining old smoke tasks and dependent training array were cancelled;
their original source snapshot/logs are retained.

The fix adds a stateless joint protein/ligand update and checkpoints that
single call. Non-checkpointed behavior, parameters, config, loss and dataset
are unchanged. Output/input-gradient/parameter-gradient parity against the
old sequential path passes with dropout0/.15 and two stacked blocks; the
full tiny-model co-attention + checkpoint backward now passes. Related suite:
82 tests passed. Smoke phase logging now distinguishes data loading from
profiling. The immutable replacement root is
`/hpc2hdd/home/shuang886/PLC/experiments/local_spatial_scan_u_20261024`.
New training still requires all four matching GPU smoke passes.

Replacement snapshot: `6cdd171`, pushed to master. New smoke array:
`13290479_[0-3]`; new training array: `13290512_[0-3]`, with
`afterok:13290479` and `kill-on-invalid-dep=yes`. Verified both arrays are on
`i64m1tga800u`, one GPU per task. At verification all smoke tasks were queued
for priority and full training was waiting on dependency. No new accuracy or
GPU-memory result is available yet.

## U Smoke Results And AMP Fix

Observed after the first U array finished (snapshot `6cdd171`):

| Arm | Job | Slurm state | Completed phases | Failure |
| --- | --- | --- | --- | --- |
| G0 | 13290479_0 | COMPLETED | 100 TRAIN steps, cap stress, FP16 VAL, strict CKPT load | None |
| S0 | 13290479_1 | FAILED | 100 TRAIN steps, cap stress | FP16 atom-readout mask overflow |
| S1 | 13290479_2 | FAILED | 100 TRAIN steps, cap stress | Same |
| T1 | 13290479_3 | FAILED | 100 TRAIN steps, cap stress | Same |

Dependent training `13290512_[0-3]` was cancelled automatically and never
started. None of these failures was an OOM. The prior G0 co-attention replay
fix passed real GPU training. Do not interpret the local failures as negative
architecture results or the tiny smoke VAL subset as generalization evidence.

G0 has 23,650,742 parameters. On an A800-80GB, 100 repetitions of the same
16-row TRAIN batch (padded length420) took 168.53 seconds, 9.494 cached-batch
pairs/second, peak allocated VRAM3.886GiB. Synthetic length1800 stress peaked
at31.591GiB; ligand sizes were unchanged from that TRAIN batch. These numbers
exclude data loading and do not bound the cost of every possible full-size
protein/ligand batch. The local arms did not persist final JSON profiles because
they failed during VAL; their logs show finite TRAIN steps and completed stress.

Root cause: `.float()` on Q/K does not prevent autocast from running their
matmul in FP16. Filling those scores with `-1e9` raises a Half overflow during
the legacy FP16 VAL path. Both local atom readout and triangle score calculation
now explicitly disable autocast for FP32 matmul/masking/softmax. Fully masked
rows return zero weights. Value projections/readout remain mixed precision.
No parameters, dataset, label, split, optimizer or loss recipe changed.

102 CPU regression tests pass, including 20 new FP16/BF16 cases covering all
three local arms in TRAIN/checkpoint backward and EVAL, all-masked keys, missing
coordinates, large attention scores and an exact FP32 reference. The new test
reproduced the original Half overflow before the fix. Smoke now prints profile
and stress metrics before VAL so a later failure cannot hide VRAM measurements.
Fresh immutable U deployment and job IDs are recorded in the tracker. Full
training remains gated on all four matching real-GPU smoke passes.

AMP-fixed snapshot `91079ad` was pushed and deployed under
`/hpc2hdd/home/shuang886/PLC/experiments/local_spatial_scan_ampfix_u_20261024`.
New smoke `13312382_[0-3]` and dependent scratch training `13312384_[0-3]`
are both on `i64m1tga800u`, one GPU/task. At 2026-10-24 23:52 CST, smoke was
Priority pending and training was Dependency pending. Input/config verification
passed; old/new configs differ only in the output directory. No full-training
results are available yet.
