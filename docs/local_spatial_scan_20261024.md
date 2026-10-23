# Local spatial scanning pilots

User-approved scope: G0/S0/S1/T1, seed44 from scratch. MoE deferred.
Architecture: [experiment plan](../refine-logs/local_spatial_scan_20261024/EXPERIMENT_PLAN.md).
Deployment: [tracker](../refine-logs/local_spatial_scan_20261024/EXPERIMENT_TRACKER.md).

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
