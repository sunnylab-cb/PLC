# Local spatial scanning experiment tracker

Date: 2026-10-24. AMP fix deployed; new U smoke/train arrays submitted.
The detailed definitions and decision gates are in [the plan](EXPERIMENT_PLAN.md).

| ID | System | Purpose | Seed | Stage | Status |
| --- | --- | --- | --- | --- | --- |
| R0 | Full V11m13 GFGF | Matched mainline control | 44;43/45 if needed | Confirmation | Planned; old run is reference only |
| G0 | Width256 global | Capacity control, same depth | 44 | Pilot | Smoke 13312382_0 queued; train 13312384_0 dependency |
| S0 | Static local GVP | Shared local baseline with ligand readout | 44 | Pilot | Smoke 13312382_1 queued; train 13312384_1 dependency |
| S1 | Ligand-conditioned local GVP | Conditional spatial operator | 44 | Pilot | Smoke 13312382_2 queued; train 13312384_2 dependency |
| T1 | Local triangle pair attention | Alternative local geometry | 44 | Pilot | Smoke 13312382_3 queued; train 13312384_3 dependency |
| M0 | S1 uniform4 | Same expert-bank capacity control | 44 | After local evidence | Deferred |
| M1 | S1 learned moe4 | Ligand-dependent routing | 44 | After local evidence | Deferred |
| M2 | S1 learned moe4 balance | Balance ablation | 44 | After local evidence | Deferred |

All runs use the existing expanded joint GLOBAL-SI30 TRAIN/VAL membership,
TRAIN normalization and labels. Freeze input crops and one common training
recipe before deployment. No TEST or external-benchmark iteration.

Record best-F1 epoch, legacy F1, sample/pooled F1 and MCC at0.5 and frozen
per-run VAL gate, AUPRC/AUROC, Top20/Top-L, source-stratified results, measured
VRAM/throughput, parameter counts and total GPU-hours. Finalist dependency
checks stay on VAL during selection. Seed44 alone is not a final claim.

## Original UE Deployment (Superseded)

- Code snapshot: `eaa5100`, pushed to `plc/master`.
- Archive SHA256: `6f034c9480e1fe02207cf772564fc4c9322c79d3fb48dc747df036def04dc577`.
- Root: `/hpc2hdd/home/shuang886/PLC/experiments/local_spatial_scan_20261024`.
- Config/input/source hashes: `prepared/provenance.json`; no data copied or re-normalized.
- Smoke: `13289651_[0-3]`, 100 cached TRAIN steps + cap stress + real VAL.
- Train: `13289792_[0-3]`, `afterok:13289651`, `kill-on-invalid-dep=yes`.
- Every task: `i64m1tga800ue`, one A800, 8 CPUs, 100G RAM. Smoke2h/train7d limits.
- Full training also verifies all four smoke reports and their provenance hashes.
- CPU validation: 61 tests passed; separate source-metrics/shared-VAL-gate check passed.
- No completed GPU smoke or training score available at this recording. Queue
  state is a snapshot, not a claim that the full training has started.

## U Replacement

G0 old smoke failed with cache-consuming `lig_to_prot` checkpoint replay, not
OOM. `13289651` remaining tasks and `13289792` were cancelled. Preserve their
logs and snapshot; no full training progress was lost. Fix: checkpoint both
co-attention outputs in a stateless joint call, without changing model
parameters or hyperparameters. 82 CPU tests pass, including output and
gradient equivalence with dropout. New root:
`/hpc2hdd/home/shuang886/PLC/experiments/local_spatial_scan_u_20261024`.
Default partition now `i64m1tga800u`, still one GPU/task.

- Replacement snapshot: `6cdd171`, pushed to `plc/master`.
- Archive SHA256: `27e66f5e9eb0053f40ff7ca97e04453e3a46b6d5d18c03ccb98c1265d360196a`.
- Smoke array: `13290479_[0-3]`, `PENDING (Priority)` at verification.
- Training array: `13290512_[0-3]`, `PENDING (Dependency)`, `afterok:13290479`.
- Both arrays use `i64m1tga800u`, one GPU/task, concurrency4, 8 CPUs/100G RAM.
- Same configs except experiment output root; labels, features, normalization,
  seed, batch, loss, optimizer and gradient-checkpointing policy unchanged.
- No successful GPU smoke yet. The cancelled UE jobs will not write into the
  replacement root. Full training must still pass all four smoke checks.

## U Outcome And AMP Repair

The queue statements above are historical submission-time observations. Final
states for `13290479`: G0 COMPLETED (17m38s); S0 FAILED (18m12s); S1 FAILED
(12m38s); T1 FAILED (3m23s). `13290512_[0-3]` was automatically cancelled before
starting. All local arms completed BF16 TRAIN100 and synthetic cap stress,
then failed at FP16 VAL atom readout: `masked_fill(-1e9)` on Half scores.
No OOM, full-training epoch, or full-VAL model comparison exists for this batch.

G0 GPU profile: 23,650,742 parameters, 168.53s/100 cached TRAIN steps,
9.494pairs/s, 3.886GiB allocated peak at16x420; length1800 stress31.591GiB
with the same ligands. Smoke VAL uses only four rows and is not a model score.

Fix: explicit autocast-disabled FP32 QK/mask/softmax shared by atom readout and
triangle attention, finite all-masked rows. AMP value paths remain enabled.
102 local tests passed, including the regression that failed before the fix
and 20 new mixed-precision cases. Early profile/stress log records now survive
a later validation failure. No data or training-recipe change. A fresh source
snapshot/root is required; previous failed runs remain immutable for audit.

## AMP-Fixed U Deployment

- Code snapshot `91079ad`, pushed to `plc/master` before deployment.
- Root: `/hpc2hdd/home/shuang886/PLC/experiments/local_spatial_scan_ampfix_u_20261024`.
- Archive: `source_91079ad.tar.gz`, SHA256
  `dcb4d85079a7d897ba39aba0727399ffd06f8cb35543b01fff65eb13f8cd1146`;
  local/remote archive hashes match.
- Source/config/input hashes regenerated and verified in `prepared/provenance.json`.
  Manifest SHA256: `475811cadc97d245945e776259003cb319f785d41a4a5fd163e0bcabacd2354f`.
- All four YAML configs match the prior U configs except the output directory.
- Smoke: `13312382_[0-3]`, submitted 2026-10-24 23:51 CST, Priority pending.
- Training: `13312384_[0-3]`, Dependency pending, `afterok:13312382`,
  `kill-on-invalid-dep=yes`, plus all-four matching smoke verification.
- Both arrays: `i64m1tga800u`, one GPU/task, max4 concurrent, 8 CPUs, 100G RAM.
  Smoke limit2h; training limit7d. Training starts fresh, not from smoke weights.
- No active old tasks, no lost full-training progress, no TEST inference.
  At this observation the replacement smoke had not started; no new score claimed.
