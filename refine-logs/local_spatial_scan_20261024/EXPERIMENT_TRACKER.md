# Local spatial scanning experiment tracker

Date: 2026-10-24. G0/S0/S1/T1 approved, implemented and submitted.
The detailed definitions and decision gates are in [the plan](EXPERIMENT_PLAN.md).

| ID | System | Purpose | Seed | Stage | Status |
| --- | --- | --- | --- | --- | --- |
| R0 | Full V11m13 GFGF | Matched mainline control | 44;43/45 if needed | Confirmation | Planned; old run is reference only |
| G0 | Width256 global | Capacity control, same depth | 44 | Pilot | Smoke 13289651_0 running; train 13289792_0 dependency |
| S0 | Static local GVP | Shared local baseline with ligand readout | 44 | Pilot | Smoke 13289651_1 queued; train 13289792_1 dependency |
| S1 | Ligand-conditioned local GVP | Conditional spatial operator | 44 | Pilot | Smoke 13289651_2 queued; train 13289792_2 dependency |
| T1 | Local triangle pair attention | Alternative local geometry | 44 | Pilot | Smoke 13289651_3 queued; train 13289792_3 dependency |
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

## Deployment

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
