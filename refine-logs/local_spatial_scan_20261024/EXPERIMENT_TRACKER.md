# Local spatial scanning experiment tracker

Date: 2026-10-24. G0/S0/S1/T1 approved and implemented; GPU smoke pending.
The detailed definitions and decision gates are in [the plan](EXPERIMENT_PLAN.md).

| ID | System | Purpose | Seed | Stage | Status |
| --- | --- | --- | --- | --- | --- |
| R0 | Full V11m13 GFGF | Matched mainline control | 44;43/45 if needed | Confirmation | Planned; old run is reference only |
| G0 | Width256 global | Capacity control, same depth | 44 | Pilot | Implemented; GPU smoke pending |
| S0 | Static local GVP | Shared local baseline with ligand readout | 44 | Pilot | Implemented; GPU smoke pending |
| S1 | Ligand-conditioned local GVP | Conditional spatial operator | 44 | Pilot | Implemented; GPU smoke pending |
| T1 | Local triangle pair attention | Alternative local geometry | 44 | Pilot | Implemented; GPU smoke pending |
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
