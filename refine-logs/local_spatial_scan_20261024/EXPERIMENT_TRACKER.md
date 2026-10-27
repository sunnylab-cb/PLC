# Local spatial scanning experiment tracker

Updated: 2026-10-27. Four runs active after about58h; local arms have plateaued.
The detailed definitions and decision gates are in [the plan](EXPERIMENT_PLAN.md).

| ID | System | Purpose | Seed | Stage | Status |
| --- | --- | --- | --- | --- | --- |
| R0 | Full V11m13 GFGF | Matched mainline control | 44;43/45 if needed | Confirmation | Planned; old run is reference only |
| G0 | Width256 global | Capacity control, same depth | 44 | Pilot | 13312384_0 RUNNING epoch42; VAL through epoch41 |
| S0 | Static local GVP | Shared local baseline with ligand readout | 44 | Pilot | 13312384_1 RUNNING epoch23; VAL through epoch22 |
| S1 | Ligand-conditioned local GVP | Conditional spatial operator | 44 | Pilot | 13312384_2 RUNNING epoch18; VAL through epoch17 |
| T1 | Local triangle pair attention | Alternative local geometry | 44 | Pilot | 13312384_3 RUNNING epoch22; VAL through epoch21 |
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

## 2026-10-25 Running Check

Observed around 12:40 CST. All `13312382_[0-3]` COMPLETED with exit0;
the last smoke ended04:23:40. Full training `13312384_[0-3]` started at
04:41:04 / 04:46:05 / 04:46:05 / 05:46:09 respectively. All four are RUNNING
on `i64m1tga800u`, GPUs allocated by Slurm. No OOM/traceback/non-finite-logit
error found. No job was cancelled, resumed or reconfigured in this check.

Latest completed full-VAL metrics (not TEST, not smoke):

| Arm | Completed VAL epoch | Legacy F1_c0 | Pooled F1@0.5 | Pooled MCC@0.5 | Pooled AUPRC | Pooled AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | 5 | 0.5551 | 0.5926 | 0.5755 | 0.5956 | 0.9493 |
| S0 | 2 | 0.4548 | 0.4777 | 0.4553 | 0.4421 | 0.9218 |
| S1 | 2 | 0.4462 | 0.4784 | 0.4593 | 0.4394 | 0.9224 |
| T1 | 2 | 0.4523 | 0.4728 | 0.4506 | 0.4406 | 0.9177 |

Best and last legacy F1 coincide so far. G0 epoch1..5 F1:
0.4503,0.5021,0.5317,0.5368,0.5551; AUPRC:
0.4453,0.5235,0.5613,0.5745,0.5956. Local epoch1->2 F1:
S0 0.4128->0.4548; S1 0.4171->0.4462; T1 0.4131->0.4523.
Same-epoch2 G0 is F1=0.5021, AUPRC=0.5235, MCC@0.5=0.5161.
Thus the early local deficit is not only unequal epoch progress. However,
local arms are still within the shared five-epoch warmup; no final rejection
or generalization claim is justified. Architecture and parameter counts also
differ, so these are not isolated locality ablations.

Source coverage is complete at each VAL: PDBbind1697 + BioLiP15698 rows.
At epoch2, PDBbind pooled AUPRC is G0/S0/S1/T1 =
0.5091/0.3676/0.3611/0.3791; BioLiP = 0.5255/0.4518/0.4499/0.4488.
The early gap is larger on PDBbind. G0 epoch5 PDBbind/BioLiP AUPRC is
0.6201/0.5920 and MCC@0.5 is0.5903/0.5733.

All logged TRAIN batch losses and total-gradient norms were finite. Recent
200 logged pre-clip gradient norms, median (max): G0 1.278(12.818),
S0 3.221(17.402), S1 3.180(10.337), T1 3.945(19.232). All recent local
samples exceed clip1; G0 fraction is78.5%. This is stronger clipping pressure,
not proof of divergence or a causal explanation for the score gap. Local
ESM/spatial/ligand/atom-QKV modules have nonzero gradient logs. Preserve the
common recipe during this early comparison; do not tune from TEST.

Measured resources (A800; smoke cached TRAIN batch16x420, then synthetic
length1800 with unchanged ligands; these are not live full-training peaks):

| Arm | Parameters | Smoke peak GiB | Cap-stress peak GiB | Full VAL-to-VAL interval hours |
| --- | ---: | ---: | ---: | ---: |
| G0 | 23,650,742 | 3.886 | 31.591 | 1.36-1.42 |
| S0 | 1,617,796 | 0.191 | 0.827 | 2.61 |
| S1 | 1,787,044 | 0.193 | 0.829 | 3.06 |
| T1 | 1,067,460 | 0.182 | 0.819 | 2.49 |

Local arms save substantial activation memory on this stress case but are
not faster end-to-end at the current implementation/batch. Timing includes
data loading, training, VAL and saving, with only one local interval observed.
Do not extrapolate the short cached-batch profile to a full-epoch ETA. Current
evidence supports waiting for more VAL rounds, not declaring a winning local
operator, changing batch sizes, or expanding MoE yet.

## 2026-10-27 Running Check

Observed around15:00-15:10 CST. All four jobs remain RUNNING on their original
U nodes after57-58h. No OOM/traceback/non-finite-VAL-logit failure; live batch
counters advance. Metrics below are full strict VAL, not TEST or benchmark.
S0 completed epoch22 during the inspection; its newest metrics are from stdout,
while the detailed TB extraction covered epoch21.

| Arm | Best F1 epoch | Best legacy F1 | AUPRC at best-F1 | MCC@0.5 at best-F1 | Last completed epoch | Last F1 | Tail5 F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | 32 | 0.601547 | 0.632418 | 0.601795 | 41 | 0.590308 | 0.588688 |
| S0 | 13 | 0.532672 | 0.538215 | 0.535717 | 22 | 0.5304 | about0.5258 |
| S1 | 12 | 0.542662 | 0.546915 | 0.545221 | 17 | 0.529617 | 0.530639 |
| T1 | 18 | 0.505269 | 0.500022 | 0.511106 | 21 | 0.501921 | 0.499380 |

Do not mix independent metric maxima: G0 epoch35 AUPRC=0.635647 and
MCC@0.5=0.602771 exceed their epoch32 values, but the retained best-F1
checkpoint is epoch32. At that checkpoint, pooled F1@0.5=0.617032, its
VAL-selected gate=0.534020, and F1@gate=0.617601. Thus this score is not
mainly a threshold-tuning gain. All source VAL counts remain1697/15698.

### Interpretation

- Global width reduction is the strongest signal so far. The historical
  full-width same-cohort run had best F1=0.588257, AUPRC=0.625655 and
  MCC@0.5=0.589897 at epoch25. G0 best-F1 improves these by approximately
  0.013290/0.006763/0.011898. This is a single-seed candidate, not a final
  claim: stopping policy differs, and historical ranking logs sometimes
  excluded nonfinite residues whereas the new VAL hook rejects nonfinite
  valid-residue logits. Re-evaluate under a common denominator for confirmation.
- G0's recent F1 oscillates around0.589 rather than holding0.602; its latest
  score is0.01124 below best. Keep the best checkpoint, not the last. TRAIN
  loss falls0.1272->0.1178 from epoch32->41 with no sustained VAL gain.
- The local gap is no longer only warmup or unequal training progress.
  At the common epoch17, F1 G0/S0/S1/T1 =
  0.5825/0.5302/0.5296/0.4957; AUPRC =
  0.6222/0.5332/0.5356/0.4905. These small local trunks also remove deep
  fusion/late-delta components and are not parameter-matched, so this does
  not disprove spatial locality or prove that global attention is necessary.
- Conditioning has a modest best-score advantage S1-minus-S0=0.009990,
  AUPRC+0.008699, but not a consistent same-epoch advantage. T1 is the
  weakest tested replacement. No evidence justifies adding MoE on top now.
- Current evidence favors retaining G0 as a candidate and, if pursuing
  locality later, testing a bounded local residual within a retained global
  trunk instead of removing the trunk. No new experiment was submitted here.

Source metrics at each arm's own best-F1 checkpoint, not same epoch:

| Arm | PDBbind AUPRC | BioLiP AUPRC | PDBbind MCC@0.5 | BioLiP MCC@0.5 |
| --- | ---: | ---: | ---: | ---: |
| G0 | 0.685824 | 0.624981 | 0.637967 | 0.596820 |
| S0 | 0.524595 | 0.540683 | 0.519555 | 0.537769 |
| S1 | 0.544186 | 0.547968 | 0.543287 | 0.545396 |
| T1 | 0.497236 | 0.500545 | 0.508807 | 0.511178 |

### Health And Timing

All sampled TRAIN losses are finite. G0 has6 nonfinite total-gradient entries
among35,959 logged observations; local streams have0 among19,118/15,414/18,886
respectively. Counts refer to sampled log points, not every optimizer update.
G0's most recent200 total-gradient observations are finite; all later full-VAL
passes accepted finite valid-residue logits. The trainer enables GradScaler
and unscales before clipping; it has nonfinite-step protection but does not
persist a skipped-update counter. Do not call the entire G0 run numerically
clean or infer an exact number of skipped updates from these sampled logs.

Direct event-stream inspection locates the six NaN norm samples at one-based
epoch/batch27/3580,28/1540,30/8100,33/3840,35/7730,41/5110. Protein and
ligand modules both have nonfinite norms at these steps; this is not merely
an isolated total-norm display issue or evidence identifying one faulty
branch. Exact root cause is not established by scalar logs alone.

| Arm | Last200 pre-clip norm median | Last200 maximum | Fraction above clip1 | Recent VAL-to-VAL hours |
| --- | ---: | ---: | ---: | ---: |
| G0 | 0.786 | 7.882 | 31.5% | 1.34-1.39 |
| S0 | 1.798 | 18.555 | 98.0% | 2.65 |
| S1 | 1.658 | 19.410 | 97.5% | 3.23-3.84 |
| T1 | 2.051 | 12.748 | 99.5% | 2.91-3.07 |

Local ESM/spatial/ligand/atom-QKV gradients remain nonzero. Median neighborhood
size is about26, with40-41% cap hits, no missing-coordinate or isolated-node
median signal. This is a diagnostic, not proof the neighbor cap causes the
performance gap. Large memory savings still do not imply faster full epochs.
No live job, optimizer, batch, F1 early-stop rule, or data was changed. The
remaining runs continue under the existing F1-patience30 policy.
