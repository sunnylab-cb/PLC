# PLC status as of 2026-10-23

## 2026-10-23 Joint HOLO4K And Nonshared-PDB Analysis Complete

CPU13279553 and GPU13279554 both COMPLETED, exit0:0. Latest joint epoch25
has7245 successful PLC predictions and7238 exactly matched LiBRe rows.
Top20 sample F1/MCC: PLC.354668/.336147, LiBRe.404402/.390166;
protein-balanced Top20 F1.478915/.527826. Native sample F1.368942/.386316.
Old PLC Top20 sample F1 was.322387: observed gain about+.0323, but external
accession overlap prevents a clean generalization or causal data-gain claim.
User requested nonshared proteins. CPU13280188 COMPLETED 0:0 in4m05s;
all14476 prediction hashes/axes/labels verified and full scores reproduced.
Primary noTRAIN+VAL cohort:3215 pairs /1528 proteins. Top20 sample F1/MCC
PLC.306322/.284673 versus LiBRe.382451/.367583; protein-balanced Top20
F1.406499/.498795; native sample F1.316193/.385165. Excluding all internal
splits gives2558 pairs, Top20 sample F1.303579/.383738. Both models use
identical rows and fixed thresholds/budgets, with all four cohorts saved.
The gap widens after filtering; this is not a causal leakage estimate
because subset difficulty/composition also changes. The analysis removes
direct accession reuse only, not homologous/identical sequences or LiBRe
training overlap; no training changes or benchmark-driven selection.
See [completed full results](joint_holo_reference_20261023.md#completed-results-2026-10-23)
and [nonshared-PDB results](joint_holo_nonshared_20261023.md#completed-results).

## 2026-10-23 Latest Joint HOLO4K Reference Submitted

User requested latest epoch25 HOLO4K inference and asked about external
isolation. **Neither COACH420 nor HOLO4K was excluded** from the rebuilt
joint cohort; only internal TRAIN/VAL/TEST global SI30 is guaranteed.
Exact accession audit:122/299 COACH protein keys and1665/3430 HOLO protein
keys share a PDB ID with TRAIN (not a full pairwise-SI or identical-pair audit).
CPU13279553 is RUNNING; dependent one-A80013279554 is submitted to
i64m1tga800ue. Fixed VAL gate and Top20, BF16/batch2, uncapped atoms,
joint TRAIN-only ligand/raw-spatial normalization, existing LiBRe predictions.
50 tests passed and68 model/data files match training. No external score yet;
this is an overlap-uncertified reference, not clean external generalization.
See [isolation audit and evaluation protocol](joint_holo_reference_20261023.md).

## 2026-10-23 LiBRe Matched-TEST Comparison Finished

Job13272845 traversed all16,486 accepted rows at04:35:25 UTC+8 in12m08s.
FAILED(1:0) is the deliberate partial-coverage exit for909 preflight BioLiP
SDF failures, not OOM; all accepted predictions and final metrics are saved.
Same-subset pooled PLC vs released LiBRe: native F1.594912/.423227,
MCC.577970/.438794, AP.614588/.457398, AUROC.949701/.923496.
Prespecified equal-budget Top20 F1.562365/.469208 (+.093157) and
MCC.547368/.448652; sample-macro Top20 F1.534880/.447704.
PDBbind Top20 F1.603521/.484445; BioLiP.557294/.467331. LiBRe is slightly
higher at some wide PDBbind budgets, so this is not an all-metrics win.
Hashes, shared denominators and16,486 prediction-file identities verified.
This supports higher observed PLC performance on the matched internal TEST,
not universal SOTA or equal-data architecture superiority. No TEST-driven
iteration or new jobs. See [complete results and limitations](libre_joint_test_20261023.md#completed-matched-comparison-2026-10-23).

## 2026-10-23 LiBRe Same-TEST Evaluation Submitted

User explicitly accepts released-checkpoint overlap and requests direct
comparison on the latest17,395-row TEST, not LiBRe retraining. Submitted
CPU preparation13272415 completed. A native stereo-H graph-count check was
too strict; corrected in7f074c9 (30 tests pass). Replacement CPU13272844
(i64m512ue) and dependent GPU13272845 (one A800,i64m1tga800ue) were submitted
at03:35 UTC+8; the superseded pending GPU job13272416 was cancelled after
replacement succeeded. Active outputs are under joint_test_20261023/v3/.
Raw ESM2/official40D inputs, exact residue/label alignment,
fixed native thresholds and all prespecified Top-K budgets are retained;
no overlap filtering or TEST-based optimization. PLC reuses frozen epoch25
predictions. Corrected preflight is complete:16,486/17,395 accepted
(PDBbind1,649/1,649; BioLiP14,837/15,746),909 native-SDF sanitization failures.
All324 false stereo-H rejections were recovered. Both models will use this
same matched subset. GPU13272845 is PENDING(Priority), dependency cleared;
no comparison scores yet. See [execution protocol](libre_joint_test_20261023.md).

## 2026-10-23 LiBRe Joint-TEST Feasibility Audit

Existing pinned LiBRe checkpoint reverified and strictly loaded on CPU;
no new download, inference or training job. The official recipe does not
certify cross-set SI30: CD-HIT90 redundancy reduction, shared-PDB removal
and a Foldseek structural filter are different guarantees. Public assets
include only50 train examples, not a complete checkpoint-linked manifest.
Those examples share3 exact chain sequences with21 BioLiP TEST rows and
3 PDB IDs with7 TEST rows (23-row union, not28). These are example overlaps,
not a measured checkpoint leakage rate. The released model can be a
reference after input adaptation, but not a certified strict-SI30 baseline.
See [feasibility, evidence and comparison boundaries](libre_joint_test_feasibility_20261023.md).

## 2026-10-22 Fixed Top-K Reference Diagnostics Completed

CPU job13260590 COMPLETED 0:0 in64 seconds, without new inference or training.
All17,395 saved TEST rows pass identity/hash/control checks. Prespecified
Top-20 pooled real F1/MCC=.563148/.548118; shuffle F1=.478709, zero=.485921,
giving real-shuffle/zero gaps=.084438/.077227. Original fixed-VAL-gate real
F1 remains.595887. Zero's gated F1=.336419 at only4.81 predictions/row;
its Top-20 result shows the zero deficit is not purely lost ranking.
All fixed and length-based budgets are reported; **no best-K selection,
TEST-driven iteration or current-model LiBRe superiority claim**.
See [protocol, all budgets and limitations](joint_topk_reference_20261022.md).

## 2026-10-22 Reference-Only TEST Collected

Job13244903 COMPLETED, exit0:0, on October21 at10:25:41 UTC+8, elapsed36m50s.
All17,395 TEST rows and6,539,679 residues are present in real/shuffle/zero;
all three modes have zero nonfinite predictions and identical label axes.
For the locked epoch25 checkpoint at its fixed VAL gate0.553555,
overall real F1/MCC=.595887/.578981, AUPRC/AUROC=.615422/.950212;
shuffle F1=.484521, zero F1=.336419. PDBbind real F1=.621448,
BioLiP=.592692; their real-shuffle gaps are.088080/.114415.
These are pooled-residue metrics, not legacy batch-averaged F1_c0.
**Reference only: no TEST-driven iteration, model/threshold selection, or
new experiment recommendation.** No new job was submitted in this check.
See [complete fixed-protocol results](joint_reference_test_20261021.md#completed-results-verified-2026-10-22).

## 2026-10-21 Reference-Only TEST Submitted

Job **13244903** was submitted to `i64m1tga800ue` (one GPU) at09:48 UTC+8;
at09:51 it is RUNNING on `gpu1-38`, with no completed inference result yet.
It evaluates only the epoch25
VAL-best-F1 checkpoint with its already-selected VAL gate0.5535550117492676.
Real/shuffle/zero run sequentially over17,395 TEST rows, reporting overall
and PDBbind/BioLiP metrics. No TEST results are available at submission.
**TEST is reference-only and must not drive model, threshold or experiment
iteration. All development selection remains TRAIN/VAL-only.**
See [locked protocol and submission](joint_reference_test_20261021.md).

## 2026-10-21 Joint Training Completed

Job13187469 COMPLETED, exit0:0, at October20 15:31:33 after40h01m01s.
34 epochs finished; the immutable run still used loss-based early stopping.
Best-F1 checkpoint is epoch25: VAL F1_c0=.588257, AUPRC=.625655,
AUROC=.954814, global F1/MCC@0.5=.605089/.589897, and
F1/MCC at VAL-selected gate=.606903/.590500. Last F1=.582968;
tail5 mean=.581594. No TEST result or new training job in this inspection.
The queue was empty at09:34. Detailed [completed results](joint_global_si30_results_20261021.md)
include late plateau, baseline-comparison limitations, gradients and the
small finite-score coverage difference that final inference should audit.

## 2026-10-19 Joint Training Update

Subsequent mainline launches use VAL F1_c0 early stopping, retaining only
best-loss and best-F1 metric-best checkpoints. AUPRC/AUROC/MCC remain logged;
latest recovery checkpoints remain enabled. This does not hot-patch the
active immutable run or delete historical outputs.

At21:21 UTC+8, expanded joint scratch training13187469 is RUNNING,18 epochs
validated and epoch19 in progress. Best VAL legacy F1/AUPRC=.5879/.6263
(epoch18); its global F1/MCC at0.5=.6047/.5886. Future summaries also report
MCC at the VAL-F1-selected gate. No TEST result has been collected here.
Read-only Slurm GPU sampling observes57.13GiB used of80GiB,22.12GiB free;
historical CUDA peak is not instrumented. Batch18 is a candidate for a separate
stress test, not a live recipe change. Recent epoch time is69.48min; unchanged
loss-based patience30 would stop at epoch34 around October20 16:00 UTC+8
unless VAL loss improves. See [metrics, ETA and memory inspection](joint_training_progress_20261019.md).

## 2026-10-18 Recovery Update

SI13146925 completed successfully:2,163,997,578 all-pairs comparisons and
902,795,631 independent cross-set comparisons, zero global SI>=0.30 violations.
Joint TRAIN/VAL/TEST counts are139,158/17,395/17,395.
All173,948 samples finished preprocessing at17:27. Loader QC then failed at18:08
because a loop shadowed the source-membership dictionary with a list; this was
not an OOM or data/SI failure. The dependency guard cancelled smoke13146711 and
training13146712 before they ran.

Fix095285c preserves completed data and immutable model code, reruns full loader
QC via a separately hashed overlay, then smoke and scratch training.16 targeted
tests passed, including the previously missed label-loop/source-lookup path.
Replacement jobs: **13187467** (QC, i64m512ue), **13187468** (smoke,
i64m1tga800ue), **13187469** (training, i64m1tga800ue). At submission QC is
PENDING(Priority), others PENDING(Dependency); training has not started.
Remote receipt/logs: `joint_global_si30_gfgf_20261017/loader_recovery_20261018/`.

Latest scheduling update (15:31 UTC+8): expanded joint-cohort SI moved from
single-node13146709 to **13146925**, RUNNING on four `i64m512ue` nodes,
256 CPUs total. All four ranks verified;6,260,000 previously completed pairs
reused. Preparation13146710 now waits for the new SI job, followed by
smoke13146711 and scratch training13146712. Training has not started.
See [handoff details](joint_global_si30_training_20261017.md).

This is an index of verified results, not a new evaluation. It separates
training validation, held-out testing, external benchmarking and data
preparation. Older PDBbind splits and current strict splits are different
protocols: historical F1 values must not be placed in one ranking table.

## Dataset and current expansion

The original externally clean strict TRAIN is **10,317 = 5,160 PDBbind + 5,157 BioLiP**;
VAL and TEST each contain 647 PDBbind rows. No expansion rows have been
admitted and no split or normalization policy changed in the October 13 audit.

The inspected PDBbind snapshot has 19,037 complexes. Its original TRAIN was
15,229; legacy ID exclusions, external chain-homology exclusion and the
strict split explain the present 5,160, not missing feature files.
The external homology predicate is identity >=0.30 with query OR target
coverage >=0.80. Connected components keep all members together, rather than
retaining only one representative.

CPU jobs 13003040 and 13003132 completed. Selective extraction produced
**1,390/1,390** new ligand tensors on known TRAIN chain sequences, all passing
first-stage tensor QC (391 route3, 999 route4). They still require residue-axis
alignment, native heavy-atom geometry/labels, chemical/site deduplication and
validity-aware normalization. Additional observations on known chains are not
new protein families. Another 104 PDBbind recovery candidates need a new
homology audit. None is approved merely because its features exist.

Details: [coverage and expansion](pdbbind_coverage_and_biolip_expansion_20261013.md),
[expansion assessment](biolip_expansion_assessment_20261013.md).

### Latest Extraction Check, October 17

**15:21 UTC+8 update:**64-worker ligand13133479 COMPLETED at12:08:01 after
9h19m13s, peak RSS364.07 GiB/448G; no OOM. It safely replaced13077843 at02:48.
Remaining-manifest outputs149,860/149,870; all10 failures are K6G molecule-read
failures, so the extractor's own summary correctly remains partial. Final
QC13077847 COMPLETED at12:33:04 and produced a verified **173,948-row pool:
154,930 BioLiP +19,018 PDBbind**,65,788 unique sequences. This is a QC pool,
not a certified TRAIN split.

The user authorized training. New independent joint pipeline **13146709 ->
13146710 ->13146711 ->13146712** is submitted: global-SI audit, TRAIN-only
normalization/loader QC, GPU smoke, then scratch GFGF seed44. SI is RUNNING
on64 CPUs in `i64m512ue`; remaining jobs are pending dependencies. GPU jobs use
one GPU each in `i64m1tga800ue`. Source33acf20 is pushed and48 tests pass.
At15:22:39 the SI stage has actually processed780,000 pairs/39 chunks,
initial7,764 pairs/s, empty stderr; the2.164-billion-pair audit is a multi-day
preparation, not an immediate GPU start.
**GPU training has not started.** See
[joint protocol, resource gates and job ledger](joint_global_si30_training_20261017.md).

Earlier checks below are retained as history:

**02:16:51 UTC+8 update:** user-requested64-worker replacement **13133479**
is submitted to **i64m512ue**,64 CPUs/448G,one node,48h limit; follow-up state
is PENDING(Resources). Old16-worker13077843 continues while it queues. Once
allocated, the successor verifies job/source identity and QC dependencies,
cancels only the old ligand job, waits for termination and the stage lock,
then resumes the original immutable extractor, skipping existing atomic
outputs. Chemistry settings are unchanged. QC13077847 now waits on
`afterok:13133479`. Source2cea12e is pushed,26 tests pass. Provisional compute
ETA is6-8h **after allocation**, not measured throughput and excluding queue/QC.

The preceding16-worker diagnosis was:
At01:54:57 UTC+8, ligand job13077843 is still RUNNING after36h12m10s:
**116,788/149,870 processed (77.93%)**,116,785 successes and3 failures.
All16 workers are computing at99.8-99.9% CPU; memory is95.93/128 GiB. It is
not stuck or OOM. Recent4-12h throughput is only1,230-1,434 pairs/hour,
compared with3,227 over the whole run. Recent completed outputs are enriched
for43-heavy-atom PEB ligands in a1% metadata sample; the50-conformer,
100,000-max-iteration MMFF path is a plausible major cost, not a measured
per-operator attribution. Current conditional ETA is **23-27h**, around
**October18 01:00-05:00**, excluding final QC13077847. This estimate and the
old QC dependency are superseded once the64-worker takeover starts. See
[runtime evidence, resubmission and ETA](biolip_ligand_runtime_20261017.md).

### ESM2 Coverage Recovery Submitted October 16

The user requested further ESM2 coverage. New jobs **13124029 -> 13124030 ->
13124031** audit the5,377 previously failed chain keys directly in existing
CIFs, then run a three-chain GPU smoke and full ESM2 recovery. This removes the
unnecessary V4-file prerequisite for sequence inference, without pretending
that V4/geometry/axis QC has passed. Missing or ambiguous CIF axes are rejected;
new sequence features remain quarantined outside current feature/overlay paths.

**October17 verification:** all three jobs COMPLETED0:0, actually starting
October16 17:42:48 and finishing17:45:26, earlier than the provisional queue
estimate. The audit resolved5,079/5,377 failed chain keys; **all5,079 new ESM
outputs succeeded**, bringing scheduled-chain cache coverage to
**110,763/111,061 (99.73%)**. Another298 chains remain unresolved. New cached
chains cover14,143 candidate pairs, but their V4/geometry admission is pending.
CPU=`i64m512ue`, GPU=`i64m1tga800ue`; old ligand13077843 and QC13077847 are
untouched. Source48e23bd is pushed, archive/scripts hash-verified,58 tests pass.
Recovered outputs remain separate from the training/overlay namespace. See
[ESM2 recovery contract and jobs](biolip_esm2_recovery_20261016.md).

### Full BioLiP Feature Completion Launched October 15

The user authorized completing missing features rather than treating cache
absence as a permanent exclusion. The new unchanged-policy inventory contains
**188,088 deduplicated candidates**, not just the old5,786 feature-ready subset.
Missing work:179,571 ligand tensors,13,613 protein V4 files and111,061 chain keys
scheduled for ESM (exact sequences deduplicated before inference). Another2,731
rejected/legacy ligand files are regenerated in the isolated output tree.

**October 15, 22:39-22:41 UTC+8 recheck:** ligand job **13077842** completed
32,432/32,432 in4h49m29s; **13077843** is still RUNNING at59,364/149,870
with zero reported extraction failures. Protein job **13077980** ended after
7h46m48s, producing **13,530/13,613** V4 files. Its83 failures all exceed the
temporary PDB format's62-chain capacity (64-1,746 chains), not OOM.
ESM jobs **13077845 / 13077846** ended with **76,824/76,827** and
**28,860/34,234** successful chain outputs, respectively: **105,684/111,061**
in total. Failures comprise5,082 missing-V4 inputs,250 chains absent from V4
and45 duplicate residue-key cases. Slurm COMPLETED is not full data success.
Final coverage/geometry QC **13077847** is still PENDING Dependency on the
remaining ligand job. CPU=`i64m512ue`, GPU=`i64m1tga800ue`, no AMD/local resources.
The ligand and ESM smokes passed. A real long-CIF-chain-name conversion bug was
fixed without changing saved residue identities; patched protein smoke passed
and full extraction is producing tensors.49 focused tests pass. Immutable
revisions `972c36f` / `e9cd122` are pushed; no production cache/model/split changed.
The outstanding V4/ESM failures affect14,583 candidate rows; they are not
biological exclusions and need format/axis repair before admission. No final
QC pool count, expanded joint SI certificate, normalization or training exists.
This is feature preparation, not an enlarged TRAIN certificate or model result.
See [job ledger and funnel](internal_only_dual_cohort_protocol_20261013.md#feature-completion-authorized-october-15).

### New internal-only protocols (PDBbind certified and trained)

**October 14 approved follow-up:** PDBbind-only now uses global SI <0.30 as
the explicit three-way cross-set guarantee; local SI is diagnostic only.
Reuse the completed combined-rule all-pairs cache with integrity checks, then
recompute cross-set SI, normalize TRAIN only and train the exact GFGF/V11m13
seed44 recipe from scratch behind CPU/loader/GPU-smoke gates. This revision
does not edit or advance the deferred joint branch. See
[replacement pipeline](pdbbind_global_si30_training_20261014.md).
**October 15 verification:** all three replacement jobs completed successfully.
CPU preparation **13036932** and GPU smoke **13036933** passed. Scratch training
**13036934** ended October 14 at 23:37:05 UTC+8, elapsed6h36m30s, with loss-based
early stopping at epoch36 (not OOM). Certified **15,214 TRAIN / 1,902 VAL /
1,902 TEST**: all 18,801,387 cross-set chain pairs were recomputed, zero global
SI violations, maximum0.299295775. TRAIN-only normalization and loader QC passed.

Best VAL F1=**0.566083 at epoch36**, AUPRC at that checkpoint=0.587411. Best
AUPRC=**0.588037 at epoch27**. Global contact F1@0.5=0.568376 and at the VAL-fit
gate0.408461=0.569704 (epoch36). These are not TEST results or controlled gains
over the old split. Logged gradients are finite; no new TEST/ligand-dependency
evaluation, resumed training or joint-cohort processing was performed.
See [completed-run analysis](pdbbind_global_si30_results_20261015.md).
Source commit `f72ed65`, 512 deployment hashes verified, 70 local tests passed.
The following paragraphs record the original failure, not completion of the
replacement audit or training.

The user separately approved two independently rebuilt cohorts: PDBbind-only
and PDBbind+BioLiP joint. Neither uses external benchmark exclusion. Each runs
its own sample QC, exhaustive Biopython global/local SI graph, connected split,
fresh cross-set audit and TRAIN-only normalization; the solo split is not a
subset of the joint split. The >=3 actual contact-tag filter and legacy ligand
chemistry/validity gates are retained. These branches do not replace the
externally clean mainline described above.

The final user decision retains **native heavy-atom d_min <5.0 A**, not 5.5 A
and not vdW. The vdW analysis is deferred. Original fixed5 jobs 13003680/13003681
and then vdW jobs 13006855/13006856 were cancelled before starting; artifacts
are retained. Fixed5 `-u` jobs 13007974/13007975 were also cancelled while pending
on October 14 at the user's request. Replacement jobs **13013561** (solo)
and **13013562** (joint) use `i64m512ue`, still 64 CPUs/160G each. Only the
partition changed; the same hash-verified code and output roots are reused.
Immutable deployment root:
`/hpc2hdd/home/shuang886/PLC/experiments/internal_si30_dual_fixed5_20261013/`.
Both actually started October 14 03:19:26 (cluster time). Solo **13013561
FAILED** at 04:12:28, elapsed 53m02s, exit 1:0, after completing all pairwise
alignments. QC retained **19,018 / 19,037** samples and 9,432 unique chains;
all **44,476,596** pairs completed, but 585,538 graph edges merge every sample
into one component. The three-way split correctly refused to break that family;
cross-set audit and TRAIN-only normalization have NOT completed.

The extra local-SI/one-sided-coverage predicate admits very short-chain hubs:
the one-residue chain `P` from 1eb1 connects to 9,282 unique sequences. A
read-only global-only diagnostic gives 1,865 components (largest 6,529).
The user subsequently approved that rule; a fresh cross-set audit is still
required, not a silent relaxation. No training had been submitted at that check.
Joint **13013562** was running at the initial check. The user clarified that
it must be **left running and not advanced further today**; no cancellation,
resubmission, joint artifact edits or new joint training were performed.
**October 15 joint verification:** that job subsequently FAILED 1:0 at October
14 04:56:37, after1h37m11s, at the same component-assignment guard (not OOM).
QC retained **24,804 = 19,018 PDBbind + 5,786 BioLiP**; all **74,231,020**
pairs across **12,185** unique chains completed. The combined-rule graph joins
all samples into one component, so no certified split/preprocessed dataset or
new joint training exists. A hash-checked, read-only global-only reconstruction
gives **2,748 components**, largest6,596; its hypothetical seed42 split is
**19,843 / 2,481 / 2,480**. This is not a fresh cross-set SI certificate or a
protocol amendment. No joint artifacts were changed and no jobs submitted.
See [joint status and source counts](internal_only_dual_cohort_protocol_20261013.md#joint-status-verified-october-15).
Both old `-u` jobs are confirmed cancelled, elapsed zero. New jobs were held
until the old jobs left the queue, then released to avoid concurrent writes.
Resubmission receipt: `resubmission_i64m512ue_20261014.json` in the experiment root.
Remote fixed5 preflights passed: 3ddq has 22 positive residues, BioLiP
1dwk_A_BS01_OXL has 5. Samples still require >=3 positives, including after the
existing residue cap. The versioned fixed5 label mask/axis is checked by the
loader; legacy caches are unchanged. **26 focused tests pass**, including both
protocols through the full tiny-cohort preprocessing/loader pipeline.
V5/V6 caches coexist in BioLiP;
old/missing-validity tensors are excluded with reasons, not silently accepted.
At that original failed-run check, no new split was certified or training
started; the replacement has since completed as reported above. Independent
holdouts mean a cross-cohort score difference is not by
itself a controlled BioLiP improvement estimate.

Protocol and entry points:
[independent internal cohorts](internal_only_dual_cohort_protocol_20261013.md).
Read-only follow-up: [PDBbind tag/geometry audit](pdbbind_tag_geometry_audit_20261013.md).
Current blocker: [short-chain connectivity diagnosis](internal_pdbbind_connectivity_blocker_20261014.md).

## Recent matched training pilots

All rows below are **seed44 strict VAL**, using legacy batch-aggregated class0
F1 at the best-F1 checkpoint. AP is global residue AUPRC at that SAME checkpoint,
not an independently selected maximum. These are not external benchmark F1s.

| Run | Best VAL F1 | AUPRC at that checkpoint | Interpretation |
| --- | ---: | ---: | --- |
| Original GFGF reference | 0.535996 | 0.546287 | Reference for the following interventions |
| Product readout | 0.527161 | 0.530372 | No gain in this pilot |
| Equal-capacity additive readout | 0.527424 | 0.536484 | No gain in this pilot |
| CNC-off | 0.535429 | 0.545365 | Close point estimate, not established equivalence |
| Residue classmass | 0.531508 | 0.539927 | Active loss, no best-F1 gain |
| Self-distillation, lambda .05 | 0.530556 | 0.543971 | No best-F1 gain |
| Self-distillation, lambda .10 | 0.529393 | 0.538321 | No best-F1 gain |
| Random context noise .05 | 0.537633 | 0.550313 | Weak single-seed lead; below +.003 screen |
| Background context noise .20 | 0.532950 | 0.546723 | No best-F1 gain; VAL gate effect is not held-out evidence |

The full noise matrix has six arms. There is no consistent targeted-background
advantage. All runs stop at epoch35 under the unchanged loss-based early stop;
that is not proof of best-F1 convergence. No follow-up arm is promoted to a
new mainline from these one-seed results. No TEST or external evaluation of
the noise/KD pilots has been completed in these reports.

Sources: [readout/loss pilots](libre_transfer_results_20261011.md),
[self-distillation](self_distillation_results_20261012.md),
[noise matrix](context_noise_results_20261013.md).

## Held-out ligand dependency

Job 12954704 evaluated both best-VAL-F1 checkpoints on the same 647 strict
TEST rows. The following is pooled contact-class0 F1 with each run's gate
selected on VAL and fixed before TEST; it is not sample-mean F1.

| Arm | Real | Shuffle | Zero | Real - shuffle |
| --- | ---: | ---: | ---: | ---: |
| FFFF | 0.527644 | 0.476445 | 0.476815 | 0.051199 |
| GFGF | 0.536686 | 0.492792 | 0.499487 | 0.043894 |

GFGF improves the real point estimate, but shuffle/zero improve more.
This does not establish stronger ligand assignment. The paired sample-mean
real-F1 interval includes zero; these are not multi-seed statistics.
See [hybrid-attention ledger](qwen_hybrid_attention_feasibility_20261006.md).

## External benchmark comparison

The approved LiBRe comparison excludes ten chemical-input failures from BOTH
models: 491 COACH420 pairs and 7,238 HOLO4K pairs remain. These are per-ligand,
contact-only sample-mean F1s with fixed decision protocols, not the original
papers' pocket-center scores or the full pre-exclusion PLC population.

| Dataset | PLC GFGF sample F1 | LiBRe sample F1 |
| --- | ---: | ---: |
| COACH420 | 0.4439 | 0.4494 |
| HOLO4K | 0.3300 | 0.3863 |

BP F1 is conditional on successful localization and can select different
subsets for each model. PDR/LAA here use a single contact centroid and are
diagnostics, not standard object-level detection metrics. Do not infer a
paired boundary advantage from conditional BP alone. Released LiBRe training
overlap is not resolved by the matched evaluation.
See [complete matched analysis](libre_benchmark_20261010.md).

## Next work, not completed claims

Finish admission QC of the isolated BioLiP expansion without relaxing frozen
holdout/benchmark rules for that original branch. The newly requested
internal-only cohorts above are a separate protocol. Preserve missing-feature masks rather than replacing
missing measurements with physical zero. Then compare a matched old-data
control against expansion from scratch. No such training was submitted by
the October 13 coverage/extraction tasks.

For reproducibility, consult the dated documents and
[experiment manifest](../refine-logs/MANIFEST.md), not this summary alone.
Small numeric exports are in
[the Git result snapshot](../results/project_snapshot_20261013/README.md).
