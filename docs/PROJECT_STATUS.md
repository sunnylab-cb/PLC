# PLC status as of 2026-10-16

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

### ESM2 Coverage Recovery Submitted October 16

The user requested further ESM2 coverage. New jobs **13124029 -> 13124030 ->
13124031** audit the5,377 previously failed chain keys directly in existing
CIFs, then run a three-chain GPU smoke and full ESM2 recovery. This removes the
unnecessary V4-file prerequisite for sequence inference, without pretending
that V4/geometry/axis QC has passed. Missing or ambiguous CIF axes are rejected;
new sequence features remain quarantined outside current feature/overlay paths.

At17:32 UTC+8 all three jobs are PENDING; a later scheduler check reports audit
Resources with provisional start October17 00:57:01, GPU jobs Dependency.
CPU=`i64m512ue`, GPU=`i64m1tga800ue`; old ligand13077843 and QC13077847 are
untouched. Source48e23bd is pushed, archive/scripts hash-verified,58 tests pass.
Actual recovered coverage awaits job output. See
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
