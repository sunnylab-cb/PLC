# PLC status as of 2026-10-14

This is an index of verified results, not a new evaluation. It separates
training validation, held-out testing, external benchmarking and data
preparation. Older PDBbind splits and current strict splits are different
protocols: historical F1 values must not be placed in one ranking table.

## Dataset and current expansion

Approved strict TRAIN is **10,317 = 5,160 PDBbind + 5,157 BioLiP**;
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

### New internal-only protocols (submitted, not yet certified)

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
on October 14 at the user's request. Active replacements **13013561** (solo)
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
read-only global-only diagnostic gives 1,865 components (largest 6,529), but
using that rule requires explicit protocol approval and a fresh cross-set
audit, not a silent relaxation. No training has been submitted on this pool.
Joint **13013562** was running at the initial check. The user clarified that
it must be **left running and not advanced further today**; no cancellation,
resubmission, joint artifact edits or new joint training were performed.
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
No new split is certified and no training has started. Latest GFGF/V11m13
seed44 scratch training is the next step after preprocessing and loader/smoke
checks. Independent holdouts mean a cross-cohort score difference is not by
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
