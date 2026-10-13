# Independent internal-only PDBbind and joint-cohort splits

Date: 2026-10-13. New protocol requested by the user. Do not overwrite the
existing externally clean split, or compare headline scores without naming
the protocol. SI computation and training are different completion gates.

**Current decision:** retain the historical fixed heavy-atom center distance
`d_min < 5.0 A`. The vdW investigation below is deferred, not the active label
rule. Current CPU jobs are **13007974** (PDBbind-only) and **13007975** (joint).

## Two independent experiments

| Cohort | Input universe | SI and preprocessing |
| --- | --- | --- |
| PDBbind-only | The full existing 19,037-complex snapshot, subject to feature/label QC | Its own unique chains, exhaustive SI graph, connected components, split, TRAIN stats and dataset root |
| PDBbind + BioLiP joint | The same PDBbind snapshot plus eligible BioLiP observations from the unfiltered metadata ledger | Recalculate the graph on the joint pool; independently split and preprocess |

The PDBbind-only split is NOT a projection of the joint split. TRAIN/VAL/TEST
in the joint experiment may all contain both sources; report pooled and
source-stratified metrics. The two independently constructed TEST sets are
different, so their score difference alone is not a controlled BioLiP benefit
estimate. Such a claim later needs an additional matched-holdout experiment.

COACH420/HOLO4K IDs and homology are not consulted. Old TRAIN/VAL/TEST labels
are ignored; the old full split text is used only as the complete ID inventory.
These models must not be presented as external-benchmark-SI-clean models.

## Keep Sample QC

No relaxation of biological/data validity is implied by removing benchmark
exclusions. Before constructing the SI graph:

- Require at least three actual class-0 contact residues with native heavy-atom
  center distance `d_min < 5.0 A`. Also require three within the unchanged model
  residue limit of 1,800. Record rejection reasons.
- Require finite features, 109D ligand atoms, 6D edges, 1280D ESM2, 62D protein
  physicochemistry, 89D raw spatial scalars and 8x3 vectors. Require exact
  V4/ESM residue keys and identities; no silent prefix truncation or zero fill.
- BioLiP retains the former regular-organic metadata scope: exact complete
  mapping, 6-128 heavy atoms and metadata contact count >=3. Actual tensor route
  must be 5, validity masks must be consistent and completeness >=0.8. Metal,
  glycan, other unexpected routes, and missing-feature rows are deferred.
- PDBbind keeps its existing V5 schema/native-coordinate provenance, at least
  six heavy atoms, and no newly imposed upper atom exclusion. The old BioLiP
  metadata eligibility bound is not the model's atom truncation parameter.
- Recompute labels on the exact input axis. BioLiP checks selected-chain CIF
  residue identity and CA coordinates against V4. Never label with optimized
  or re-centered ligand conformers.

The 704,660 raw BioLiP records are not 704,660 training-ready samples. File
presence, sequence availability, geometry QC and training admission are distinct.
Every exclusion is logged. Exact duplicate BioLiP metadata observations have an
alias ledger; same-PDB and exact-sequence copies cannot cross split boundaries.
Broader chemical/site deduplication must not be confused with protein SI.

## Active Contact Labels: Fixed 5 A

On the user's final instruction, both independent cohorts use protocol
`plc_heavy_atom_dmin_lt5_v1`: class0 for `d_min < 5.0 A`, class1 for
`5.0 <= d_min < 8.0 A`, and class2 for `d_min >= 8.0 A`. Distances are minima
over native ligand heavy atoms and the heavy atoms of each input residue.
Exactly 5.0 A is noncontact, matching the original loader; neither 5.5 A nor
a vdW radius correction is active. There is no two-atom-pair requirement.

The isolated output is `dataset/residue_ligand_distances_dcenter.pt`, containing
physical `distances`, explicit protocol, contact mask and exact residue keys.
The versioned loader validates the protocol/mask/axis, so a stray configuration
threshold cannot silently reinterpret these labels. The physical distances
remain the regression target. Old caches and existing training are untouched.
5.5 A counts are recorded only as diagnostics; vdW counts are not calculated
for these jobs. Both cohorts rerun sample QC, since label rules affect admission.

## Deferred Contact Alternative: vdW Atom Pairs

The following alternative was implemented and tested, then deferred by the
user before either vdW job started. It does not describe the active jobs.
A 5.5 A fixed threshold is NOT equivalent to this atom-type-dependent rule:

`contact(a,b) = [d(a,b) <= r_vdW(a) + r_vdW(b) + 0.5 A]`.

- **Primary:** at least two distinct heavy-atom pairs per residue (BioLiP-style).
- **Diagnostic:** at least one pair per residue (CASP9-style); retain this mask
  and all per-residue pair counts, not just the active binary labels.
- **Sample QC:** at least three positive residues under the PRIMARY rule.
  This is our project filter, stricter than BioLiP's two-residue site criterion.
- Two pairs do not require two different protein atoms. One protein atom
  contacting two different ligand atoms also counts as two pairs.
- Exclude hydrogens/deuterium. Select a single CIF alternate conformer rather
  than double-counting the same physical atom in multiple positions.

Sources: [CASP9 organizer submission definition](https://predictioncenter.org/CASP9/src/cgi-bin/index.cgi?page=format),
[CASP9 assessment](https://pmc.ncbi.nlm.nih.gov/articles/PMC3204792/),
[BioLiP official curation rules](https://www.zhanggroup.org/BioLiP/help.html).

The exact radius table is embedded in every label artifact as
`PLC_Bondi_extended_v1`, retaining the previous project table (C=1.70, N=1.55,
O=1.52 A, etc.) without its unsafe unknown-element fallback. This is an explicit
implementation of the published contact criterion, **not certification of
bitwise-identical BioLiP annotations or CASP labels**. Native ligand positions
come from `coords_original`. Where the original SDF exists, its atom elements,
order and coordinates are checked. Otherwise the raw V5/V6 identity one-hot can
identify its nine explicit heavy elements because the extractor preserves native
heavy-atom order. `Other` is never guessed to be carbon. Such rows are excluded
with a reason if no checked source can identify the element. Protein elements
come from CIF or a validated canonical amino-acid atom-name map, so CA is carbon,
not calcium. Geometry is on the exact input protein axis; repaired input receptor
atoms can differ from BioLiP's original annotation geometry.

For the existing three-class head, class0 is the active vdW mask. Noncontact
residues retain the PLC auxiliary split: physical heavy-atom d_min <8 A is class1,
otherwise class2. Raw d_min remains the distance-regression target; it is never
replaced with a surface gap or fabricated binary distance. Near/far are project
conventions, not additional BioLiP requirements.

The new output is `dataset/residue_ligand_contacts_vdw.pt` with
`contact_label_protocol`, `atom_contact_counts`, `contact_mask`,
`contact_mask_casp1`, `contact_mask_biolip2`, physical `distances` and diagnostic
`vdw_surface_gap`. The updated PocketDatasetV8 checks protocol, masks and exact
residue axes before using it. Legacy label files retain their historical behavior.
Do not feed this artifact to an old loader that only thresholds `distances`.

One real local sanity case (3ddq, 1,134 residues / 26 ligand heavy atoms):

| Definition | Positive residues |
| --- | ---: |
| Fixed d_min <5 A (current rule) | 22 |
| Fixed d_min <=5.5 A | 25 |
| vdW sum +0.5 A, >=1 pair | 13 |
| vdW sum +0.5 A, >=2 pairs | 7 |

This is one smoke case, not a cohort-wide estimate. Both cohorts independently
rerun QC before freezing their sequence pool, since changing labels changes
which samples meet the >=3-positive requirement. Retain both old-distance counts
as diagnostics. Do not interpret a new-protocol F1 change as a model improvement
relative to the old labels.

## Exact Pairwise Contract

Use `Bio.Align.PairwiseAligner`, not deprecated `pairwise2`, on every unordered
pair of exact unique input-chain sequences, without a search prefilter.

Fixed scoring: BLOSUM62; gap-open -10; gap-extension -0.5; global terminal gaps
penalized; epsilon 1e-6. Rare U/O/J use C/K/X substitution scores, but SI still
counts equality of original letters. Biopython version is recorded.

An edge exists if either:

1. Global SI >=0.30, where SI = identical letters / alignment columns, including gaps.
2. Local SI >=0.30 and either chain's consumed-residue coverage >=0.80.

Global SI is an additional guard beyond the historical local/coverage-only
contract. The local guard catches homologous domains embedded in longer chains.
Without coverage, a tiny exact local motif would spuriously connect unrelated
proteins. This is not a guarantee that every possible alignment has SI<0.30:
the contract uses the first optimal-SCORE alignment in canonical sequence-hash
orientation. The exact scoring/tie policy is part of the reproducible definition.

PDBbind's full input chains are checked, not a concatenated pseudo-sequence.
BioLiP's selected chain is its input. All observations from one PDB ID are
indivisible across both sources. Exact-sequence members and all qualifying
pairwise edges are joined by union-find connected components, not representative
set-cover clustering. Assign whole components approximately 80/10/10 with seed42;
report actual ratios and largest components, never break a family to hit quotas.

After splitting, recompute all TRAIN-VAL, TRAIN-TEST and VAL-TEST unique-chain
pairs from scratch. Check exact expected pair counts and zero violating edges.
Missing/corrupt chunks or changed contracts fail closed. Atomic chunk results
allow a long CPU run to resume. This certifies observed input sequences, not
unknown full-length proteins, ligand scaffolds or PLM-pretraining independence.

References: [Biopython pairwise alignment documentation](https://biopython.org/docs/latest/Tutorial/chapter_pairwise.html).

## Separate Preprocessing And Training

Each cohort has its own `pool/`, `split/`, `dataset/`, exclusion ledger and
hashes. Fit normalization from its TRAIN only, starting from raw ligand V5/V6
and `spatial_scalar_raw_features`, never from old full-dataset normalized input.
Apply the frozen statistics to its VAL/TEST. Preserve the legacy selective
continuous ligand normalization and spatial z-score/clamp policy; ESM2 and
fixed physicochemical encodings remain unchanged. No source feature is overwritten.

Target the latest V11m13 group-rescompat + GFGF bidirectional protein-hybrid
configuration, seed44, separately on both cohorts. Random initialization only:
old checkpoints may already have seen the new holdouts. No torch.compile,
self-distillation or background-noise experiment is added. Use VAL for checkpoint
and threshold selection, then lock both for TEST. Keep best-F1 and ranking-metric
checkpoints under the existing selection utilities.

Training is not automatically submitted by the CPU script: require completed
SI/feature preprocessing, production-loader coverage checks and a GPU smoke test
first. The current mainline and its results remain unchanged.

## Execution

`scripts/slurm/run_internal_pairwise_cohort.slurm` is submitted twice, with
different `COHORT` values. Each job requests one `i64m512u` CPU node, 64 CPUs,
160G RAM and a seven-day limit. No a128, GPU or external download is requested.
Deployment hashes source files into a new HDD experiment directory and does not
`git pull` the dirty production checkout. Job IDs are recorded in deployment.json.

Superseded fixed-5 A jobs: **13003680** (PDBbind-only), **13003681** (joint).
They were held while still pending, before any dataset processing started,
then cancelled after replacement submission. Their retained experiment root:
`/hpc2hdd/home/shuang886/PLC/experiments/internal_si30_dual_20261013/`.
Local submission receipt: `tmp/internal_si30_dual_20261013/deployment.json`.

**Deferred vdW jobs:** **13006855** (PDBbind-only) and **13006856** (joint),
same `i64m512u`, 64 CPUs/160G each. Both were still `PD (Priority)` when checked
and were cancelled on the user's instruction to return to fixed 5 A.
Root: `/hpc2hdd/home/shuang886/PLC/experiments/internal_si30_dual_vdw2_20261013_r2/`.
Receipt: `tmp/internal_si30_dual_vdw2_20261013_r2/deployment.json`.

**Active fixed-5 A jobs:** **13007974** (PDBbind-only) and **13007975** (joint).
Root: `/hpc2hdd/home/shuang886/PLC/experiments/internal_si30_dual_fixed5_20261013/`.
Receipt: `tmp/internal_si30_dual_fixed5_20261013/deployment.json`.
Both request `i64m512u`, 64 CPUs/160G, seven days. Source hashes were verified
before submission. Remote preflight under the active rule passed with 22
contacts for PDBbind 3ddq and 5 for BioLiP 1dwk_A_BS01_OXL. The old experiment
directories are retained; unrelated jobs were not cancelled.
Post-submission `squeue` and `sacct` agree: both new jobs are `PENDING (Priority)`;
both vdW jobs are `CANCELLED` with zero elapsed time. No GPU training was submitted.

The fixed5 revision passes **26 focused tests**, including strict 5 A/8 A
boundaries, rejection of fewer than three actual contacts, and full tiny-cohort
QC -> SI -> TRAIN-only normalization -> production-loader tests under BOTH
protocols. CPU script syntax and the changed-file whitespace check pass.

The old PDBbind tag cache was inspected after submission. Its positive-distance
median is 3.5705 A, but 208 complexes contain zero-distance records. Source
geometry checks on three examples reproduce near-coincident protein/ligand
atoms; the cause needs separate chemical curation, not a threshold change.
No additional exclusion was silently introduced. Details:
[tag geometry audit](pdbbind_tag_geometry_audit_20261013.md).

The initial vdW deployment (without `_r2`) submitted **no jobs**: its preflight
incorrectly stopped after 20 older V5 BioLiP tensors. The directory contains
mixed V5/V6 caches, not a separate V6 tree. The corrected preflight selects an
actual route5 V6 smoke example; full cohort QC still records every rejection.
No old holdout allowlist is used to select the new cohort. Availability of
existing V6 tensors can still reflect earlier extraction scope; this is not a
claim that all BioLiP annotations have complete features.

Remote real-data preflight passed:

| Sample | Fixed <=5.5 A | vdW >=1 pair | vdW >=2 pairs |
| --- | ---: | ---: | ---: |
| PDBbind 3ddq | 25 | 13 | 7 |
| BioLiP 1dwk_A_BS01_OXL | 7 | 4 | 3 |

The AF3 environment reports Biopython 1.87; local unit tests use 1.85.
Alignment version is pinned in each run contract, and chunk resume refuses a
changed contract. The original full local CPU suite passed **306 tests**;
the vdW revision passes **315 tests**, including geometry, mask/axis validation,
BioLiP chain-sidecar geometry, and a complete tiny-cohort
QC -> SI -> normalization -> production-loader integration test. Warnings concern unavailable
optional CUDA extensions, library deprecations and test-process fork behavior;
this is not a GPU training smoke test.
