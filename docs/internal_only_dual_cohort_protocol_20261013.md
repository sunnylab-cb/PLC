# Independent internal-only PDBbind and joint-cohort splits

Date: 2026-10-13. New protocol requested by the user. Do not overwrite the
existing externally clean split, or compare headline scores without naming
the protocol. SI computation and training are different completion gates.

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

- Require at least three actual class-0 contact residues: native ligand heavy
  atoms vs input residue heavy atoms, minimum distance <5 A. Also require three
  within the unchanged model residue limit of 1,800. Record rejection reasons.
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

Submitted jobs: **13003680** (PDBbind-only), **13003681** (joint). Both were
pending `Priority` at the first post-submission check, not running or finished.
Experiment root:
`/hpc2hdd/home/shuang886/PLC/experiments/internal_si30_dual_20261013/`.
Local submission receipt: `tmp/internal_si30_dual_20261013/deployment.json`.
The deployed AF3 environment reports Biopython 1.87; local unit tests use 1.85.
Alignment version is pinned in each run contract, and chunk resume refuses a
changed contract. Twenty-three focused tests and Slurm shell syntax passed.
The full local CPU suite also passed: **306 tests**. Warnings concern unavailable
optional CUDA extensions, library deprecations and test-process fork behavior;
this is not a GPU training smoke test.
