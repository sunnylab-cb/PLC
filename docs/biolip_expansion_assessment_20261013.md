# BioLiP expansion assessment

Date: 2026-10-13. Initial status: metadata/file-presence assessment completed.
Follow-up: jobs13003040 and13003132 completed actual candidate tensor QC and
selective extraction of1,390 known-chain ligands. See the
[PDBbind funnel and expansion update](pdbbind_coverage_and_biolip_expansion_20261013.md).
No new training allowlist, homology search or model training. The ceilings
below remain the original pre-extraction snapshot, not current admitted counts.

## Decision

Expansion is worth testing, but the useful intervention is broader compatible
chemistry and independent pocket observations, not removing every filter.
Keep leakage prevention, atom/residue alignment and geometry QC. Reconsider
the legacy route5 whitelist, redundant observations and historical exclusions.
There is no measured prediction gain from expansion yet.

The current merged TRAIN has 10,317 pairs: 5,160 PDBbind + 5,157 BioLiP.
VAL and TEST remain 647 PDBbind pairs each. The approved BioLiP subset covers
2,131 PDB entries and 4,357 selected-chain keys. All its metadata ligands have
6-50 heavy atoms. The 704,660 cached BioLiP rows are not 704,660 independent,
feature-ready small-molecule training examples.

## Measured candidate ceilings

The audit streams the 2026-08-11 full metadata ledger and refreshes file
presence on 2026-10-13. "Core" means fixed receptor, protein V4 and chain ESM2
files exist. "All assets" additionally requires a ligand feature file. These
counts do NOT establish valid tensor contents, compatible normalization,
complete labels, chemical identity or homology clearance.

| Metadata policy | Eligible rows | Core files present | All files present | New all-file candidates after direct evaluation-PDB exclusion |
| --- | ---: | ---: | ---: | ---: |
| Old route5, heavy atoms 6-128, contact residues >=3 | 188,124 | 38,218 | 8,517 | 2,446 |
| Same route5, remove upper heavy-atom limit | 188,124 | 38,218 | 8,517 | 2,446 |
| Compatible chemistry, heavy atoms 6-512, contacts >=3 | 332,326 | 71,667 | 9,962 | 3,333 |
| Same chemistry, heavy atoms 3-512, contacts >=1 | 379,903 | 79,723 | 10,261 | 3,549 |
| All types, exact mapping, >=1 heavy atom and contact | 607,405 | 136,846 | 12,973 | 5,718 |

Compatible chemistry here means metadata categories regular organic,
nucleotide/cofactor, carbohydrate/glycan, lipid and metal complex. It excludes
pure ions, inorganic polyatomic ligands, peptides and nucleic-acid fragments.
All policies require complete exact mapping and nonempty pair identifiers.
Metadata contact counts still need geometric recomputation on the input axis.

The final column removes current BioLiP TRAIN rows and direct PDB-ID overlaps
with current VAL/TEST and COACH420/HOLO4K. It does NOT remove homologs. Rows
across policies overlap; do not add table rows together. In particular, 71,667
is a pre-filter ceiling including current rows, not an approved expanded TRAIN.

The broader chemistry policy increases core-file candidates by 33,449 (+87.5%).
Only 253 of its 332,326 metadata rows exceed 128 heavy atoms: 252 have 129-256,
and one has 257-512. Most of the opportunity is chemistry/route coverage,
including the 53,508 rows with 51-128 heavy atoms, not extremely large ligands.
Metadata heavy-atom limits are different from the model's total graph-node cap
(currently 256, potentially including hydrogen atoms).

## Why the current subset is so small

### 1. Route5 hides a stricter size and chemistry filter

[ligand_biolip_types.py](../preparation/ligand/ligand_biolip_types.py:692)
changes regular organic ligands with >50 heavy atoms or >20 rotatable bonds to
the no-conformer route4. Known cofactors and other biological types can also
use route4 without being large. The old metadata policy requires regular-organic
route5, and [tensor QC](../scripts/diagnostics/prepare_biolip_v11_pilot.py:238)
again requires actual route5 and completeness >=0.8.

Consequently, removing only the nominal 128-heavy-atom cap adds exactly ZERO
metadata candidates in this cache. A route denotes an extraction strategy,
not a biological relevance or quality grade. Route4 should not automatically
mean unusable ligand.

### 2. Historical filtering happened before the new strict split

The old sequence audit retained 9,721 candidate pairs against external
benchmarks alone, but 6,999 against external benchmarks plus the OLD internal
VAL/TEST. The difference, 2,722, is a pool to reconsider, NOT a recovery count.
Those 6,999 extracted tensors included actual routes2/3/4/5 =
811/186/208/5,794. Route5 plus geometry QC retained 5,786. The new split then
filtered this already-small pool to 5,158, followed by one final audit removal
to 5,157. It never reconsidered the full 704,660-row ledger.

Also, the old manifest rejects ALL PDBbind PDB-ID overlaps. The broader
chemistry pool includes 9,525 observations at current PDBbind TRAIN PDB IDs,
but only 17 have all files. Such overlap need not be evaluation leakage;
retention still requires exact pair/site deduplication and current homology QC.

Old homology passes cannot be reused as current certificates: that search used
query-only coverage, max-seqs 1,000 and concatenated target sequences. The
current protocol searches actual chains and filters on query OR target coverage.

### 3. Feature availability limits usable scale

Fresh inventory: 110,092 protein V4 files, 80,521 chain ESM2 files, 137,069 fixed
receptors and 13,673 ligand tensors. These have different counting units.
Among compatible-chemistry metadata rows, 171,130 have a protein file but no
matching chain ESM2 file. These are pair rows, not unique sequences or an ESM
runtime estimate. Deduplicate exact sequences before scheduling extraction.

### 4. Missing-feature masks are not consumed by the current loader

V6 stores `x_group_mask [N,14]` and `x_group_source` to distinguish unavailable
charge/SASA/flex/shape values from genuine zeros. Successful route3/4 extraction
can have 13/14 valid groups, with conformer-derived RMSF missing; pure ions can
have only 1/14. These are typical cases, not guarantees for every tensor.

[PocketDatasetV8](../src/data/pocket_dataset_v8.py:574) loads atom features,
coordinates and edges, but does not propagate these validity/source fields to
the model. See the [V6 specification](../DOCs/LIGAND_FEATURE_EXTRACTION_V6_MIN.md:209).
Broader input must be audited per feature group. Do not silently normalize
missing sentinels as real measurements. If mask-aware handling is introduced,
apply it to a matched old-data control as well as the expanded-data arm.

### 5. Current cluster weighting is effectively uniform

The merged split has two columns, while the
[loader](../src/data/pocket_dataset.py:155) reads cluster size from column four
and defaults to one. Current training logs report cluster size one for all
11,611 rows and weight range [0.9102, 0.9102]. This gives no relative balancing
between clusters, despite the weighting option being enabled. Connected-
component split separation and cluster-balanced sampling are different things.

Uniformly adding many repeated cofactor sites can increase their weight without
adding independent tasks. In the compatible metadata pool, CLA/HEM/PEB alone
have 26,045/14,919/13,302 rows. The all-types pool includes 208,503 pure-ion rows.
Neither different CCD IDs nor different PDB IDs imply independent chemistry.

## What is most worth recovering?

### Low-cost existing files

Recheck the 2,446 old-policy or 3,333 broader-chemistry new all-file candidates
under the FINAL frozen holdouts, actual tensor QC and pair/site deduplication.
Many may be historical failures or homologs; file existence is not eligibility.

### Targeted additions on existing TRAIN chain keys

| Candidate policy, excluding existing TRAIN pairs | Additional rows on known chain keys | All files present | Ligand missing, core present |
| --- | ---: | ---: | ---: |
| Old route5 policy | 104 | 104 | 0 |
| Compatible chemistry 6-512, >=3 contacts | 1,494 | 104 | 1,390 |
| Compatible chemistry 3-512, >=1 contact | 1,966 | 104 | 1,862 |

This is the most economical targeted preparation opportunity: reuse protein
and ESM inputs, evaluate the 104 existing tensors first, then consider the
1,390 missing ligands. These 1,494 are additional ligand observations, not
1,494 proven new pockets. A known `(PDB,chain)` key must still match the exact
approved sequence and residue axis before treating it as reusable.

### Larger expansion, only after measuring the clean yield

Among the broader chemistry pool, 56,635 new core-file candidates remain after
direct evaluation-PDB exclusion and removal of current TRAIN rows. This is the
next preparation ceiling before homology, deduplication and tensor QC. Do not
extract all their ligand features before rejecting homologs and duplicates.

## Is this likely to improve HOLO4K assignment?

There is a plausible opportunity, not a measured improvement. A cheap proxy
counts chain groups containing different CCD IDs with <=2 overlapping metadata
binding residues and Jaccard <=0.2:

| Pool | Chain groups | Groups with distinct masks | Different-CCD, low-overlap groups |
| --- | ---: | ---: | ---: |
| Current approved BioLiP | 4,357 | 648 | 157 |
| Old-policy, core files present | 26,945 | 4,202 | 1,771 |
| Broader chemistry, core files present | 45,426 | 11,028 | 6,215 |

The two candidate rows exclude direct evaluation PDB IDs, but not homologs.
They lack centroid, coordinate-frame, chemical-equivalence and residue-axis
checks. Thus 6,215 is not a valid-negative count. The
[previous strict audit](strict_assignment_results_20260920.md:28) found only
66 eligible geometry/chemistry groups and 179/10,317 TRAIN rows with conservative
same-geometry distinct-chemistry peers. Different chemistry still does not
prove biological nonbinding; these are supervised distinct-site comparisons.

Another remaining mismatch is scope: the BioLiP materializer slices the selected
protein chain, whereas full-protein benchmark cases can include multiple chains
and symmetric copies. More chain-level data need not teach full-receptor
assignment. Keep scope fixed for the initial data-only comparison; whole-
receptor BioLiP is a separate variable requiring homology checks on EVERY
input chain and a symmetry/equivalent-site evaluation policy.

## Expansion rules and acceptance

1. Freeze current VAL/TEST and external benchmark identities, labels and metric
   definitions. Use only TRAIN for normalization and sampling-statistic fitting.
2. Build candidates from the full ledger, not only the old 5,786 materialized
   pairs. Exclude direct evaluation IDs, but do not blanket-exclude TRAIN IDs.
3. Use connected components over the homology graph, not representative-only
   clustering. Newly added sequences can bridge to evaluation components.
   Reject such additions; do not move existing holdouts to accommodate them.
4. Independently audit final TRAIN-vs-VAL/TEST/external chain alignments using
   identity >=0.30 AND (query coverage >=0.80 OR target coverage >=0.80) as the
   exclusion predicate. The existing audit uses search floor 0.28, sensitivity
   7.5 and max-seqs 100,000. This is a coverage-qualified search audit, not a
   proof that every possible local sequence alignment has identity <0.30.
5. Deduplicate equivalent structure/chain/ligand/site observations while keeping
   distinct ligands and distinct sites on the same protein. Audit feature masks,
   atom/residue alignment, contact labels, finite values and normalization.
6. Start with ordinary organic/cofactor/glycan chemistry. Handle pure ions,
   nucleic-acid fragments and long peptides as separate task/type strata, not
   silent replacements for the current small-molecule task.
7. Compare original GFGF scratch on old vs expanded TRAIN at matched optimizer
   steps, schedule and source mixture. No self-distillation, added context noise
   or architecture change. Any new mask/sampler behavior needs the same old-data
   control. Select checkpoints using the existing best-VAL-F1 convention.
8. Judge real F1/AUPRC and ligand-shuffle sensitivity, then protein-balanced
   external per-ligand F1 and multi-site assignment. Do not count a better
   threshold, conditional BP score or a larger zero-ligand gap as sufficient.

The [staged plan](../refine-logs/biolip_expansion_20261013/EXPERIMENT_PLAN.md)
keeps these comparisons small. Selective preparation has since completed;
no new training has been submitted.

## Literature boundary

BioLiP2 applies geometric rules and literature validation to biological ligand
relevance, and covers interaction types beyond conventional small molecules.
This supports separating our model-compatibility restrictions from biological
quality; it does not establish that every BioLiP row will improve this task.
[BioLiP2 paper](https://pubmed.ncbi.nlm.nih.gov/37522378/).

PLINDER explicitly maintains nonredundant annotations, similarity clusters and
split/quality metadata. That is a useful data-engineering precedent for treating
quality, redundancy and leakage as separate axes, not a prediction of our gain.
[Official dataset documentation](https://plinder-org.github.io/plinder/dataset.html).

## Reproducibility

CPU job **13001698**, `emergency_cpu`, completed exit 0 in **95 seconds**;
two allocated CPUs, 8G allocation, MaxRSS 817,188 KiB. No tensor loading or GPU
work. Remote root:
`/hpc2hdd/home/shuang886/PLC/experiments/biolip_expansion_audit_20261013/`.

Local evidence: [summary.json](../tmp/biolip_expansion_20261013/summary.json),
[accounting](../tmp/biolip_expansion_20261013/sacct.txt), and hash-verified
`sync_manifest.json` plus six historical summaries in the same directory.
The summary records source sizes, timestamps and hashes. Script:
[audit_biolip_expansion.py](../scripts/diagnostics/audit_biolip_expansion.py).
Six focused unit tests and Slurm syntax checks pass.

| Input | SHA256 |
| --- | --- |
| Historical full metadata gzip | `96d40774b89c6b0915dae7f2745990522a9bde1929a491031d014156ebb23f12` |
| Final BioLiP TRAIN IDs | `663c0ba1a3fc43fe079a180066260acdc36f0aac5140409c6d8a240887911cbc` |
| Final PDBbind split | `42e5fdf2449122302a4226efd0757ec9a212ee4f9255c5828f5a6fa250f67d72` |
| Audit script used by job 13001698 | `5a9ffda96638621d82c6dc94f5538475af1756c0b9ac9f88b2e893ea0d4da806` |

No original data, split, model, running job or remote shell configuration was
changed. Only small audit artifacts were transferred, not tensors/checkpoints.
