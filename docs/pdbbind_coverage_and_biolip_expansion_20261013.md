# PDBbind coverage reconstruction and first BioLiP expansion

Date: 2026-10-13. Both CPU jobs completed. No training, production split change,
normalization change or automatic sample admission was performed.

## Why PDBbind TRAIN has 5,160 rows

The original training split really did have 15,229 samples. The current strict
split is a substantially different protocol, not that original split minus
VAL/TEST. Local `New/` currently contains 19,037 distinct four-character PDB
directories, each with protein PDB and ligand SDF. All 19,037 have matching ESM2,
V4 conf0 and ligand V5 files locally AND on HPC2. No 22k snapshot was identified
in these inspected inputs; this is not a statement about every PDBbind release.
The WDisk `New/` mirror has the same three year buckets and a `Broken` directory,
not an extra later-year dataset bucket.

| Stage | Removed at this stage | Remaining / allocation |
| --- | ---: | ---: |
| Original local snapshot | 0 | 19,037 |
| Original split | 0 | TRAIN 15,229 / VAL 1,903 / TEST 1,905 |
| Legacy direct benchmark PDB-ID exclusions | 1,260 | 17,777 |
| Reclustered legacy split used by V11 | 0 | TRAIN 14,221 / VAL 1,778 / TEST 1,778 |
| Chain-level COACH420/HOLO4K homolog exclusion | 11,306 | 6,471 |
| Connected-component strict split | 0 | TRAIN 5,177 / VAL 647 / TEST 647 |
| Independent final audit repair | 17 TRAIN | TRAIN 5,160 / VAL 647 / TEST 647 |

Thus `19,037 - 1,260 - 11,306 - 647 - 647 - 17 = 5,160`.
The 1,260 exclusions match the legacy PDB-ID exclusion list exactly. That list
names COACH420, HOLO4K, CryptoBench and UniSite Benchmark. It contains 5,240
total IDs, of which only 1,260 occur in the local PDBbind snapshot.

No feature-availability loss was found in this funnel. The connected-component
step does not keep just one representative per family: it assigns all members
of each component together. Its input has 1,245 components, largest size720.
Atom truncation is an independent model-input issue, not the cause of this
sample-count reduction.

Evidence: [full split](../preparation/dataset_split/mmseqs_splits_strict.txt),
[legacy exclusions](../configs/splits/excluded_samples.txt),
[legacy V11 split](../configs/splits/splits.txt), and
[remote coverage summary](../tmp/pdbbind_coverage_20261013/pdbbind_coverage.json).

## What excludes the 11,306 complexes?

An entire PDBbind complex is excluded when ANY of its input polymer chains has
alignment identity >=0.30 AND (query coverage >=0.80 OR target coverage >=0.80)
to an external benchmark chain. This protects the full receptor input, rather
than checking just the selected contact chain or a concatenated sequence.

Reparsing the saved alignment TSV exactly reproduces all 11,306 excluded IDs:

| External source | Excluded complexes |
| --- | ---: |
| COACH420 hit, including shared hits | 7,721 |
| HOLO4K hit, including shared hits | 10,950 |
| Both datasets | 7,365 |
| COACH420 only | 356 |
| HOLO4K only | 3,585 |
| Union | 11,306 |

Among those excluded complexes, their maximum qualifying identity is exactly
1.0 in 3,928 cases and >=0.9 but <1 in another 2,865. Thus 6,793 exclusions
have >=90% identity support, not just marginal 30% matches.

Sensitivity checks, NOT proposed policy changes:

- Requiring BOTH coverages >=0.80 still excludes 10,474 complexes; only832
  depend on the asymmetric coverage predicate.
- Only13 excluded complexes have ALL qualifying hits involving a shorter
  chain below50 residues; only6 satisfy the corresponding below30 condition.
- Therefore short-chain hits do not explain the bulk of the reduction.

The counts do not show an obvious bulk-filtering bug. They do show that full
external SI30 exclusion removes much of this PDBbind snapshot's family coverage.
That plausibly increases the learning/generalization challenge, but does not
by itself establish the cause of poor HOLO4K assignment.

Putting excluded complexes back into either supervised pretraining or final
training would change the external leakage protocol. A full-data model could
be a separately labeled experiment, but not an externally clean comparator.
No such relaxation was made here.

## Recoverable PDBbind candidates

All 1,260 early ID-excluded cases still have the three main feature files.
Of them, 1,156 occur directly in the current COACH/HOLO target PDB list; **104**
do not. Those104 are the defensible small recovery audit pool under the current
two external benchmarks, not 1,260 or 11,306 immediately recoverable samples.

They remain historically excluded with respect to the broader benchmark list.
They need new chain-level comparisons against frozen VAL/TEST/COACH/HOLO,
connected-component compatibility and feature/label QC. None has been added.
If CryptoBench/UniSite is used later, retain this provenance and audit it too.

The [candidate TSV](../tmp/pdbbind_coverage_20261013/pdbbind_coverage.candidates.tsv)
contains row-level flags. Use the REMOTE version for external-ID flags. The
local-only inventory v2 explicitly labels them `not_checked`; v1 used a default
zero without loading external metadata and is superseded for that column.
All local v1/v2 count statistics are identical. The remote job did load the
external metadata, so its104-candidate count is unaffected by that correction.

## BioLiP: beyond file-presence ceilings

Job **13003040** checked the actual 3,333 broader-chemistry candidates, not just
filenames. It checked 109D atom features, coordinate/edge shapes, finite values,
pair identity, heavy-atom counts, group-mask validity and completeness >=0.8.

| Actual tensor category | Rows | First-stage interpretation |
| --- | ---: | --- |
| Legacy route -1, missing group masks | 1,491 | Not admitted by the new V6 validity contract; not proof that every legacy feature is unusable |
| Route2 inorganic polyatomic | 811 | Below the compatible-feature completeness gate |
| Route3 metal complex | 186 | Passed expanded tensor gate |
| Route4 no-conformer biological | 208 | Passed expanded tensor gate |
| Route5 regular organic | 637 | Passed old and expanded tensor gates |
| Total passing expanded tensor gate | 1,031 | Still needs homology, alignment, geometry and deduplication |

The 104 existing-file BioLiP candidates on known TRAIN chain keys are a
DIFFERENT set from the104 PDBbind recovery candidates. Of these BioLiP104,
only11 pass the expanded tensor gate (one passes route5). On new chain keys,
1,020 candidates pass. No tensor in this inspected pool exceeds256 total atoms.

The cached sequences of 1,023 known chain keys were checked against exact
sequence provenance of current approved TRAIN pairs: zero mismatches. All1,390
missing-ligand candidates pass this sequence check. This does not substitute
for residue-key alignment, heavy-atom geometry or canonical/site deduplication.

## Completed selective feature extraction

Job **13003132** subsequently extracted precisely those1,390 missing ligands,
using the existing V6 implementation and cached pair manifest. It did not
refresh the 704,660-pair cache, overwrite production features or start a model.

| Result | Count |
| --- | ---: |
| Requested / produced / first-stage tensor-QC passed | 1,390 / 1,390 / 1,390 |
| Route3 metal complex | 391 |
| Route4 no-conformer biological | 999 |
| Old route5 gate passed | 0 |
| Expanded route3/4/5 gate passed | 1,390 |
| Over256 total atoms | 0 |
| Approved additional TRAIN rows | 0 |

This directly validates a PREPARATION claim: the route5 whitelist excludes
additional feature-extractable ligand observations on existing clean chains.
It does NOT yet validate a generalization, assignment or training-size claim.
These additions increase ligand observations, not independent protein families.

Outputs are isolated at:
`/hpc2hdd/home/shuang886/PLC/experiments/biolip_known_chain_expansion_20261013/prepared/ligand_features/`.

Next admission gates: exact V4/ESM residue axes; native heavy-atom geometry and
contact labels; canonical chemical/site duplicates; validity-aware normalization
and a matched old-data input-policy control. Do not manufacture missing RMSF
values or equate missing values with physically measured zero.

## Execution and checks

| Job | Work | Partition / resources | Status | Elapsed | Peak RSS |
| --- | --- | --- | --- | --- | --- |
| 13003040 | PDBbind funnel + BioLiP tensor/sequence QC | emergency_cpu, 2CPU/16G | COMPLETED 0:0 | 4m06s | 521,992 KiB |
| 13003132 | Selective ligand extraction + tensor QC | emergency_cpu, 8CPU/32G | COMPLETED 0:0 | 1m56s | 20,641,400 KiB |

Nineteen focused unit tests pass; both Slurm scripts pass bash syntax checks.
Source/input hashes were verified during isolated deployment. Synced reports
are SHA256 checked under `tmp/pdbbind_coverage_20261013/`. No large tensor or
checkpoint download, production data deletion, GPU task, git push, mainline
model edit or evaluation-split change occurred.

The current TRAIN therefore remains **10,317 = 5,160 PDBbind + 5,157 BioLiP**.
The next useful task is admission QC for the new ligand tensors, plus the104
PDBbind recovery audit, not simply disabling the external homology filter.
