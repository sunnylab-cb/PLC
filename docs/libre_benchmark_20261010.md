# LiBRe matched COACH420 / HOLO4K evaluation

## Status and provenance

**Final report accepted with explicit input-QC exclusions, 2026-10-10.** The user
approved excluding the ten recorded chemical-input failures from both models.
The final matched populations are **491 COACH420 pairs / 299 proteins** and
**7238 HOLO4K pairs / 3424 proteins**. The exact exclusion policy is saved in
[`libre_benchmark_20261010_exclusions.json`](libre_benchmark_20261010_exclusions.json).

Both requested datasets were traversed in full on `i64m1tga800ue`, one A800 per
task. All requested rows are accounted for, but strict all-row inference
acceptance is still not passed. Report acceptance does not change raw coverage,
Slurm exit codes or original results. Do not describe this as 100% prediction
coverage or as an exact reproduction of paper scores.

| Dataset | Task | Elapsed | Successful / requested pairs | Successful proteins | Rejected |
|---|---|---:|---:|---:|---:|
| COACH420 | 12960610_0 | 1m54s | 491 / 494 (99.39%) | 299 / 299 | 3 |
| HOLO4K | 12960610_1 | 25m14s | 7238 / 7245 (99.90%) | 3424 / 3430 | 7 |

Both Slurm tasks report `FAILED 1:0` because the runner deliberately exits
nonzero after saving results if any requested row lacks a prediction. This is
not OOM or an interrupted dataset traversal. Peak allocated CUDA memory was
2.65 GiB / 2.82 GiB. Progress files confirm 299/299 and 3430/3430 proteins visited.
The earlier array 12960178 failed before inference due to a launcher typo;
that separate failure was fixed and its artifacts were preserved.

- Official LiBRe revision: `e9151a85bab2d05b67b953bc55a74c780aa2bf24`.
- Checkpoint SHA256: `ba40bdf78ba4e596e7429e56d2c955c42b6debc67fb159a7438dfe6b9993c871`.
- Remote results: `/hpc2hdd/home/shuang886/PLC/external_baselines/libre/results/benchmark_12960610/`.
- Local small artifacts: `tmp/libre_benchmark_12960610/` (summaries, CSVs, QC,
  rejection records, frozen reference CSVs, paired analysis, transfer hashes).
- Comparator: frozen **GFGF best-F1, val-selected gate**, benchmark job 12956268.
  This is a named reference run, not a claim to compare every historical PLC model.
- All 15 transferred input/result files passed SHA256 checks. Remote prediction
  filenames match successful sample keys exactly. Successful and rejected keys
  form a disjoint, complete partition of the requested reference populations.

## Fixed evaluation protocol

LiBRe uses fresh raw ESM2 L33 FP32 representations and its released 40D
SMILES-to-graph features, without PLC feature normalization. Threshold is the
official fixed `p > 0.6`; no benchmark threshold search was performed. Every
protein chain is predicted for every ligand, then reassembled onto the PLC
residue axis. There is no oracle binding-chain selection and no ligand atom cap.
Native protein-ligand geometry is used only for label QC and metric calculation.

Targets are exact per-ligand contacts at heavy-atom distance below 5 A. The
metric axis matches PLC's first `min(full_length, 1800)` residues. HOLO has 94
successful pair rows affected by that existing protein cap, excluding 79 positive
residue labels in total; COACH has none. No evaluated chain exceeded 1500, so the
implemented overlapping-window path was not exercised by these real samples.
Single-chain raw ESM and fixed 1500-position padding avoid the upstream batch
slicing's retained-EOS ambiguity. This remains an external-protocol adaptation.

All successful rows passed independent heavy-atom tag geometry verification,
residue-key alignment, and reference residue/contact-count/centroid checks.
The binary target here is contact-only, **not** PLC's contact-plus-near 8 A metric.

## Matched results

Rows below use the identical successful pair set for both models.

| Dataset | Model | Sample F1 | Protein mean of ligand mean | Protein mean of ligand max | Mean precision | Mean recall | Mean AP |
|---|---|---:|---:|---:|---:|---:|---:|
| COACH | LiBRe | 0.4494 | 0.5494 | 0.6043 | 0.3573 | 0.6814 | 0.5357 |
| COACH | PLC GFGF | 0.4439 | 0.5358 | 0.6077 | 0.4312 | 0.4923 | 0.4966 |
| HOLO | LiBRe | 0.3863 | 0.4912 | 0.5231 | 0.2721 | 0.8456 | 0.4403 |
| HOLO | PLC GFGF | 0.3300 | 0.4470 | 0.6005 | 0.3182 | 0.3596 | 0.3696 |

LiBRe pooled-residue F1 is **0.5168 on COACH** and **0.3312 on HOLO**.
These differ from sample means and must not be interchanged. Its pooled AUROC
is 0.9453 / 0.9438; pooled AP is 0.5903 / 0.3345.

| Dataset | Model | Conditional BP F1 | BP eligible rows | Centroid PDR | Centroid LAA | Mean predicted contacts | Predicted p50 / p90 |
|---|---|---:|---:|---:|---:|---:|---:|
| COACH | LiBRe | 0.6804 | 219 | 0.6029 | 0.2070 | 29.60 | 29 / 46 |
| COACH | PLC GFGF | 0.7210 | 150 | 0.4969 | 0.1754 | 17.86 | 17 / 28 |
| HOLO | LiBRe | 0.6836 | 1804 | 0.2865 | 0.0973 | 87.94 | 66 / 176 |
| HOLO | PLC GFGF | 0.7129 | 1714 | 0.4439 | 0.1461 | 23.81 | 23 / 35 |

True contact means are 14.47 / 20.16 on COACH / HOLO. LAA has 285 / 5619
eligible pair rows, respectively. BP selects different successful-localization
subsets for each model: its numerical difference is not a paired, unconditional
boundary comparison. PDR/LAA use a single centroid of all predicted contacts;
broad or multimodal predictions can place that centroid between real pockets.
These are diagnostics, not standard object-level pocket detection metrics.

### Paired uncertainty

5,000 paired bootstrap resamples, grouped by protein, fixed seed 20261010:

| Dataset | LiBRe minus PLC sample F1 | 95% interval | Protein-balanced F1 difference | 95% interval |
|---|---:|---|---:|---|
| COACH | +0.00551 | [-0.01224, +0.02394] | +0.01355 | [-0.00564, +0.03301] |
| HOLO | +0.05634 | [+0.05218, +0.06040] | +0.04412 | [+0.03877, +0.04931] |

COACH does not establish a clear F1 advantage under this bootstrap. The HOLO
sample-F1 difference is positive under protein-resampled dataset uncertainty.
These intervals do not measure training-seed variance or resolve training-set
overlap for the released LiBRe checkpoint.

### Ligand multiplicity

Buckets count reference ligand rows per protein, not unique chemicals or distinct
pocket clusters. F1 values are sample means on the matched successful pairs.

| Dataset | Ligand rows per protein | Pair rows | LiBRe F1 | PLC F1 |
|---|---|---:|---:|---:|
| COACH | 1 | 200 | 0.6382 | 0.6184 |
| COACH | 2-3 | 159 | 0.4147 | 0.4082 |
| COACH | 4+ | 132 | 0.2050 | 0.2225 |
| HOLO | 1 | 1613 | 0.6274 | 0.6165 |
| HOLO | 2-3 | 2638 | 0.4208 | 0.3473 |
| HOLO | 4+ | 2987 | 0.2257 | 0.1600 |

The single-ligand subsets are much closer across datasets than their overall
means. This supports ligand multiplicity / population composition as an important
part of the COACH-HOLO difference; it does not establish a causal explanation or
prove every multi-ligand row represents a distinct biological binding site.

## Rejected inputs

| Dataset | Sample keys | Audit result |
|---|---|---|
| COACH | `1bs1A_DAA_A850` | Observed SDF has one fewer carbon than CCD DAA; sanitizable CCD SMILES is not an equivalent drop-in replacement. |
| COACH | `1xnyA_191_A5601`, `1xnyA_191_B9601` | SDF oxygen valence fails; CCD 191 lookup returns 404. Candidate 1VU matches A5601's heavy composition but not connectivity; B9601 has one fewer carbon. |
| HOLO | `1chw_HXC_B391`, `1d4o_NAP_A201` | CCD SMILES can sanitize and heavy composition matches, but connectivity differs from the observed SDF. |
| HOLO | `1d0y_ONP_A999`, `1d0z_PNQ_A999`, `1d1a_DAE_A999`, `1d1b_DAQ_A999`, `1d1c_NMQ_A999` | Current RDKit rejects the listed CCD SMILES' Be valence as well. |

An independent audit downloaded only small CCD JSON records directly on HPC2.
All ten rejected rows were inspected; no same-heavy-connectivity, sanitizable
replacement was found. No source files were edited, atoms discarded, charge
guessed, or sanitization bypassed to manufacture complete coverage. For example,
[RCSB's ONP record](https://www.rcsb.org/ligand/ONP) lists the beryllium-containing
component and its SMILES; the local audit records the installed RDKit rejection.
This does not establish that these experimental ligands are chemically invalid.

Current [RCSB 1XNY ligand validation](https://www.rcsb.org/ligand-validation/1XNY/1VU)
lists propionyl-CoA component 1VU at A5601/B5601. This supplies an identity
candidate for the historical benchmark's 191 rows, not proof of a universal
191-to-1VU alias or an atom mapping for B9601. Even with this candidate, the
composition/connectivity audit found no validated drop-in replacement for the
ten failures. Evidence is in `chemistry_qc_composition.json`; input files remain
unchanged.

If every missing row had F1 anywhere in [0,1], the full-population sample mean
would lie in **[0.44666, 0.45274] for COACH** and **[0.38594, 0.38691] for HOLO**.
These are algebraic bounds, not measured missing-row predictions. The frozen PLC
full-population values are 0.44353 / 0.33016. Exclusions therefore cannot explain
the observed HOLO sample-F1 advantage, but full prediction coverage is still unmet.

## Historical ligand preparation audit

The production extractor snapshot is archived at
`tmp/libre_benchmark_12960610/historical_qc/extractor_snapshot.py`.
Its `_sanitize` helper first tries full sanitization, then accepts a fallback
that excludes `SANITIZE_PROPERTIES` (lines 366-379). Its last extraction fallback
uses `DetermineConnectivity` without inferred bond orders and labels the output
`connectivity_only` (lines 423-433). The old SDF QC checked readability, heavy-atom
count and conformer presence with `sanitize=False`; `ok=True` was not a certificate
of chemically correct bonds or valence.

Historical QC job 9870184 has exactly the same sample key sets as the frozen
reference: 494 COACH and 7245 HOLO rows. The following are historical method
counts, not a new full-cache chemistry validation. All ten currently rejected
SDFs independently retain the `connectivity_only` property.

| Dataset | Extraction method | Requested | Successful | LiBRe F1 | PLC F1 | Difference |
|---|---|---:|---:|---:|---:|---:|
| COACH | ccd_template | 159 | 159 | 0.5414 | 0.5126 | +0.0289 |
| COACH | determine_bond_orders | 68 | 68 | 0.2523 | 0.2973 | -0.0450 |
| COACH | connectivity_only | 267 | 264 | 0.4447 | 0.4403 | +0.0044 |
| HOLO | ccd_template | 6116 | 6116 | 0.4022 | 0.3390 | +0.0632 |
| HOLO | determine_bond_orders | 289 | 289 | 0.1847 | 0.2167 | -0.0319 |
| HOLO | connectivity_only | 840 | 833 | 0.3397 | 0.3028 | +0.0369 |

F1 is a sample mean on each method's identical successful pair subset. Historical
`connectivity_only` fractions are 54.05% for COACH and 11.59% for HOLO. This flags
unverified chemistry beyond the ten outright rejections, but does not mean every
fallback molecule is wrong: some legitimately have only single bonds. Nor does
`ccd_template` alone prove complete correctness of charge/stereochemistry.

These groups differ in ligand family, size and protein/site composition, so their
scores are not a causal ablation. COACH has more fallback inputs, and LiBRe's HOLO
advantage persists within the CCD-template subset; chemistry fallback alone
cannot be asserted to explain the entire COACH-HOLO gap. A future chemistry-clean
branch should validate authoritative atom-name/CCD mappings for both models,
keep observed coordinates/labels separate from complete molecular identity, and
not overwrite this frozen benchmark.

## Final interpretation

1. LiBRe's stronger HOLO sample F1 accompanies much higher recall, substantially
   broader predictions, and higher AP. It is not evidence that simply removing
   PLC features improves performance or that LiBRe solves multi-pocket assignment.
2. HOLO LAA and centroid PDR are lower for LiBRe despite its higher F1. Candidate-
   pocket-level analysis is needed before treating the centroid diagnostics as
   a true assignment comparison. Do not tune a new gate on these benchmark labels.
3. Use this as a named, matched-population external baseline, not a SOTA claim.
   Author-checkpoint training overlap and source ligand chemistry for otherwise
   sanitizable SDFs remain unaudited. No blanket chemistry equivalence is claimed.
4. The user approved explicit exclusion of the ten input failures, applied
   symmetrically to both models. No performance-based filtering, chemistry repair,
   benchmark threshold search or additional GPU run is part of this finalization.
   A future CCD-complete-molecule branch would be a separate protocol.

### Completion audit

Final read-only revalidation on 2026-10-10 confirmed terminal Slurm states for
both 12960610 tasks, no live LiBRe benchmark task, and unchanged remote coverage
(491/494 and 7238/7245). All 15 locally transferred artifacts still match their
recorded SHA256 digests. The ten-row composition audit has no audit errors and
no sanitizable same-heavy-connectivity candidates. There is no unfinished GPU
process to wait for or transient inference failure to retry.

Execution, output collection, matched-subset analysis and failure diagnosis are
finished. The user has now resolved the input-policy decision by accepting the
ten explicit exclusions. The finalized machine-readable report is
`results/libre_benchmark_12960610/final_report.json`. Its acceptance check requires
exact equality between approved exclusions and recorded failures, checks their
chemistry-failure reason, and verifies that the matched sample set equals the
reference minus those exclusions. It separately retains
`all_original_rows_have_predictions=false` and original artifact hashes.
Strict full-population prediction coverage remains unmet, but is no longer a
requirement for this user-approved matched-subset report. Original inputs and
inference summaries are unchanged; no resubmission is needed.

The report, policy, analysis script snapshot and final JSON are archived remotely
under `external_baselines/libre/results/benchmark_12960610/accepted_report_v1/`
within `/hpc2hdd/home/shuang886/PLC/`, separate from the original inference files.

Reproduce the local accounting/bootstrap report with:

```bash
python scripts/benchmark/analyze_libre_benchmark.py \
  --root tmp/libre_benchmark_12960610 \
  --historical-qc-root tmp/libre_benchmark_12960610/historical_qc \
  --accepted-exclusions docs/libre_benchmark_20261010_exclusions.json \
  --output results/libre_benchmark_12960610/final_report.json
```
