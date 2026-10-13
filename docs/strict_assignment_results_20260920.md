# Strict assignment diagnostics: completed collection and label-axis repair

Follow-up authorized 2026-09-26: see `strict_site_quality_probe_20260926.md`.
Frozen-base candidate-quality heads (stats/protein/ligand controls) were submitted
as smoke 12868601 and afterok-dependent full run 12868608. Rechecked 2026-09-26:
the smoke FAILED on the pre-cross independence assertion, not OOM; the dependent
full job was CANCELLED without running. No trained-head result is available.
These follow-ups do not change the completed results below or add test evidence.
Parallel follow-up was subsequently approved and queued; see
`strict_site_parallel_execution_20260926.md`. No new trained-head result yet.

## Job ledger (updated 2026-09-20)

| Job | Work | Result |
| --- | --- | --- |
| 12804060 | Merged TRAIN negative coverage | Scanned 10317 rows; saved full tables; exit 1 because four rows failed label-axis QC. 4m44s, about 5.1 GiB RSS. |
| 12804061 | Strict VAL candidate proposals | Exported 142/647 base44 predictions, then failed on 3au6 label length. 2m41s, no OOM. No complete candidate comparison and no ESM3 arm yet. |
| 12804249 | CPU label-axis preflight | Completed 651 rows in 3m22s; six val and four train length mismatches. |
| 12804286 | Corrected TRAIN audit retry | COMPLETED, exit 0, 4m58s, MaxRSS 5311188K. All 10317 rows; zero QC failures. |
| 12804287 | Corrected VAL export/proposals retry | COMPLETED, exit 0, 10m17s. Both models exported all 647 VAL rows and all 19 proposal protocols. |
| 12814298 | CPU hybrid-candidate follow-up | COMPLETED, exit 0, 1m43s, MaxRSS 64616K. Both model arms, all 647 rows, six pool/scorer combinations each; no training or GPU inference. |

The original failures and their partial outputs are preserved. Corrected retries
used `code_label_fix`; the hybrid follow-up uses a separate `code_hybrid` snapshot.
Completion is verified by sacct exit status, completion markers, and full result
populations, not just disappearance from squeue.

## TRAIN audit: final numbers, all 10317 rows QC-passed

| Eligibility level | Samples / 10317 | Coverage |
| --- | ---: | ---: |
| At least one exact residue-axis peer | 5754 | 55.77% |
| Distinct mask after overlap and two-frame center checks | 1739 | 16.86% |
| Also equivalent geometry (proper rigid fit max CA error <=0.05 A) | 1152 | 11.17% |
| Also known, different chemical identity | 179 | 1.74% |
| Still usable after training-visible truncation / axis checks | 179 | 1.74% |

The conservative chemistry-distinct set contains 109 unordered pairs, all
BioLiP-to-BioLiP, covering 179/5157 BioLiP rows (3.47%), 64 unique canonical
chemical identities and 66 of 6259 exact-axis groups. These final conservative
counts are unchanged by the four label-overlay repairs.
PDBbind contributes zero pairs at the strict **same-geometry** tolerance. This
does NOT mean PDBbind contains no useful embedding-level or homolog negatives:
distinct complexes can have different conformations and require another setup.

Among 848 geometry-verified distinct-pocket pairs:

| Chemical relation | Pairs |
| --- | ---: |
| Identical canonical isomeric SMILES | 558 |
| Unknown / failed strict chemistry sanitization | 181 |
| Known different chemistry | 109 |

All same-axis pair Jaccard median is 0.75. Negative counts including zero rows:
conservative p50/p75/p90 are all zero, mean 0.02113 per TRAIN row. Under uniform
sampling without replacement, batch 16 gives expected valid-negative availability
for a random anchor of only 0.00307%. This is an anchor probability, not batch
accuracy or dataset coverage. Explicit paired sampling/loading would be required
to use this sparse signal reliably; even then it is not a broad main objective.

Less restrictive exploratory geometry (RMSD<=2 A, max CA error<=5 A) plus
different chemistry covers 587/10317 rows (5.69%), including 88 PDBbind and
499 BioLiP. These are **not** automatically safe direct-region negatives.
Different SMILES also does not certify that a ligand cannot bind another site.

Chemistry is known for 9891 rows, including all 5160 PDBbind rows. All 426
unknowns are BioLiP: 226 RDKit valence failures and 200 kekulization failures,
not absent SDF files. Do not force unsanitized molecules into a known-negative
bucket. Source normalization/template repair would need a separate audit.

The two rows flagged `training_contacts_match=False` are exactly 2xct and 4glx.
The audit does not pretend their corrected supervision was used by the existing
checkpoint. None of the 179 conservative eligible rows depends on those repairs.

## Completed strict VAL results

These are **sample-mean binary contact F1 on strict VAL**, not held-out test,
not the old non-strict-split 0.599+ scores, and not three-class macro F1.
Each arm has 647 unique exact structure keys, so exact-structure-protein mean
equals sample mean here. That does not imply independence at sequence-cluster
level. Both models are frozen and use their previously saved contact gates.

| Model | Model-observed-prefix F1 | Full-axis F1 | Gate | Best checkpoint epoch |
| --- | ---: | ---: | ---: | ---: |
| Strict merged V11m13 base44 | 0.522127 | 0.521868 | 0.391170 | 28 |
| Strict merged ESM3 global prior44 | 0.524701 | 0.524366 | 0.390824 | 20 |

ESM3 minus base full-axis F1 is +0.002499. Paired exact-structure bootstrap,
10000 draws, RNG seed 20260920: 95% interval [-0.008404, +0.013369]. ESM3 has
292 sample wins, 285 losses and 70 ties. This is not evidence of a reliable
gain; the interval is across these VAL examples, not across training seeds.
Also, the training schedules differ, so this is not a clean ESM3-only ablation.

All six VAL overlays change zero protein contact labels; `legacy_model_f1`
equals corrected `model_f1` exactly for both models. There are 31 rows with
unobserved tails and 47 total unobserved true contacts. The mean F1 effects of
full-axis scoring are only -0.000260 / -0.000334, respectively. That does not
justify extrapolating this small truncation impact to HOLO4K.

### Actual selection versus oracle

All selected candidates use top-20%-mean probability, without GT. Oracle selects
the best candidate using GT only to diagnose the candidate family, never for
deployment. The original prediction is not a candidate in the initial 19 runs.

| Candidate family | Base selected F1 | Base oracle F1 | ESM3 selected F1 | ESM3 oracle F1 |
| --- | ---: | ---: | ---: | ---: |
| Original unmodified prediction | 0.521868 | n/a | 0.524366 | n/a |
| Contact components, 8 A, K<=16 | 0.484510 | 0.507882 | 0.488455 | 0.506034 |
| Probability NMS spheres, 8 A, K=32 | 0.277911 | 0.429888 | 0.267695 | 0.425663 |
| Probability NMS spheres, 12 A, K=32 | 0.336781 | 0.474850 | 0.323332 | 0.473315 |
| Geometry FPS spheres, 12 A, K=32 | 0.264054 | 0.374370 | 0.254940 | 0.374370 |

Every single fixed protocol has mean oracle F1 below the original prediction.
Therefore training only a better selector over one of these fixed families is
not yet justified. Component selection loses 0.03736 F1 on base44; paired 95%
bootstrap interval [-0.04555, -0.02951]. It improves 113 rows, worsens 221 and
ties 313. For ESM3 the drop is 0.03591, interval [-0.04407, -0.02841].

Broad coverage is not correct boundary localization. For base44 probability
NMS with 12 A / K=32, the candidate union covers 95.74% of true contact residues
but also 63.91% of the protein. Only 5.10% of samples have any one candidate
with IoU>=0.5. The original component family has lower target-union coverage
(53.92%) but much better IoU>=0.5 candidate hit rate (34.00%). These are distinct
notions of coverage; none should be described as a 95.74% pocket detection rate.

The FPS candidate oracle is identical between models, as expected: its geometry
and labels are identical and its construction is probability-independent. Its
selected F1 can differ because candidate ranking still uses model probabilities.

### Complementarity across scales, not an achieved gain

Taking the per-sample maximum oracle across all 19 protocols gives:

| Diagnostic candidate pool | Base oracle F1 | ESM3 oracle F1 |
| --- | ---: | ---: |
| Union of all 19 families | 0.619741 | 0.624372 |
| Same union, also allow the original prediction | 0.633283 | 0.638677 |

The union oracle exceeds the original F1 on 55.80% / 54.10% of samples.
This is evidence of shape/scale complementarity and remaining selection
headroom. It is NOT a trained reranker result, proof that such a selector can
be learned, or a generalization claim. The large candidate pool and GT-aware
per-sample choice make this upper bound optimistic.

## Follow-up and decision

Completed CPU job 12814298 tested three deduplicated pools: previous-protocol union,
that union intersected with the original predicted boundary, and both combined.
Every pool retains the complete original prediction (including empty predictions).
Each is ranked by the old top-20%-mean control or the fixed probability/count
proxy `2 * sum(p in candidate) / (candidate_size + sum(p over protein))`.
This is a ratio of expected counts, not the expectation of F1; it relies on
probabilities whose calibration is not assumed correct. Ties prefer the original.
Labels enter evaluation only; no weights, radii, gates or thresholds are fitted.
This remains exploratory VAL analysis after seeing the first results.

| Pool / selector | Base selected F1 | ESM3 selected F1 | Base oracle F1 | ESM3 oracle F1 |
| --- | ---: | ---: | ---: | ---: |
| Original prediction | 0.521868 | 0.524366 | n/a | n/a |
| Raw union + top20 mean | 0.325254 | 0.305994 | 0.633283 | 0.638677 |
| Raw union + soft-count F1 proxy | 0.521736 | 0.524329 | 0.633283 | 0.638677 |
| Boundary-preserving + top20 mean | 0.216092 | 0.208184 | 0.566437 | 0.561850 |
| Boundary-preserving + soft-count F1 proxy | 0.521832 | 0.524472 | 0.566437 | 0.561850 |
| Hybrid union + top20 mean | 0.216946 | 0.209160 | 0.644453 | 0.649790 |
| Hybrid union + soft-count F1 proxy | 0.521832 | 0.524472 | 0.644453 | 0.649790 |

All pools above include the original prediction. The hybrid pool averages about
170 deduplicated masks per example, so its oracle remains a generous bound.
Actual top20 scoring prefers tiny high-confidence fragments: hybrid-selected
size averages 4.42 / 4.54 residues, versus original 21.12 / 21.13 and GT 18.65.
Thus its very poor F1 is a selection/size-bias failure, not evidence that the
underlying model suddenly lost pocket recognition.

The count-aware proxy avoids collapse but mostly reproduces the original:
base44 retains it for 634/647 examples (97.99%), ESM3 for 641/647 (99.07%).
Base44 changes improve eight rows and worsen five; ESM3 improves five and
worsens one. Paired exact-structure bootstrap (5000 draws, seed 20260920):

| Hybrid proxy minus original | Mean F1 delta | 95% interval |
| --- | ---: | --- |
| Base44 | -0.000036 | [-0.000310, +0.000229] |
| ESM3 global prior44 | +0.000105 | [-0.000097, +0.000309] |

There is no useful demonstrated gain. Retaining the original was only an option,
not a guarantee: the heuristic can still choose a genuinely worse mask. These
intervals are descriptive VAL checks, not a multiplicity-corrected test claim.

Exploratory contact-size stratification is informative but NOT an input to
selection. The 45 examples with 1--10 true contacts have base F1 0.272592 and
ESM3 F1 0.249218, compared with 0.529292 / 0.533068 for the 519 examples with
11--25 contacts. The 83 examples with >=26 contacts reach 0.610593 / 0.619130.
Small-target examples have more union-oracle headroom (0.538463 / 0.532405),
but their small sample count and F1's size sensitivity prevent a causal claim.
Long proteins (>500 residues, n=299) score 0.504840 / 0.504319, versus
0.536498 / 0.541591 for <=500 residues (n=348).

The current evidence supports these decisions:

1. Do not replace the original residue predictions with single sphere/component
   masks: both actual and within-family oracle results are worse.
2. Do not make same-geometry, chemistry-distinct hard negatives the main training
   loss: coverage is 1.74% and random-batch availability is almost zero. They may
   support a carefully sampled auxiliary diagnostic. Embedding-level cross-complex
   matching requires its own false-negative masks and is not covered by this bound.
3. Stop blind probability-only proposal/scoring sweeps here: the follow-up did
   not improve strict VAL. Existing scores cannot extract the demonstrated union
   headroom; this does not prove that a learned selector would succeed either.
4. If proceeding to a trained selector, use TRAIN-only candidate-quality or
   ranking supervision with an unchanged original-output control. Candidate
   representations should provide information beyond final contact probabilities;
   use protein-side pre-cross states and ligand graph/local54 evidence. Measure
   held-out candidate ranking, boundary F1 and fallback decisions separately.
   Equivalent/ambiguous sites must not be assigned arbitrary nonbinding labels.

### Do not repeat the old assignment experiment unchanged

Code review during this collection found two existing paths:

- V11m15 `_assignment_embeddings` in `src/models/pocket_gvp_fusion_v11m15.py`
  pools `h_fused` and `h_ligand_fused`, which V11 returns **after cross-attention**.
  Its loss is auxiliary, not an explicit inference selector. Reusing it as an
  independent ligand-pocket retrieval head unchanged risks matching shortcuts;
  this is a design concern, not proof that a measured old run used a shortcut.
- V11m16 `_candidate_rank_forward` already uses `h_prot_global` and
  `h_ligand_context` before cross-attention, with protein-only pocketness NMS,
  Gaussian membership and a bounded centered class0 logit gate. The historical
  ledger `architecture_roadmap_20260811.md` reports essentially no improvement
  for the old non-strict PDBbind probe and gate alpha 0.00782. High training rank
  accuracy alone was not proof of held-out assignment quality.

Therefore a future proposal cannot claim novelty simply by adding a candidate
query or a tiny scalar gate. A meaningful next probe would first audit/fix
candidate quality and train a site-quality/ligand-compatibility selector on the
current strict TRAIN, then verify explicit output selection or bounded
candidate-specific boundary refinement. It must differ from the old V11m16
gate that could shrink to near zero, and retain a no-change inference control.
No such model training is launched by this collection.

No HOLO4K/COACH420/test predictions are used by this follow-up. A frozen successful
protocol would need separate held-out validation of multi-site behavior.

Artifacts under `tmp/strict_assignment_20260919/`:
`negative_12804286/{summary.json,samples.csv,distinct_pairs.csv}` and
`val_12804287/{base44,esm3gp44}/{export_summary.json,per_sample.csv,candidates/}`.
The follow-up is `hybrid_12814298/{base44,esm3gp44}/`, including summary JSON/CSV,
per-sample rows and SHA256 manifests of every prediction consumed.
The remote experiment root additionally retains all NPZ predictions. Local and
remote follow-up source SHA256 match:
`bf08ea6872f65799dd59ebbfbf1bdc46d8e868ef91a01ed3a49f0e3337c69792`.

## Why the strict check failed

| Sample | Split | Cached labels | Protein / ESM rows |
| --- | --- | ---: | ---: |
| 2xcs | train | 1403 | 1365 |
| 2xct | train | 2782 | 2702 |
| 4glx | train | 638 | 586 |
| 4lq3 | train | 528 | 526 |
| 3au6 | val, first failure | 592 | 575 |
| 3oyl / 3oyn / 3s3m / 3s3n / 3s3o | val | 826 each | 790 each |

The original `New/` PDBs contain DNA/RNA ATOM residues. The historical label
enumeration retained all non-HETATM heavy-atom residues, while the protein
feature parser retained amino acids with CA. Local reconstruction of the two
enumerations reproduces the row counts above. Some nucleic-acid chains follow
the protein (extra tail only); others precede/interleave it, so prefix clipping
can shift actual residue labels. Length mismatch must not be silently ignored.

This is a label-enumeration issue, not evidence that the strict sequence split
leaks. Its small observed sample count also does not explain the whole HOLO4K
performance gap.

Source verification completed for all ten anomalies. Maximum reconstructed
distance error is 6.126e-5 A (tolerance 1e-4 A); cached protein CA positions
match source positions exactly. All six val cases have **only extra nucleic
acid tail rows**: zero shifts or class/contact changes within the protein axis.
The original exported protein-prefix labels were already correct for these six.
The strict length check was deliberately stronger than that assumption.

Train 2xcs and 4lq3 also have extra tails only. For 2xct, 1334 protein positions
shift, changing nine three-class labels and four contact labels on the full axis;
for 4glx, all 586 positions shift, changing 81 classes and 26 contact labels.
Training-visible impact is separately checked after the 1800-residue cap.
All four 2xct contact changes remain within its 1800-residue training prefix;
all 26 changes for 4glx are visible. The other two train rows have zero changes.
The verified overlay SHA256 is
`13eb33764c2fcde766805723df4eaffd0f5b139cc6444db9dbb73adfe0d5635b`.

## Repair scope and next measurement

`prepare_internal_label_alignment.py` collects mismatch metadata, then verifies
the **entire historical distance vector** against the original PDB/SDF. It maps
canonical chain/residue/insertion keys only after residue names and CA positions
agree with the cached protein. The overlay is bound to split hash, raw-distance
digest, reference-axis/coordinate digest and source file hashes.

No original labels, features, weights, split files or training code are changed.
The VAL exporter preserves model inputs and the checkpoint gate, while reporting
both legacy-label and corrected-label F1. The negative audit flags corrected
rows whose current training labels differ and does not count their corrected
contacts as already-available training supervision.

43 related tests pass, including rejection of stale, duplicated or unverified
overlay mappings, GT-independent hybrid selection, original-prediction retention,
empty predictions, bounded finite probabilities and shape-preserving masks.
Slurm shell syntax passes. The source data, weights and model code remain unchanged.
