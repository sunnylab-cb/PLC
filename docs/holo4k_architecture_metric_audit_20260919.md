# HOLO4K architecture and metric audit (2026-09-19)

## Scope and main finding

This is a read-only reanalysis of job `12722325`, seeds 43/44/45,
strict-split PDBbind+BioLiP V11m13 + ESM3 global prior, plus the seed44
non-ESM3 baseline from job `12567314`. No model, labels, thresholds, or
training runs were changed. All main tables below use the original
validation-selected gate, not a threshold selected on external benchmarks.

**The multi-ligand deterioration is real, but the claim that pocket detection
is already strong and only an assignment head is failing is not established.**
The evidence supports four interacting problems:

1. Conditional metrics overstate the apparent gap between detection and assignment.
2. One residue map and one centroid do not represent multiple pocket instances.
3. Repeated chemical ligands introduce instance ambiguity without copy-specific inputs.
4. Confirmed residue alignment and non-finite feature defects require repair before
   attributing all residual error to architecture.

## Data and reproducibility

- Remote result root: `/hpc2hdd/home/shuang886/PLC/results/v11m13_chain_si30_esm3_benchmark_12722325`.
- Baseline: `/hpc2hdd/home/shuang886/PLC/results/v11m13_chain_si30_eval_12567314/pdbbind_biolip/benchmark_per_ligand_val_gate`.
- Local synchronized CSV/JSON: `tmp/holo_reaudit_20260918/esm3/` and `base44/`.
- Actual archived training config: `tmp/holo_reaudit_20260918/seed44_config.yaml`.
- Machine-readable audit: `tmp/holo_reaudit_20260918/audit.json`.
- Reproduce with `python scripts/diagnostics/audit_per_ligand_csv.py tmp/holo_reaudit_20260918/esm3`.
- The directory date reflects when the audit started; it continued past midnight.
- Standard deviations below are sample SD across three seeds, not confidence intervals.

The evaluated sets contain **494 rows / 299 proteins** for COACH420 and
**7,245 rows / 3,430 proteins** for HOLO4K. These are not the full nominal
420 / 4,009 protein lists. A publication needs an inclusion/exclusion ledger;
do not label these results as complete canonical benchmark coverage.

## Actual input, architecture, and output

The completed seed44 config, not the class defaults, specifies:

| Component | Actual setting |
|---|---|
| Protein sequence | ESM2 final, 1,280 dimensions per residue |
| Protein chemistry | 62-dimensional physicochemical features |
| Protein geometry | 89 scalar features, 8 spatial vectors plus 4 derived atom/backbone vectors; CA coordinates and chain metadata |
| Protein backbone | 3 GVP layers, geometric/ESM gated fusion, 4 self-attention layers; width 512 |
| ESM3 | 1,536D, projected and masked-mean pooled over protein residues; bounded broadcast global prior |
| Ligand | 109D/atom: local54 + global45 + shape10; 6D bond attributes + 8D bond-distance RBF |
| Ligand encoder | 3 ligand GVP layers + 2 internal self-attention layers, global pooled token |
| Protein-ligand fusion | 4 V12-style co-attention blocks with residual gates; original protein self-attention retained |
| Late evidence | local54 + graph hidden K/V; base33-anchor group interactions and residue-chemistry readout |
| Active extra modules | Light ligand FiLM, 4 pocket virtual tokens, `center_aux_head`, source dropout 0.03 |
| Inactive modules | UniMol input, global delta gate, local54 chemistry adapter/reweight, extra edge-context K/V |
| Local protein gradient | 0.6 on the local-delta path, not a 0.6 multiplier on every trunk gradient |
| Output | `[B, L, 3]` residue logits; base logits + bounded local delta |
| Target | Contact <5 A, near 5-8 A, far >=8 A, using nearest ligand/protein heavy-atom distance |
| Evaluation | Val-selected contact-probability gate, no ligand atom cap, protein cap 1,800 |

```text
protein geometry/chemistry -> GVP --+
ESM2 -> projection ---------------+-> gated fusion -> protein self-attention
ESM3 -> projection -> global mean -> bounded broadcast prior ---------^

ligand local54/global55 + bonds/internal geometry -> GVP -> ligand self-attention

protein state <-> ligand state: 4 co-attention blocks
                 |
                 +-> base residue classifier
                 +-> FiLM + local54/graph read + group chemistry compatibility
                     + virtual-token readback + extra contact scalar
                     -> local residue delta

final logits = base + scaled delta -> contact / near / far for every residue
```

Important qualifications:

- `base_logits` are already downstream of ligand co-attention, not a protein-only
  predictor. A base-only ablation is not a ligand-independent control.
- In `src/models/pocket_gvp_fusion_v11m10.py`, the late attention normalizes over
  **ligand atoms for each residue**. It does not impose competition over distinct
  protein pockets. Co-attention still supplies global context, so this does not
  prove the network cannot learn specificity; it identifies a missing explicit
  output representation and training constraint.
- The existing pocket tokens pool protein context and write it back into one
  residue map (`v11m8.py:_apply_pocket_virtual_tokens`). They are **not** separately
  supervised candidate masks with objectness and ligand assignment scores.
- `center_aux_head` adds another scalar to each residue's class0 logit
  (`v11m10.py`, around line 460). There is no dedicated centroid/set-matching loss
  in the active V11m13 training path. Its name must not be treated as proof that
  explicit pocket-center supervision was already tested.
- ESM3 global pooling broadcasts the same extra vector to every residue. This
  can modulate residue-dependent nonlinear processing, but does not itself select
  a ligand-specific pocket or identify a copy of a ligand.
- Cross-attention receives no joint protein-ligand coordinates. Ligand geometry
  describes internal distances/directions, not which bound copy is next to which
  protein chain. Adding native ligand-to-protein distances would leak the answer
  in blind site prediction and is not an acceptable fix.

## Metric correction: BP is conditional

Definitions were checked in `scripts/infer/infer_v11_benchmark.py`,
`compute_sample_metrics` and `enrich_rows_with_interpretability_metrics`.

| Metric | Actual definition | Consequence |
|---|---|---|
| Sample F1 | Mean class0 F1 over ligand-instance rows | Primary strict contact-map metric |
| Binary F1 | Merge contact and near classes, i.e. <8 A | Different target, not generic detection accuracy |
| Current-pocket hit | One predicted contact centroid within 4 A of current true contact centroid | Localization diagnostic |
| BP F1 | Class0 F1 **only on current-pocket-hit rows**; other rows set to NaN and excluded | Selected successful subset, not overall detection strength |
| PDR | Same single centroid within 4 A of any annotated contact centroid in the protein | Not recall over a predicted pocket set |
| LAA | On rows with >=2 valid true centers: current center is nearest and within 4 A | Joint localization-and-ranking metric, not conditional assignment accuracy |

The evaluator does not cluster equivalent ligand sites before LAA; two copies
or overlapping ligands count as two candidates. Tied nearest centers can both
have rank1. Thus neither its candidate count nor its chance level equals the
number of independent biological pockets.

| Metric, mean +/- SD | COACH420 | HOLO4K |
|---|---:|---:|
| Sample F1 | 0.4417 +/- 0.0133 | 0.3197 +/- 0.0040 |
| Binary F1 | 0.5521 +/- 0.0030 | 0.3821 +/- 0.0011 |
| BP F1, successful subset only | 0.7176 +/- 0.0157 | 0.7160 +/- 0.0037 |
| Current-pocket hit, nonempty-GT denominator | 0.3251 +/- 0.0131 | 0.2255 +/- 0.0107 |
| PDR | 0.4980 +/- 0.0166 | 0.4200 +/- 0.0299 |
| LAA | 0.1719 +/- 0.0122 | 0.1401 +/- 0.0115 |

HOLO4K BP uses only **1,675 / 1,543 / 1,680** successful rows for seeds43/44/45,
respectively, out of 7,245 total. Failed-current-localization rows still have
mean F1 **0.2044**, versus conditional BP **0.7160**. This is consistent with
localization and representation problems, not proof of a solved detection stage.

On the multi-center subset, PDR is about **0.4795 / 0.3938** for COACH/HOLO,
while `LAA / PDR` computed within the same subset is **0.3586 / 0.3559**.
This approximate factorization separates any-site localization from nearest-site
assignment conditional on that localization. The latter is similarly poor in
both datasets; the aggregate LAA difference is not evidence of an exclusively
HOLO-specific assignment-head failure.

Single-centroid failure example: predict two correct 20-residue sites centered
at x=0 and x=30 A. Their merged centroid is x=15, so current PDR can be zero even
though both sites were detected. Against one 20-residue ligand-instance label,
the merged map has precision 0.5, recall 1, F1 2/3. Center failure and residue
failure are not interchangeable.

## Composition and repeated-ligand evidence

All bins below count **evaluated ligand rows per protein**, not distinct sites.

| Ligand rows/protein | COACH rows | COACH F1 | HOLO rows | HOLO F1 |
|---|---:|---:|---:|---:|
| 1 | 200 | 0.6167 | 1,619 | 0.5803 |
| 2 | 130 | 0.4177 | 2,132 | 0.3591 |
| 3-4 | 82 | 0.3006 | 2,019 | 0.2194 |
| 5-9 | 70 | 0.1952 | 1,143 | 0.1252 |
| >=10 | 12 | 0.1848 | 332 | 0.0756 |

At 200-399 residues, F1 is nearly identical: **0.4847 COACH / 0.4851 HOLO**.
HOLO drops to 0.2835 at 400-799 residues and 0.1639 at 800-1799 residues.
Length, chain multiplicity, and ligand count are confounded; this is a descriptive
stratification, not a causal attribution to any one factor.

Reweighting HOLO ligand-count bins to COACH row proportions yields F1 **0.3854**,
instead of 0.3197. The change accounts descriptively for about 54% of the raw
0.1219 dataset gap, but does not remove within-bin length/chemistry differences.
Protein-balanced F1 is **0.5339 COACH / 0.4279 HOLO**. This does not replace the
strict row metric; it shows how repeated rows weight the aggregate.

Repeated ligand-code rows within the same protein:

- COACH: 190/494 rows (38.5%), F1 **0.2681**.
- HOLO: 4,715/7,245 rows (65.1%), F1 **0.2068**.
- HOLO multi-ligand rows whose ligand code occurs only once: 911 rows, F1 **0.4409**.
- Seed44 has 1,698 repeated-code groups with true-center spread >15 A. In 1,249
  of them (73.6%), predicted-center spread across ligand rows is <=1 A.
- Example `1tyx_MAN_A2` and `1tyx_MAN_A6`: identical predicted centers and 11
  predicted contacts, but per-instance F1 **0.0000 / 0.5882**; true centers
  differ by about 16.01 A. These two actual ligand tensors are not byte-identical
  (atom order/geometry/features differ), so this is evidence of weak switching,
  not a proof of identical-input impossibility for those particular rows.

The general identifiability issue remains: if two ligand instances yield the
same effective chemical/conformational input and the same protein input, a
deterministic predictor cannot output two different copy-specific target masks.
Changing an assignment head cannot create a missing copy identity. Biological
equivalence must be audited with molecular graph identity and chain/site
symmetry, not ligand residue code alone.

Keep strict instance scores unchanged. Add clearly named chemical-equivalence
union or symmetry-aware set metrics as complementary evaluations. If the task
requires a specified chain/copy, that condition must be part of the task input,
not recovered from the native ligand pose after seeing the answer.

## Confirmed pipeline defects and bounded impact

### Residue alignment

`BenchmarkV11Dataset.__getitem__` currently chooses
`min(protein_length, ESM_length, label_length, 1800)` and takes prefixes. It does
not align ESM2 to protein residue keys. ESM3 aligns to the ESM2 reference; its
equal-length fast path also skips key verification.

There are length-mismatched rows in **56/494 COACH** and **692/7,245 HOLO**.
Length mismatch alone is not proof of positional shift: a tail-only difference
can leave the retained prefix correct. Remote key-level spot checks confirmed:

| HOLO protein | Protein / ESM2 length | Positional mismatches in retained prefix |
|---|---:|---:|
| 12as | 656 / 654 | 0 (tail-only omitted keys) |
| 1a2c | 301 / 298 | 2 |
| 1a4w | 283 / 282 | 0 |
| 1a5s | 646 / 645 | 0 |
| 1ae8 | 299 / 298 | 1 |
| 1afe | 299 / 298 | 1 |
| 1ah3 | 315 / 314 | **314** (protein A:1 versus ESM A:2 at first row) |
| 1atp | 356 / 354 | **172**, starting at protein E:197 / ESM E:198 |

The safe fix is a canonical `(chain, residue number, insertion code)` axis with
sequence checks and explicit missing-feature masks. Preserve geometry/labels on
that axis; do not silently repair a middle omission by truncating the tail.
Equal length is not sufficient to prove matching order. A full key-level audit
of both benchmark and training cache is still outstanding.

### Non-finite ligand features

Each HOLO seed has the same **10 positive-label rows with non-finite AP** and
zero predicted contacts under the gate. The CSV alone cannot establish why AP
failed; remote restricted tensor inspection confirmed source NaNs in three:

- `1bs1_DAA_A850`: 4 non-finite atom-feature values, from 2 SASA and 2
  lipophilic-SASA values; geometry and bond tensors were finite.
- `1cwu_TDB_A502`: same pattern.
- `1cwu_TDB_B504`: same pattern.

These are real cache defects, not merely proof of FP16 overflow. The current V4
SASA extractor already has a finite-value fallback; old caches need regeneration
or explicit versioned repair and re-QC, not just a source-code fix. Do not silently
drop failed rows or interpret NaN argmax/gating as a model decision.

Even improving all ten affected row F1 values from 0 to 1 would add at most
**10/7245 = 0.00138** to mean F1. Thus this defect cannot explain the 0.122 gap.

Benchmark AMP uses default CUDA FP16, while training `--amp` plus `use_bf16`
uses BF16. Validation also lacks explicit BF16 selection in `train_v11m3.py`.
Unify inference precision and fail on non-finite logits, but do not assume that
doing so alone repairs non-finite source data.

### Truncation and normalization

- Actual benchmark `max_atoms=-1`; **zero rows were atom-cropped** and no evaluated
  ligand exceeds the training limit of 256 atoms. The old atom cap is not the
  explanation for this table.
- Only **94/7,245 HOLO rows** have protein length >1800 (22 proteins), versus
  zero COACH rows. Their mean F1 is 0.1323. They need chain-aware coverage auditing,
  but account for only 1.3% of rows, not the whole collapse.
- Normalization is not wholly missing: benchmark metadata records training
  ligand statistics applied to 56 dimensions and spatial clamping on four dims.
  This does not certify residue alignment, provenance, or numerical finiteness.
- Nine COACH and five HOLO rows have zero true contact residues; keep their
  inclusion and metric-denominator policy explicit and investigate why.

## What ESM3 does and does not show

Seed44, same evaluation rows, baseline versus ESM3:

| HOLO metric | Baseline | ESM3 | Delta |
|---|---:|---:|---:|
| Sample F1 | 0.319767 | 0.318005 | -0.001762 |
| LAA | 0.136275 | 0.127379 | -0.008896 |
| BP F1 (model-specific successful subset) | 0.704702 | 0.720054 | +0.015352 |

BP subsets change with the model: 1,159 rows are successful for both, 423 only
for baseline and 384 only for ESM3. On the common successful subset, F1 increases
from **0.7184 to 0.7352**. Thus there is some shared-subset boundary improvement,
but it does not compensate for other cases or yield overall improvement.

Do not make a pure ESM3 causal claim yet: baseline and ESM3 use different LR/
warmup settings, and baseline seeds43/45 external evaluations are missing from
this comparison. Three ESM3 seeds show stable weakness, not a proof of an
irreducible architecture ceiling or statistical equivalence.

Additional implementation risks to audit rather than assume solved:

- BioLiP ESM3 extraction is sequence-only (`esm3_extract_biolip_chains.py:75`);
  the original complex extractor uses `ESMProtein.from_protein_complex`.
  Their feature-generating conditions differ.
- Global pooling uses the protein-valid mask, not an auxiliary-ESM3 matched-row
  mask. Zero-filled missing embeddings can be passed through biased projections
  and counted in the denominator.
- ESM3 global averaging may mix repeated chains and different domains, while
  BioLiP materialization slices chain-level inputs. This is a plausible domain
  shift, not a quantified cause from the current CSVs.

## Literature cross-check

The quoted UniSite numbers are correct, but its evaluation uses the **mlig,
single-chain HOLO4K-sc** subset and site-set IoU AP, not our multichain,
ligand-instance-conditioned residue F1. UniSite predicts separate site masks
with set matching, which supports testing explicit pocket instances; it does
not prove our encoder is sufficient or our head is the sole cause.
Source: [UniSite, final arXiv revision](https://arxiv.org/html/2506.03237v2).

P2Rank explicitly describes HOLO4K as mainly multimers, whereas COACH420 is
single-chain. Its dataset repository distinguishes canonical 4,009 structures
from 4,543 files in the raw folder and describes MOAD-based mlig selection.
Sources: [P2Rank paper](https://link.springer.com/article/10.1186/s13321-018-0285-8),
[dataset specification](https://github.com/rdk/p2rank-datasets).

GrASP groups surface-atom segmentation into separate predicted sites and reports
HOLO4K(Mlig+) Top-N DCA recall of 81.3% (P2Rank 81.2%), on 4,514 systems and 6,368
ligands. That dataset/protocol differs again. Neither its recall nor UniSite AP
is a numerical baseline for our conditional BP or strict residue F1.
Source: [GrASP](https://pmc.ncbi.nlm.nih.gov/articles/PMC11182664/).

## Recommended next steps

1. **P0: repair and audit evaluation/input integrity.** Use key-based alignment,
   explicit missing masks, cache finite checks, logits finite checks, consistent
   precision, inclusion ledger, and BP/LAA denominators. Re-evaluate unchanged
   seed44 checkpoints first. Do not launch another large backbone sweep before
   establishing the corrected baseline.
2. **P1: distinguish candidate availability from selection without training.**
   Export probabilities/CA coordinates; cluster predicted contact regions using
   validation-fixed rules and report candidate site recall, oracle candidate F1,
   predicted-score-selected F1, and chemical-equivalence-aware diagnostics. GT
   oracle selection is diagnostic only, never the deployment metric. A large
   oracle-versus-actual gap supports an assignment branch; low oracle recall
   instead points to proposal/representation failure.
3. **P2: explicit ligand-conditioned candidate selection.** Keep the strong trunk;
   represent several predicted site masks, pool ligand-independent protein site
   embeddings, score them with ligand local54/graph features, and couple these
   scores back to the residue output. Mask equivalent sites and use multi-positive
   training when appropriate. Merely adding a disconnected retrieval auxiliary
   loss is insufficient. Existing virtual tokens can seed this design, but their
   current behavior is not already a site-set predictor.
4. **Re-audit negatives on the current strict merged training set.** The historical
   10.89% valid-same-protein-negative figure was for an earlier PDBbind preflight,
   not this PDBbind+BioLiP split. Do not assume either abundant negatives or the
   same scarcity. Train only with allowed training metadata; external benchmark
   labels remain evaluation-only. Preserve strict SI/coverage exclusions.
5. **Guardrails:** raw strict F1 and protein-balanced F1, distinct-site subset
   selection, candidate recall, BP together with its coverage, ligand-swap
   sensitivity, and no-contact/NaN counts. Retain same-chemistry copies as a
   separately reported task, not hidden by post-hoc filtering.

Defensible present claim: performance is stable within the tested ESM3 seeds
but degrades strongly with protein/ligand-instance multiplicity. The current
model/output/evaluation does not disentangle pocket instances explicitly, and
confirmed input defects coexist with that limitation. Neither a solved detection
stage nor guaranteed gains from a new assignment head has been demonstrated.
