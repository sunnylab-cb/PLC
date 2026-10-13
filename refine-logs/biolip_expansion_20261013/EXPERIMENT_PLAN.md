# BioLiP expansion experiment plan

Date: 2026-10-13. Status update: metadata audit13001698, tensor/sequence QC13003040
and selective extraction13003132 completed. All1,390 new tensors pass first-stage
QC; no additions are approved for TRAIN. Homology/geometry/deduplication and
training below remain proposed. Parent assignment/noise plans are unchanged.

Problem: current merged TRAIN uses only 5,157 BioLiP pairs, with an effective
route5 6-50-heavy-atom chemistry restriction and sparse distinct-site coverage.
Method thesis: test whether broader, leakage-audited ligand/site coverage helps
the ORIGINAL GFGF model, without new modules or self-distillation.

## Claim map

| Claim | Minimum convincing evidence | Blocks |
| --- | --- | --- |
| C1: Clean BioLiP expansion improves generalization | Matched-budget old/expanded-data scratch comparison, seeds 44/43/45; strict held-out F1/AUPRC and external per-ligand metrics | B1-B3 |
| C2: Added chemistry/site diversity helps ligand assignment | Audited increase in independent sites and chemical identities; multi-site protein-balanced improvement with stable detection/boundary metrics; size-matched data control | B1, B3-B4 |

Anti-claims: more optimizer steps, changed feature semantics, larger batch,
new sampler, homology leakage, duplicate cofactor frequency or VAL threshold
optimization alone explains the result. Neither claim is currently supported
by a completed expansion-training experiment.

## Paper storyline

Main evidence should be one data-only intervention and a clean baseline, not a
new backbone claim. Candidate ceilings and exclusions belong in preprocessing
QC. Type/length/multisite failures and the size-matched check support the appendix.
Cut all-types naive mixing, new attention modules, KD and huge seed matrices
from the initial screen. No frontier-model novelty is claimed.

## Experiment blocks

### B1. Freeze and audit an expanded data release

MUST-RUN before any training. Freeze the current 647 VAL, 647 TEST and external
benchmark cases, normalization provenance and exact metric definitions.

First inspect the 104 existing tensors on current TRAIN chain keys and the
2,446 old-policy new all-file candidates; 3,333 is the broader-chemistry ceiling.
Recheck actual tensor route/validity, chain sequence hash, residue axis and
geometry. Rebuild candidates from the full metadata ledger. Next prioritize the
1,390 missing ligands on known TRAIN chain keys. These sets overlap and are not
approved sample counts.

Construct connected-component homology exclusions, then independently re-audit
the final allowlist at SI>=.30 and (qcov>=.80 OR tcov>=.80). Check all chains
that will enter the model, not only the contact chain. Do not reuse old split
pass flags or move fixed holdouts. Reject duplicate structure/ligand/site
observations, but preserve distinct ligands/sites on the same chain.

Produce per-reason rejection counts, independent sequences/clusters/chemistries,
valid distinct-site coverage, source/type/length distributions, feature-group
validity, total-vs-heavy atom counts and contact-count distributions. Fit any
new statistics on TRAIN only. A missing group must not become a measured zero.
Do not expose native protein-ligand relative pose as a new model input.

Success: immutable compatible candidate release with no detected violations,
valid labels and a quantified diversity gain. Failure: few usable additions
or mostly repeated sites; stop before expensive feature extraction/training.
Output: preprocessing flow table and hashed QC manifest.

### B2. Matched seed44 scratch screen

MUST-RUN after B1 passes. Compare:

| Arm | Dataset | Question |
| --- | --- | --- |
| A: original_data_s44 | Current 5,160 PDBbind + 5,157 BioLiP | Same-budget control |
| B: organic_expand_s44 | A + newly audited compatible regular-organic observations | Does more compatible data help? |
| C: chemistry_expand_s44 | B + audited cofactor/glycan/other accepted chemistry | Does broader chemistry help? |

Run A/B first; C only after missing-feature semantics are approved. Skip B as a
separate arm if it recovers too few genuinely new observations, and document why.
Never interpret a changed mask encoder as a pure data effect: A must use the
same handling as C, with any architecture change explicitly declared.

Keep original GFGF, seed44, scratch initialization, full training, B16/BF16,
LR1e-4, original losses and best-F1 selection. No context-noise/KD proposals,
torch.compile or increased capacity. Keep original inference and caps in this
first test, and quantify whether caps remove new contacts; capped-away
supervision is a stop condition, not permission to silently change labels.

Use the same PDBbind/BioLiP draw ratio (initial proposal 50:50) in all new arms.
If cluster/site balancing is needed, apply exactly the same TRAIN-only rule to
A/B/C; current weighting is uniform and is NOT an adequate balancing control.
The existing checkpoint remains a historical reference, not a substitute for
A under a changed sampling/step schedule. Review duplicate/type frequencies
before deciding whether a separate balancing ablation is needed.

Budget pilot by optimizer updates, not equal enlarged-dataset epochs. The old
35-epoch, 645-batch trajectory corresponds to 22,575 updates. A matched pilot
can use this budget with validation every 645 updates and the same LR/warmup
curve expressed in updates (old 120-epoch horizon =77,400 updates). Use a common
fixed pilot budget or compare equal-length prefixes; independent early stops
must not create unequal-compute claims. Thirty-five epochs is NOT convergence
evidence. If both arms are still rising, continue both equally.

Decisive screen: clean strict VAL F1 improvement around +.003 together with
AUPRC support and no severe P/R/contact-count collapse. This is a practical
screen, not a statistical significance threshold. Report legacy best-F1 and
global residue F1/MCC at .5 separately; VAL-optimized gate is diagnostic only.
Failure: no consistent curve/threshold-free gain; examine usable diversity and
data shift before scaling. Output: matched-update training/VAL curves and table.

### B3. Seed and held-out confirmation

MUST-RUN only for a surviving recipe. Add seeds43/45 to the matched control and
winner, report mean +/- sample SD. Freeze best-VAL-F1 checkpoints and VAL-only
operating thresholds before strict TEST real/shuffle/zero and COACH420/HOLO4K.
Keep shuffle settings identical. Zero gap is diagnostic, not an objective.

Primary metrics: test real F1/AUPRC, real-shuffle gap, external sample F1 and
protein-mean-of-ligand-mean F1. For C2, report protein-balanced LAA on multi-
ligand low-overlap cases. Keep PDR, BP_F1, protein-best F1, precision/recall and
contact-count p50/p90 as guardrails; state any GT-conditioning of BP/PDR.
Use protein-level resampling for external uncertainty, not independent-row
bootstrap on repeated ligand rows. Stratify chemistry, size and chain count.

Success: repeatable held-out gain rather than one seed, one gate or a conditional
metric. Failure: only training/VAL gains, lost shuffle sensitivity or collapsed
detection; no generalization/assignment claim. Output: main results table.

### B4. Diversity versus quantity

NICE-TO-HAVE unless claiming C2. Compare a size-matched expanded-BioLiP subset
against the original BioLiP subset with equal updates and common source/sampler
rules. Choose subsets using TRAIN-only diversity information, not benchmark
labels or test scores. Keep architecture and feature validity handling fixed.

Success: diversity coverage and multi-site gain persist without more rows or
steps. Failure: report a data-scale/mixing result instead of an assignment-
mechanism result. Output: one ablation table, not a new model sweep.

## Run order and resources

| Milestone | Action | Decision gate | Cost/status |
| --- | --- | --- | --- |
| M0 | Metadata inventory | Reproducible counts | DONE: 13001698, CPU95s, no GPU |
| M1 | Existing-file and known-chain QC | Compatible independent additions | DONE13003040; 1,031 existing tensors pass, not admitted |
| M2 | Homology/dedup, selective feature completion, data freeze | Zero detected leakage; valid feature/label axes | Extraction13003132 DONE1,390/1,390; admission QC still proposed |
| M3 | A/B seed44, then C if justified | Matched-update VAL evidence | PROPOSED; one GPU per run |
| M4 | Seeds43/45 and held-out eval | Reproducible real/assignment gain | CONDITIONAL |

Data stay on HPC2 HDD, not jhspoolers. No full metadata/tensor download is
needed. Actual TRAIN release size and sequence/atom lengths are unknown, so an
expansion GPU-hour ETA would be speculative. Old GFGF runs of 22,575 updates
took roughly 4.1-4.5 GPU-hours; this is a historical scale reference, NOT a bound
for expanded chemistry. Measure an input/forward/backward smoke before sizing
the run. Star, if used later, remains limited to one task on one GPU.

## Checklist

- [x] Current data restrictions and candidate ceilings measured.
- [x] Main claim and anti-claims separated from observed facts.
- [x] Nice-to-have ablation separated from required checks.
- [ ] Exact candidates, sequence components and valid tensors approved.
- [ ] Missing-group and sampling rules fixed for both arms.
- [ ] Matched-budget scratch control and expansion run completed.
- [ ] Seeds and frozen held-out metrics support a claim.

Evidence and code references:
[assessment](../../docs/biolip_expansion_assessment_20261013.md).
