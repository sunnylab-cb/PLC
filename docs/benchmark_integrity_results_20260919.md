# Corrected benchmark and candidate-availability audit

## Status and scope

- P0 full input audit `12793034`: completed, 12m28s. COACH420 494/494 and
  HOLO4K 7245/7245 passed, no excluded rows.
- P0 frozen-checkpoint evaluation `12793043`: completed, 47m25s, one GPU.
  Two seed44 checkpoints x three controls x two datasets. No OOM or nonfinite
  evaluated logits. Same sample-key sets, checkpoint hashes, saved val gates and
  training-stat hashes were checked across controls.
- P1 CPU candidate audit `12803928`: completed, 39s, eight CPUs, no GPU.
  All corrected prediction arrays reproduced their CSV F1 to numerical tolerance.
- Current local validation: 32 related tests pass. No training was launched and
  no benchmark labels were used to fit a model, threshold or candidate selector.

These are evaluated subsets: 299 COACH420 proteins and 3430 HOLO4K proteins,
not the nominal complete 420/4009 benchmark collections. Conclusions below are
for seed44, not new multi-seed evidence.

## P0 controlled results

All rows in this table use BF16 and the same explicit raw-SASA fallback policy.
Therefore these differences isolate the audited alignment/masking/coverage
intervention, not the combined FP16-to-BF16 and NaN-repair change in older reports.

| Model / dataset | Legacy F1 | Key-aligned, old budget | Corrected protein-axis F1 | Total delta |
| --- | ---: | ---: | ---: | ---: |
| base44 / COACH420 | 0.446231 | 0.446249 | 0.449910 | +0.003678 |
| ESM3gp44 / COACH420 | 0.443429 | 0.443251 | 0.446987 | +0.003557 |
| base44 / HOLO4K | 0.319797 | 0.327157 | 0.327468 | +0.007671 |
| ESM3gp44 / HOLO4K | 0.317814 | 0.325635 | 0.325681 | +0.007867 |

Protein-resampled paired bootstrap, 2000 replicates, RNG seed44:

- HOLO4K base44 total sample-F1 delta 95% CI: [0.006185, 0.009231].
- HOLO4K ESM3gp44 total sample-F1 delta 95% CI: [0.006314, 0.009416].
- Corrected HOLO4K ESM3-minus-base sample F1: -0.001787,
  CI [-0.003920, 0.000349]. No convincing sample-F1 win for ESM3 here.
- Corrected HOLO4K ESM3-minus-base protein-balanced F1: -0.005608,
  CI [-0.008555, -0.002781]. This is still an evaluation-protein bootstrap of
  one training seed, not evidence about training-seed variance or all ESM3 models.

Most HOLO4K gain comes from alignment/masking with the OLD residue budget; adding
the previously lost residue tail contributes only +0.000311 / +0.000046.

| Corrected result | Sample F1 | Protein-balanced F1 | LAA | PDR | Conditional BP F1 | BP rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base44 / COACH420 | 0.449910 | 0.540237 | 0.166090 | 0.493927 | 0.722433 | 154/494 |
| ESM3gp44 / COACH420 | 0.446987 | 0.537304 | 0.169550 | 0.508097 | 0.724959 | 157/494 |
| base44 / HOLO4K | 0.327468 | 0.442011 | 0.137520 | 0.410904 | 0.707148 | 1621/7245 |
| ESM3gp44 / HOLO4K | 0.325681 | 0.436403 | 0.130582 | 0.403313 | 0.723042 | 1581/7245 |

BP is evaluated only after the predicted center localizes the CURRENT pocket.
The BP subsets differ between models. A higher conditional BP cannot establish
better overall detection. LAA is evaluable for 289 COACH rows and 5621 HOLO rows;
COACH's old-budget LAA denominator was 285, so its raw change is not same-subset.
There are five zero-contact rows in each corrected dataset; these remain in
strict F1. No positive-label row has a nonfinite AP after repair.

## What the audit fixed

| Audit flag | COACH rows / proteins | HOLO rows / proteins |
| --- | ---: | ---: |
| ESM2 missing evaluated residues | 51 / 12 | 810 / 382 |
| ESM2 raw positional mismatch | 0 / 0 | 537 / 273 |
| Historical tag-axis positional mismatch | 4 / 1 | 292 / 143 |
| Extra tag-only residues | 5 / 3 | 29 / 21 |
| ESM3 source duplicate keys | 0 / 0 | 7 / 5 |
| ESM3 missing evaluated residues | 0 / 0 | 259 / 117 |
| Explicit ligand SASA fallback | 0 | 10 |

Every raw label distance vector was verified against the original atom+H PDB and
ligand SDF. Maximum absolute reconstruction error: 3.81e-6 A / 7.63e-6 A.
The seven key-only failures were 1a4w, 1ceb (two ligands), 1dv2 (two), 1fpc,
and 1rtf. Unique same-chain CA and sequence identity resolved these without
dropping samples or overwriting source caches.

The 292 HOLO rows flagged for tag-axis shift improved by +0.146401 F1 on base44
and +0.150676 on ESM3gp44. These are bundled repair effects in a flagged subset,
NOT an isolated causal attribution to labels alone; input flags can overlap.
Raw ESM3 positional mismatches alone are not proof of wrong old model inputs,
because the old loader already attempted key alignment for unequal lengths.

## P1: simple post-hoc component selection is not the solution

Candidate protocol fixed before running: connected components of predicted
contact residues, CA edge radius 8 A, minimum three residues; score by mean of
the top 20% contact probabilities. Radii 6/10 A are sensitivity controls and all
are reported. Neither candidate generation nor actual selection sees labels.
Oracle selects the highest-F1 candidate with GT and is DIAGNOSTIC ONLY.

| HOLO4K / radius | Original F1 | Score-selected F1 | GT-oracle candidate F1 | Mean candidates |
| --- | ---: | ---: | ---: | ---: |
| base44 / 6 A | 0.327468 | 0.20845 | 0.26537 | 2.511 |
| base44 / 8 A (primary) | 0.327468 | 0.30983 | 0.32295 | 1.291 |
| base44 / 10 A | 0.327468 | 0.32556 | 0.32787 | 1.071 |
| ESM3gp44 / 6 A | 0.325681 | 0.20239 | 0.25478 | 2.492 |
| ESM3gp44 / 8 A (primary) | 0.325681 | 0.30211 | 0.31692 | 1.288 |
| ESM3gp44 / 10 A | 0.325681 | 0.32284 | 0.32637 | 1.048 |

COACH420 primary-radius result is also negative: base original/selected/oracle
0.44991/0.42183/0.43572; ESM3 0.44699/0.40054/0.42162.

For HOLO base44, any-candidate recall at DCC<4 A is 0.2141, and at IoU>=0.25 it
is 0.4663. These are per-current-target candidate coverage diagnostics, not the
old any-pocket PDR. Small radius fragments useful boundaries; large radius
mostly reproduces one candidate. Even perfect ranking within this family offers
almost no whole-set gain. This rejects the specific proposal "split the current
thresholded mask and add a ranking head", NOT every possible learned proposal
or assignment mechanism. Low component-oracle coverage does not establish an
intrinsic encoder upper bound.

## Multiplicity and chemical identity

| HOLO ligand rows per protein | Rows | Corrected base44 F1 | Corrected ESM3 F1 |
| --- | ---: | ---: | ---: |
| 1 | 1619 | 0.603901 | 0.595090 |
| 2 | 2132 | 0.367204 | 0.359504 |
| 3-4 | 2019 | 0.221991 | 0.225917 |
| 5-9 | 1143 | 0.123726 | 0.129488 |
| >=10 | 332 | 0.067142 | 0.076849 |

Single-row COACH F1 is 0.622237 / 0.617461. Dataset composition therefore matters;
the HOLO drop is not uniform over proteins or explained by seed luck alone.

Chemical identity uses sanitized canonical heavy-atom isomeric SMILES, not just
PDB ligand code. It was available for 491/494 COACH rows and 7238/7245 HOLO rows.
Unknowns remain in strict metrics and are explicitly absent from chemical-only
metrics. No tautomer or protonation-state equivalence is assumed.

- 4527 HOLO rows have repeated identical chemistry within their protein.
- In that subset, base strict-instance F1 is 0.20565; scoring the unchanged
  prediction against the best same-chemistry target gives 0.48500. ESM3 gives
  0.20800 versus 0.49873. These are relaxed/oracle-label diagnostics, NOT improved
  strict predictions or substitutes for the main benchmark.
- Across 4827 same-chemistry, distinct-pocket pairs, base prediction Jaccard is
  0.91798, target centers differ by 43.61 A on average, and symmetric ligand-swap
  F1 gap is 0.00060. ESM3: Jaccard 0.92774, gap 0.00098.
- Across 3674 different-chemistry distinct-pocket pairs, base Jaccard is 0.64076
  and swap gap 0.01237; ESM3 Jaccard 0.68697 and gap 0.01242. Protein-balanced
  swap gaps are 0.02422 and 0.01971, respectively.

Identical chemistry is not proof of identical full feature tensors (conformers,
atom ordering, SASA etc. may differ). Nonetheless, without a relative pose or
explicit instance cue, chemically equivalent copies need careful treatment:
the model should not be expected to distinguish arbitrary copies by their names.
The different-chemistry subset remains weak, so copy ambiguity is not the only
problem either.

## Revised next step

Do not start a new large backbone sweep or deploy the failed component selector.
Do not jump directly to training a ranker over the current thresholded output.

1. On strict VALIDATION, test a richer whole-protein candidate proposal mechanism
   independent of the final contact threshold, then freeze its protocol. Examples
   are ligand-independent spatial proposals or jointly trained multi-site masks.
   First establish increased candidate coverage/oracle F1 without GT proposals
   at inference. Candidate generation must improve before ranking can help.
2. Re-audit the CURRENT strict PDBbind+BioLiP TRAIN split for distinct-site and
   chemical-equivalence coverage. The old PDBbind 10.89% number is not current.
   The existing legacy preflight compares centroids from separate structures
   without a coordinate-frame check; do not reuse that artifact blindly. Exact
   residue-order matches alone do not establish a shared coordinate frame.
3. Only then couple local54/graph candidate compatibility into residue inference,
   with multi-positive treatment of equivalent sites and masked distinct-site
   negatives. Keep raw strict F1 and protein-balanced metrics as guardrails;
   report relaxed same-chemistry results separately.

No P2 training was launched: P1 invalidated the prerequisite that useful
alternative sites are already present in the current thresholded candidate set.

## Reproducibility

- Paired results and bootstrap: `scripts/diagnostics/summarize_benchmark_integrity.py`.
- Candidate/swap/chemistry audit: `scripts/diagnostics/benchmark_candidate_audit.py`.
- CPU submission: `scripts/slurm/run_benchmark_candidate_audit.slurm`.
- Local results: `tmp/benchmark_integrity_20260919/{paired_analysis.json,candidate_12803928/}`.
- Remote artifacts: `/hpc2hdd/home/shuang886/PLC/experiments/benchmark_integrity_20260919/`.
- Original weights, feature caches, old benchmark files and training code were not
  changed by this continuation. New diagnostic source lives in remote `code_p1/`.
