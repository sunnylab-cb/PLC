# Parallel assignment experiments

Date: 2026-09-26. Status: proposal only; no new branch submitted.

## Problem and evidence

The current hypothesis is that candidate selection and ligand-specific boundary
decisions limit transfer more than another larger encoder. This is not proven
by conditional BP F1 alone: BP evaluates a selected localized subset and must
always be reported with its denominator/coverage.

Audited strict VAL base44 sample F1 is 0.521868. The hybrid candidate pool,
including the original prediction, has oracle F1 0.644453. Probability/count
selection achieves 0.521832; top-20%-probability selection collapses to tiny masks.
The oracle uses GT to select masks and is NOT an achievable performance forecast.
The strict same-axis, equivalent-geometry, known-different-chemistry negative
audit finds 179/10317 eligible TRAIN rows (1.74%). This is a conservative bound
on directly transferable region negatives, not all possible contrastive pairs.

Source: `../docs/strict_assignment_results_20260920.md`.

## Claim map

| Hypothesis, not established claim | Minimum convincing evidence | Blocks |
| --- | --- | --- |
| C1: learned ligand-site evidence improves selection beyond probability/size heuristics | Strict held-out F1 gain; parameter-matched ligand-blind control; query intervention; retained count distribution; seed confirmation | B1, B2 |
| C2: local refinement improves residue boundaries beyond selecting an existing mask | Improvement over unchanged logits and matched protein-only refiner; fixed-gate comparison; no GT proposal input | B3 |

Rule out gains from a larger candidate search space, extra parameters alone,
threshold drift, selecting tiny high-confidence fragments, or GT-dependent
inference. The eventual external claim concerns ligand assignment, not just F1.
No new foundation-model component or architecture novelty is claimed.

## Shared protocol and immediate blocker

Use strict merged TRAIN 10317 / VAL 647 / TEST 647, split SHA256
`170a4e6364c19662ad922ace7957e1733409dfe146ebdb74b15c95130bb622bc`.
Frozen base44 checkpoint SHA256:
`d8d54dd9d8183f8c8038a656dd0d3937adac325434af7d7ef32783afc3e6e848`.
Use the source-verified label-axis overlay, existing normalization and current
atom/residue budgets unchanged. Do not mix cap changes into these comparisons.
The protocol is conditioned on the existing trained backbone, not a scratch
backbone comparison. New heads start from scratch. No torch.compile.

Smoke 12868601 FAILED on pre-cross-state independence; 12868608 was cancelled
by afterok protection. Full feature export and head training have NOT completed.
Before resubmitting, repeat identical forwards, clone at module boundaries, check
input mutation, and compare FP32/BF16 interventions. Do not just raise tolerance.
The code path inspected is protein-only before cross-attention, but runtime
cause remains unresolved. Record any revised numerical policy and evidence.

After that passes, export shared caches once. Current proposed schema stores
mean-pooled hidden states only. B2/B3 need a NEW token-level export, not an
attempt to reconstruct tokens from pooled vectors. Store per-residue/atom
states once per pair plus masks/indices; do not duplicate every candidate's
tokens or store dense candidate-residue-atom tensors. Check disk budget first.

All pilots use seed44, LR 2e-4, at most 20 epochs and frozen backbone; B1 pooled
batch 32, B2/B3 token batch 8 as initial smoke settings. Log parameter count,
actual peak memory and throughput before estimating production cost. Confirm
successful branches with head seeds43/45; this alone does not establish
backbone-seed robustness.

## B1: original-relative candidate gain

- Run: `site_gain_rank` (P0). Purpose: learn when replacing the original helps,
  not merely whether a candidate looks like a high-confidence pocket.
- Target: `gain_j = F1(candidate_j) - F1(original)` on TRAIN only. Input includes
  candidate and original features, with shared projections and explicit
  differences. Regress signed gain; retain a near-tie region rather than forcing
  nearly equivalent masks into a strict ranking. Use the same fixed 0.02
  replacement margin as the existing pilot, now in predicted-gain units.
- Control: current absolute-quality head already uses MSE plus a soft ranking
  loss. This is NOT the first ranking experiment. Use a matched paired-feature
  absolute-quality control to separate the target change from added context;
  keep candidate masks fixed across all comparisons.
- Reuse pooled cache. Train only small heads; no additional backbone passes.
- Decisive metrics: protein-balanced selected F1; sample F1; replacement rate;
  improved/worsened counts and gain magnitudes; regret relative to pool oracle;
  contact-count p50/p90 and performance by original prediction size.
- Stop if gain prediction is optimistic on TRAIN but no better on VAL than
  absolute quality or retaining the original. Approximate +0.003 VAL F1 is a
  pilot screen, not a statistical-significance claim.
- Paper role if successful: main selection ablation. Literature motivates
  learned mask quality [1] and metric-aware ranking [2], not this exact loss.

## B2: atom-residue late interaction

- Run: `site_lateinteraction` (P0). Purpose: test whether averaging ligand/site
  hidden states loses the chemical matches needed to distinguish candidates.
- Use independently encoded ligand atom and candidate residue tokens, projected
  to 64D and normalized. Start with a bidirectional, length-normalized soft
  matching summary (log-mean-exp of token similarities), plus current 19 stats.
  No native protein-ligand distances; only within-protein/within-ligand geometry.
- Chemistry-bearing atom context comes from the graph/local54 path. Do not add
  several new chemical adapters in the same comparison. Candidate chunks avoid
  materializing all masks times all residues times all atoms simultaneously.
- Keep the existing quality objective/fallback for the first test. Compare
  against a capacity-matched mean-pooled head and a ligand-blind site scorer.
  Query permutation/zero with fixed candidates is a HEAD-ONLY control, not full
  model ligand dependency. A positive branch later needs full real/shuffle/zero.
- New token cache required. One GPU per job; report normalization and candidate
  size effects because even token matching can favor larger pockets.
- Success: improves over matched pooling on strict VAL without contact-count
  collapse and uses the added ligand query. A protein-only gain is not evidence
  of improved assignment. Failure would challenge the pooling-loss hypothesis.
- Paper role if successful: main mechanism/control. ColBERT [3] supports cached
  independent encoders plus fine-grained late interaction in IR; the atom/site
  transfer is our hypothesis, not a published result for this model.

## B3: candidate-conditioned boundary refinement

- Run: `site_boundary_residual` (P1). Purpose: test a failure mode that selection
  cannot fix, where a promising region has the wrong residue boundary.
- Freeze V11m13. A small residue-level head receives protein state, original
  contact log-odds, pooled candidate context and ligand context. Operate inside
  each candidate and a fixed protein-CA 4 A shell, processing candidates in
  chunks. Add a signed bounded residual to binary contact log-odds; start with
  zero output and fixed maximum amplitude 0.5. Outside the shell, preserve
  original logits. Preserve original output as a fallback candidate.
- TRAIN uses current per-ligand contact labels with per-sample normalization,
  never GT centroids or GT masks to define the candidate region. Treat labels
  as this annotation task, not proof that other pockets cannot bind the ligand.
- Control: same-capacity protein-only refiner. Train a separate refined-mask
  quality head using TRAIN targets; do not select refined masks using VAL GT or
  an unvalidated raw-mask quality scorer. Keep this selector protocol matched
  between ligand-aware and blind refiners. Do not depend on B1/B2 winning.
- New shared token cache required. Two head runs including the control; more
  engineering than B1/B2. Unlike V11m16, this predicts per-residue corrections,
  rather than a tiny centered scalar site gate.
- Success: strict VAL F1/precision/recall improve at the ORIGINAL contact gate;
  the extra path beats protein-only refinement and does not simply shrink masks.
  Report all-row F1 and contact counts, not conditional BP alone. Stop on broad
  boundary degradation even if a conditional assignment metric improves.
- Paper role if successful: supporting refinement result. PointRend [4] is an
  analogy for selective boundary refinement, not evidence for protein transfer.

## B4: TRAIN-only multi-site annotation audit (CPU, optional)

- Run: `train_multisite_union_audit` (P1, no training yet). Audit which TRAIN
  structures permit merging known site labels using verified residue mapping
  and equivalent geometry. Canonicalize chemical identity and equivalent copies.
- Count unique proteins, distinct pockets and chemistry groups after merging,
  annotated/unknown residues, and sample reweighting. Do not equate 70% repeated
  protein rows with 70% useful distinct-pocket negatives.
- Never transfer homolog residue masks by index. No val/test/COACH/HOLO labels
  enter TRAIN unions. New external data would require a fresh leakage audit.
- If coverage supports it, later compare a protein-only multi-site proposer
  with the same existing selector. Do not silently mark all unobserved sites
  biologically negative. Partial-annotation handling must be specified before
  training; this audit alone is not a complete PU-learning recipe.
- UniSite [5] motivates protein-centric multi-site annotation and set prediction;
  it does not establish that a ligand-specific selector will work here.

## Run order and decision gates

| Stage | Work | Parallelism | Stop/go |
| --- | --- | --- | --- |
| M0 | Resolve runtime independence; reproduce audited VAL output | One existing pilot retry after diagnosis; CPU B4 can be independent | No full cache/head work until smoke passes |
| M1 | Shared pooled/token export; current three-head control | One exporter; then reusable immutable cache | Hashes, axes, baseline F1, finite states, memory/disk budget |
| M2 | B1, B2, B3 with their controls | Independent HPC single-GPU jobs, initially at most 3; Star at most 1 job/1 GPU | Each compares against its own matched control |
| M3 | Seeds43/45 for successful heads | Only branches passing strict VAL guardrails | No ranking by external benchmark during tuning |
| M4 | Freeze protocol, strict TEST and COACH420/HOLO4K | Final evaluation, full real/shuffle/zero | Protein-balanced LAA, per-ligand F1, PDR, BP with denominator |

Reserve one 12-hour GPU allocation for the current full pilot only after smoke
passes; it is a scheduler cap, not an elapsed-time estimate. Initially cap each
new head branch at 4 GPU-hours after export, profiling before extending. A
three-branch pilot envelope is 12 head GPU-hours plus measured shared export
and current-control cost; seed confirmation and future backbone retraining are
excluded. B4 can start as one 4-CPU/32G/2-hour audit request after implementation.
No resources are reserved by this document and no new jobs were submitted.

## Reporting and risks

Primary VAL model selection is protein-balanced actual selected/refined F1;
always also report sample mean F1, pooled precision/recall, head calibration,
gain/regret and predicted-count distributions. Bootstrap by audited sequence
connected component if mapping is available; otherwise label exact-structure
bootstrap limitations. VAL selection plus many proposals is not independent
confirmatory evidence. External multi-site/low-overlap subsets and equivalence-
aware diagnostics must be specified before evaluating a selected protocol.

The current backbone saw the same TRAIN examples used to fit heads. First
measure TRAIN/VAL quality, candidate count and feature-distribution shifts.
If every selector overfits, consider a backbone-out-of-fold cache using a TRAIN-
only sequence-component holdout. Cross-validating just the head does not make
the backbone features out of fold; honest OOF costs extra backbone training.

Keep top-level claims narrow: strict VAL is PDBbind-only and cannot establish
HOLO4K multi-site improvement. Same-chemistry copies may be inherently
ambiguous without pose, so do not demand arbitrary exclusive assignments.
Do not repeat V11m15's post-cross auxiliary matching or V11m16's near-zero
scalar gate unchanged. Do not expand encoder depth, ESM fusion, thresholds,
loss sweeps and atom budgets simultaneously.

## Primary literature and transfer limits

1. [Mask Scoring R-CNN, CVPR 2019](https://arxiv.org/abs/1903.00241): learned
   instance-mask quality rather than using classification confidence as quality.
2. [LambdaLoss, CIKM 2018](https://research.google/pubs/the-lambdaloss-framework-for-ranking-metric-optimization/):
   metric-driven learning to rank. Our original-relative F1 head is not LambdaLoss.
3. [ColBERT, SIGIR 2020](https://arxiv.org/abs/2004.12832): independent encoding
   and token-level late interaction for document retrieval, not molecular binding.
4. [PointRend, CVPR 2020](https://arxiv.org/abs/1912.08193): point-based adaptive
   image-mask refinement; only a design analogy for B3.
5. [UniSite, NeurIPS 2025](https://arxiv.org/abs/2506.03237): protein-centric
   multi-site annotation and set prediction; not a per-ligand assignment result.

## Checklist

- [x] Each proposed branch has a distinct hypothesis and control.
- [x] Legacy comparisons, partial labels and transfer limits are explicit.
- [x] No novel-backbone or guaranteed-improvement claim is made.
- [ ] Runtime independence failure resolved.
- [ ] Matched controls implemented and real-data smoke passed.
- [ ] Strict held-out gains reproduced and final external protocol frozen.
