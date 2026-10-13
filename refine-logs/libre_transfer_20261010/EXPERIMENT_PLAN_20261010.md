# LiBRe-inspired transfer experiment plan

Date: 2026-10-10. Status: proposed, NOT implemented or submitted.
This is separate from the historical assignment-head plan in the parent folder.

## Problem and claims

The matched HOLO audit shows a residue-ranking and candidate-coverage gap, not
just an operating-threshold or candidate-selection gap. At the same Top-20
budget, LiBRe/PLC sample F1 is 0.404411/0.322387. Released LiBRe and PLC have
different training populations and protocols, so this does not identify a causal
architecture advantage. No new pretrained model or shared protein-ligand pose
input is proposed.

| Hypothesis, not established claim | Minimum useful evidence | Main control |
|---|---|---|
| Explicit product channels improve ligand-conditioned residue ranking | Strict-VAL ranking gain over a matched non-product head, followed by seed confirmation and retained real/shuffle gap | Same-width, same-parameter additive readout |
| More task-aligned residue supervision improves per-ligand discrimination | Gain from a loss-only arm with unchanged inference, contact counts and coverage | CNC-on unchanged baseline; classmass separately |

Anti-claims: more parameters, threshold changes, a selected seed, benchmark
filtering, or test exposure alone explain a gain. This is transfer of known
mechanisms, not a claim that multiplication or class contrast is novel.

## Frozen protocol

- Strict merged TRAIN/VAL/TEST: 10317/647/647; split SHA256
  `170a4e6364c19662ad922ace7957e1733409dfe146ebdb74b15c95130bb622bc`.
- Primary base: GFGF seed44 V11m13 group_rescompat from the source-frozen
  `protein_hybrid_gdn_20261006/code_v1` recipe. Preserve all current local54,
  FiLM, virtual-token and classifier paths except the stated variable.
- Start all training arms from scratch, no frozen trunk and no checkpoint
  resume. The shared modules of paired readout arms must start identically.
- Keep preprocessing, TRAIN-fitted normalization, label-axis overlay, atom cap,
  residue cap, batch, optimizer, scheduler horizon and stopping policy matched.
  Keep best-F1 selection as requested; also retain best-AUPRC as a secondary
  checkpoint, not an extra test-selection search.
- Do not interpret the previous 35 epochs as a new 35-epoch cosine horizon.
  Existing runs were early stopped; report both matched-epoch and best-VAL
  summaries when training durations differ under the shared stopping rule.
- Pilot seed44, then seeds43/45 for surviving arms and matched controls. Three
  seeds, not one selected maximum, are needed for a stable performance claim.
- Keep torch.compile off. No new foundation-model downloads or features.

## Block 1: product versus additive readout

Insertion point: in V11m10's inherited `_compute_pair_delta`, after residue-chem
compatibility and virtual-token refinement of `lig_read`, before construction of
`delta_in`. This retains both the raw local54 path and existing co-attention.

```text
u_i = LayerNorm(Linear_no_bias_512_to_64(hp_src_i))
v_i = LayerNorm(Linear_no_bias_512_to_64(lig_read_i))

product arm:  z_i = u_i * v_i
control arm:  z_i = (u_i + v_i) / sqrt(2)

delta_in = concat(hp_src, lig_read, attention_stats4, z)
delta_logits = existing_local_delta_head(delta_in)
```

Only widen the local-delta head. The separate `center_aux_head` must continue to
consume the original 1028D `[hp_src, lig_read, stats]`, not the widened input.
Otherwise the probe changes a second classifier and the parameter budget below
is wrong (or the forward fails on a shape mismatch).

Both heads widen their input from 1028 to 1092, keeping hidden width256,
normalization, dropout, scale and three output classes the same. Extra parameter
count is identical: 65,536 projection weights +256 projection-LN parameters
+128 widened-input-LN parameters +16,384 head weights =82,304. This is an
algebraic budget, to verify against instantiated modules before submission.
Zero padded/missing-ligand interaction channels in both arms; never pool padding.
Preserve the same initialization policy in both widened heads.

The additive control deliberately lacks an explicit cross-product; its redundant
additive channels control parameter budget and input widening, not equal
functional expressivity. The existing unchanged baseline remains a third
reference. This is a local-read product, not exact LiBRe global-vector fusion.

Success: product beats both unchanged and additive controls on strict-VAL F1 and
AUPRC/fixed-budget ranking, with retained ligand-swap sensitivity. Failure:
product and additive improve equally, or only the selected gate changes F1.
Target: primary matched-control table, not a backbone-scaling claim.

## Block 2: supervision-only controls

`plc_cnc_off`: change only `training.cnc.enabled` to false. Keep class weights,
focal/Tversky mixing and distance supervision unchanged. This can test whether
downweighting confident far-labelled residues impedes pair-specific ranking.
Observed CNC late activity is only about0.72% of all TRAIN residues, so no large
effect is assumed. Report FP locations, predicted counts and coverage; improved
precision through universal shrinkage alone is not success.

`plc_residue_classmass`: separate arm, CNC unchanged. Use the existing pre-final-
linear local-delta hidden representation, sampled during training only. Initial
proposal: lambda0.02, temperature0.2, at most32 contact +32 far residues per pair;
exclude near/padding. With batch16 this bounds the contrast set at1024 residues.
Use normalized embeddings, diagonal exclusion, stable logsumexp, and skip
anchors with no other same-label positive. Average per class within each protein
before averaging proteins. Handle zero-contact cases without NaN or changing the
main residue loss. Log eligible anchors and observed per-class counts.

This borrows the released positive-mass objective, not canonical SupCon, and
sampling/averaging differ intentionally from the public LiBRe implementation.
Do not mark randomly shuffled ligands as experimentally confirmed nonbinders.
Success requires ranking gains with maintained paired-ligand sensitivity, not
only lower auxiliary loss or cleaner UMAP plots. Target: loss-only ablation.

## Block 3: conditional, not first-wave expansion

Before changing co-attention scales, measure per-block valid-residue
`norm(actual_scaled_update)/norm(input)` and contact-margin changes on strict
VAL. Existing TB norm averages include padding and precede gating. Only test a
conservative scale schedule if these measurements support underuse.

A separately masked, scratch LiBRe-style model on the same strict data would
help distinguish data recipe from architecture, but requires raw ESM cache and
ligand40/label alignment checks. Proper packed sequence/attention masks make it
a clean LiBRe-style baseline, not unchanged released LiBRe. Do not silently
claim exact reproduction of its uncertain checkpoint training recipe.

Cut from the first wave: more co-attention depth, another protein LM, multiple
new paths in one arm, or protein-ligand cross-distance inputs.

## Metrics and decisions

Keep best-F1 checkpoint selection fixed. Report strict-VAL sample- and protein-
balanced F1, AUPRC, fixed Top20 and Top L/20/L/10/L/5, precision/recall, predicted
contact-count quantiles, and candidate coverage. Global/macro definitions must
remain explicit. Real/shuffle use one VAL-fixed gate for every intervention;
zero is a diagnostic, not a robustness target to maximize. No GT-guided proposal
construction or test-count selection.

Suggested pilot go/no-go: at least+0.003 absolute VAL sample F1 together with a
consistent ranking improvement over the appropriate control. This is an
engineering screening rule, not statistical significance. Report protein-paired
bootstrap intervals and confirm with seeds43/45 before a performance claim.
Freeze all choices before TEST/COACH/HOLO evaluation, preserving the accepted
ten-case shared benchmark exclusion. Reserve an untouched final evaluation
population for publication because these external sets already informed design.

## Run order and budget

| Stage | Work | Gate | Cost/status |
|---|---|---|---|
| M0 | Mask/gradient/parameter-count tests, small real-batch smoke | Finite loss, identical shared initialization, correct masks | Pending; no training implementation yet |
| M1 | seed44 product and additive control | Valid implementation; same recipe as frozen base | Two single-GPU training arms |
| M1 diagnostic | seed44 CNC-off | No architecture or inference changes | One single-GPU training arm |
| M2 | seed44 classmass | Sampler/denominator tests pass | One additional single-GPU arm |
| M3 | seeds43/45 for survivors and controls | Strict-VAL improvement plus ligand guardrails | Conditional |
| M4 | Fixed-protocol held-out evaluation | Selection frozen | Conditional |

Prior FFFF/GFGF runs took about4.1/4.4 GPU-hours before loss-based early stopping;
use this only as an observed reference, not an ETA. Measure pilot throughput and
stopping duration before budgeting all seeds. One GPU per task; if Star is used,
only one task occupying one GPU at a time. No jobs have been submitted here.

## Risks and checklist

- Released LiBRe training overlap remains unaudited; no architecture-causality or
  leakage-controlled SOTA claim follows from the current comparison.
- Product features and class contrast can strengthen a protein prior instead of
  ligand specificity; retain intervention and equal-budget metrics.
- CNC-off can punish genuinely missing annotations; it is a control, not a
  presumed correction.
- [x] Main hypotheses and parameter/loss controls defined.
- [x] Simplicity check and no-shared-coordinate constraint explicit.
- [x] First-wave versus conditional work separated.
- [ ] Training arms implemented, tested and approved for submission.
- [ ] Multi-seed confirmation and untouched final evaluation available.
