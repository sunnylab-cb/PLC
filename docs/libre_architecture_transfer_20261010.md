# LiBRe to PLC: architecture and supervision audit

Date: 2026-10-10. Analysis only: no model edits, training launches, or changes to
the accepted benchmark population. Paper downloads and small CPU checks ran on
HPC2; no model weights or feature caches were downloaded to the workstation.

## Evidence and scope

- Paper: Kang et al., *LiBRe: A Ligand-Aware Sequence-Based Binding Residue
  Prediction Model for Virtual Screening*, JCIM 66(7), 3550-3561 (2026).
  [DOI](https://doi.org/10.1021/acs.jcim.5c02883),
  [open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC13080968/).
- Official code: [GoldRiver0/LiBRe](https://github.com/GoldRiver0/LiBRe), pinned
  commit `e9151a85bab2d05b67b953bc55a74c780aa2bf24`, matching the reproduced weights.
- PLC comparator: `protein_hybrid_GFGF_scratch_seed44_v11m13_20261008_202709`,
  `best_f1_c0_model.pt`; stored epoch is 31. This is the actual matched benchmark
  comparator, not a claim that it is the best historical PLC model on every metric.
- The inspected local V11/V11m3/V11m4/V11m6/V11m8/V11m8x/V11m10/V11m13 model
  files and `train_v11m4.py` have matching SHA256 hashes to the frozen remote
  `protein_hybrid_gdn_20261006/code_v1` snapshot.
- Raw paper text, source snapshots, the exact config and small diagnostic results
  are under `tmp/libre_architecture_review/`. Source hashes are recorded there.

Remote literature directory:

```text
/hpc2hdd/home/shuang886/PLC/external_baselines/libre/literature/20261010/
```

`libre_main.pdf` is a validated publisher PDF, 5,237,109 bytes, SHA256
`21610ebc30e06d42e2799cfcb7a3b498088eb8bbb3ab059c5399d00359733080`.
`libre_fulltext.xml` was obtained through the official Europe PMC REST API,
SHA256 `52d687b9ea0662267a035d237abf698214147ce576a8bf44401dede4522ac435`.
The main PDF was converted with `pdftotext`; its architecture figure was visually
checked on page 3. The supplementary PDF also downloaded successfully through the
Europe PMC supplementary-files API after a slower retry: 1,681,267 bytes, SHA256
`dcfec3b619e034f91a0204550e120f24a576fe65968bc7175a682c0c52ae09bc`.
Its text confirms additional UMAP examples, not a new controlled assignment test.
See `download_manifest.json` for successful sources and earlier failed attempts.
Do not mistake a returned HTML challenge for a PDF.

## What the benchmark establishes

From [the matched ranking audit](libre_ranking_diagnostics_20261010.md), HOLO
Top-20 F1 is 0.4044 for LiBRe versus 0.3224 for PLC. At PLC's native prediction
count, LiBRe obtains 0.4210 versus 0.3300. Tie randomization does not change the
conclusion. Single-ligand proteins also show a ranking gap. PLC's native-mask
component oracle F1 is only 0.3213 under the audited proposal rule.

Therefore, simply broadening PLC predictions or reranking its existing sparse
components is insufficient. These observations motivate investigating residue
ranking and proposal coverage, but do not identify an architectural cause.

## Paper-specific findings and limits

The study trains on PDBbind 2020 plus sc-PDB 2017, filters sequences above 1500
residues and samples with fewer than ten binding residues, and reports 90%
training sequence deduplication. Test filtering removes shared PDB IDs and
Foldseek similarity above 0.4; this is not our pairwise SI30 protocol.

Its HOLO ablations report F1 0.6225 for the full model, 0.4918 without ligand,
0.5073 without CNN/BiLSTM, and 0.5570 without contrastive learning. ESM-C 600M
gives 0.6184. These are its own filtered-population results, not expected PLC
gains. Reported training uses weighted BCE plus supervised contrastive learning,
AdamW, weight decay 1e-5, and dropout 0.1. A threshold of 0.6 is selected by
cross-validated F2. The supplement is described as additional UMAP examples.

This supports testing interaction readout and representation supervision rather
than assuming that a larger protein language model is necessary. It does not
establish that BiLSTM is intrinsically better than our GDN/attention trunk.

## Released implementation versus current PLC

| Aspect | Released LiBRe | Audited PLC GFGF |
|---|---|---|
| Protein features | ESM2 L33, 1280 dimensions | ESM2 plus 62D physicochemistry, 89D spatial scalars and vector geometry |
| Protein context | Two width-3 CNNs: 1280 -> 256 -> 64; three-layer bidirectional LSTM, 32 hidden per direction | Three GVP layers; gated geometry/ESM fusion; 512D GFGF protein context |
| Ligand features | 40D atom chemistry, bond adjacency | 109D atom input = local54 + global55; bond features and intraligand geometry |
| Ligand context | Three GAT layers, then mean pool to ONE 64D token | Three graph layers, two ligand self-attention layers, atom and global tokens |
| Cross-modal processing | One attention in each direction, then shared LayerNorm | Four V12-style co-attention blocks with inner self-attention and ReZero scaling |
| Final decision | Elementwise product of residue and protein-conditioned ligand vectors, then binary MLP | Co-attention classifier plus scaled local54/graph delta; three output classes |
| Extra ligand paths | No separate local54 delta | Group K/V, residue-chemistry compatibility, FiLM, virtual tokens |
| Protein model upgrade | Frozen ESM2 features | This comparator has neither ESM3 nor MLC enabled |
| UniMol | Not used | Atom-input and context-prior UniMol paths disabled in this config |
| Training head objective | Public script defaults to binary BCE; optional class contrast | 0.7 focal + 0.3 contact Tversky, distance auxiliary loss, CNC |
| Regularization | Default model dropout 0.2; GAT attention dropout 0.2 | Multiple source dropouts, attention/drop-path regularization, AdamW decay 0.05 |

Sources: copied `source/src/models/libre.py`, `source/src/train.py`,
`source/src/utils.py`; `plc_gfgf_config.yaml`; local files listed below.

Important corrections to historical shorthand:

1. **PLC's base logits are not protein-only.** They come after four co-attention
   blocks. `base_logits + scale * delta_logits` means base-classifier plus local
   refinement, not an independent ligand-free predictor plus ligand correction.
2. **We already have ligand-conditioned multiplication.** FiLM applies a bounded
   multiplicative/additive update inside the local-delta path. The difference to
   test is an explicit product at the final classifier input, not "add FiLM".
3. **GFGF is not Qwen's one-full/three-linear schedule.** It has two bidirectional
   GDN blocks and two full-attention blocks, before the co-attention stack.
4. **`use_center_aux_head` is not proof of centroid supervision.** This branch
   contributes an extra class-0 logit; inspect its actual loss before calling it
   an independently supervised pocket-center predictor.

## What LiBRe's attention actually computes

Let `R` be `[B,L,64]` and pooled ligand `g` be `[B,1,64]`.

```text
r_to_l = MHA(query=R, key=g, value=g)
l_to_r = MHA(query=g, key=R, value=R)
R_new  = LN(R + r_to_l)
g_new  = LN(g + l_to_r)
z_i    = R_new[i] * g_new[0]
logit_i = MLP(z_i)
```

The first attention has only one key, so its softmax is exactly `[1]` for every
residue and head. Its output is a projected ligand vector broadcast over all
residues, independent of the query. It is not atom selection, and the Q/K score
projections in this direction have zero gradient through this computation.
The residual and LayerNorm still yield different ligand-conditioned residue
states because each residue's original vector is different.

The reverse attention is nontrivial: a ligand query selects a weighted mixture
of residue states. That protein-conditioned ligand vector then participates
directly in **every final classification** through the product.

Using the released weights, the CPU algebra probe returned attention weights
exactly 1, zero output change after changing the queries, and zero Q/K gradients.
The ligand value path has nonzero gradient. See
`single_token_attention_probe.json`; runner:
`tmp/probe_libre_single_token_attention.py`.

For a linearized product head, `score_i = sum_c w_c * R_i,c * g_c`. Thus the
effective residue classifier is ligand-dependent: `w_effective = w * g`.
Two toy residues `(2,0.5)` and `(0.5,2)` score `(2.5,1)` under ligand context
`(1.2,0.2)`, but `(1,2.5)` under `(0.2,1.2)`. The ordering flips. A shared scalar
logit bias cannot do that. This is a mechanism example, not evidence of chemical
interpretability or proof that PLC cannot represent the same function.

**Transfer the direct conditional readout, not the redundant single-key MHA.**
Retain fine-grained local54 evidence unless a controlled ablation shows it is
unnecessary. A pooled path and an atom path can be complementary.

## Actual trained PLC scaling

The best checkpoint was read on CPU without changing any weights. For each
co-attention block, protein-from-ligand residual coefficient is
`0.25 * 0.2 * tanh(rezero_p2l)`. The reverse coefficient is
`0.25 * rezero_l2p`. These are stored model scalars, not hypothetical defaults.

| Block | Protein update coefficient | Ligand update coefficient |
|---|---:|---:|
| 0 | 0.01225 | 0.01771 |
| 1 | 0.01393 | 0.02807 |
| 2 | 0.01334 | 0.02482 |
| 3 | 0.01459 | 0.01701 |

The protein update starts at 0.0125 in this config and remains near that scale.
Local-delta logit scale is `tanh(0.47212)`, approximately 0.440, while fractional
protein gradient 0.6 affects only the local-delta backward path, not all trunk
gradients. `plc_checkpoint_scalars.json` records the raw values.

These observations justify a conservative scale-control probe, but **do not mean
the ligand contributes only 1-3%**. Learned projection magnitudes, normalization,
multiple layers, and the local branch all matter. First measure, on strict VAL,
`norm(coefficient * update) / norm(input)` on valid residues, plus base-versus-
final AUPRC and swap sensitivity. A small scalar alone is not a dead branch.

## Supervision differences worth testing

LiBRe's optional contrastive code groups post-product residue embeddings by
binding/nonbinding label. It is not a ligand-pocket retrieval objective, nor
same-protein distinct-pocket InfoNCE. Both binding-binding and nonbinding-
nonbinding pairs count as same-class positives. The implementation forms a dense
all-residue similarity matrix, which should not be copied at PLC batch lengths.

PLC's near class receives weight 7 versus contact weight 9, with focal gamma 2,
ordinal smoothing 0.05, and contact Tversky. These may be useful for boundary
learning, but introduce an objective different from binary contact ranking.
Their contribution needs a loss-only control, not an assertion that three-class
learning is wrong. In particular:

```text
contact_log_odds = logit_contact - logsumexp(logit_near, logit_far)
sigmoid(contact_log_odds) = softmax(logits)[contact]
```

A small binary-contact BCE term on this quantity can test direct ranking
supervision without changing the inference output shape or threshold protocol.

CNC also deserves a controlled audit. After warmup it downweights a far-labelled
residue to 0.1 when predicted contact+near probability exceeds 0.85, capped at
30% of each sample. This can protect unannotated binding regions, but it can also
forgive confident predictions in the wrong pocket for the current ligand.
Measure its actual activity and error locations before attributing the gap to it.

For sampled residue contrast, start with a small weight, balance contact/far
examples within proteins, and exclude near-boundary/ambiguous residues from that
auxiliary objective. Capping at, e.g., 32 contact and 32 far residues per pair
avoids the full all-residue quadratic matrix. No synthetic ligand is assumed to
be a known nonbinder. No validation/test labels enter feature construction.

## Reproduction discrepancies: do not copy defaults blindly

| Item | Paper-described recipe | Pinned public training code |
|---|---|---|
| Classification loss | Weighted BCE | `BCEWithLogitsLoss(reduction="none")`, unweighted mean |
| Contrastive loss | Included | `--use_contrastive` is opt-in; README example omits it |
| Optimizer | AdamW, decay 1e-5, cosine | Adam at 1e-4; no scheduler in `src/train.py` |
| Train/eval batch | 32 / 64 | 16 / 8 |
| Dropout | 0.1 | Model constructor default 0.2 |
| Internal validation | Threshold chosen by CV | Training script uses a random 20% row split |

The released state dict cannot reveal whether contrastive learning, weighting or
a different unpublished recipe produced it. Thus the paper ablation is evidence
for a hypothesis, not provenance of our downloaded checkpoint.

The code also masks padding out of losses/metrics but not out of its LSTM or
ligand-to-residue attention. Our reproduction preserves fixed-length behavior;
a fresh method should use proper sequence lengths and attention masks. Do not
silently patch a released model and still call it unchanged LiBRe.

Finally, the released training population has not been audited against our full
COACH/HOLO cohort. Our same-input comparison is useful, but it is not a matched-
training or leakage-controlled architectural contest. The paper's own numbers
must not be compared directly with our all-chain per-ligand sample means.

## Follow-up: actual training signals and loss semantics

The continuation on 2026-10-10 read the existing FFFF/GFGF TensorBoard events on
HPC2. It did not rerun inference or change a training run. The full audit is
`tmp/libre_architecture_review/literature/plc_tb_transfer_audit.json`; the remote
copy is in the literature directory above. Event-file and config SHA256 hashes
are embedded. Runner: `scripts/diagnostics/audit_libre_transfer_logs.py`.
Duplicate scalar entries are collapsed by step and latest wall time, with their
maximum value disagreement retained. No conflicting duplicate was found.

### Checkpoint selection is not an immediate escape route

| Existing strict-VAL run | Best TB F1 | Best global AUPRC | Epoch for both, 1-based | CNC peak fraction | CNC last-five-epoch mean |
|---|---:|---:|---:|---:|---:|
| FFFF seed44 | 0.529855 | 0.544503 | 33 | 1.7769% | 0.7322% |
| GFGF seed44 | 0.535996 | 0.546287 | 32 | 1.7808% | 0.7179% |

Raw TB steps are zero-based: FFFF 32 and GFGF 31. Each run contains 35 validation
epochs. F1 and AUPRC peak at the same step in both runs, so changing between
these two checkpoint-selection metrics does not identify a new epoch to test.
This is not a claim that all historical runs share this property, or that these
early-stopped runs exhausted all possible training gains.

### CNC is active, but not broadly suppressing all negatives

The fractions above are over all valid TRAIN residues, not the fraction of false
positives or contact residues. They are per-epoch fractions; the last-five number
is an unweighted average over epochs. Clipping begins at zero-based epoch 5 and
then declines. It is too strong to call CNC the proven cause of the benchmark
gap. However, it selectively affects hard/confident far-labelled predictions,
which can matter even when they are a small fraction of the whole protein.

Code inspection confirms that CNC affects both focal loss and the Tversky FP
term, not just focal. A synthetic CPU check executes the actual function ASTs
from `compute_cnc_weights` and `compute_combined_loss_v4`, without importing the
trainer or modifying it. In a four-residue example, a far-labelled residue with
probabilities `(contact=.90, near=.05, far=.05)` receives weight 0.1. Its Tversky
FP contribution falls from 0.90 to 0.09, and its contact-logit gradient in the
combined loss falls from 0.189779 to 0.019052. Warmup matches CNC-off exactly.
These are algebra checks, not trained-model gains or evidence that the selected
residue belongs to a different real pocket. The toy covers the combined
focal/Tversky function, not the separate distance-regression auxiliary branch.
Artifacts:
`tmp/probe_libre_loss_semantics.py` and `literature/loss_semantics_probe.json`.

For per-ligand localization, "binds another ligand" does not automatically make
the current pair's far label erroneous. Conversely, incomplete experimental
annotations can make some negatives genuinely uncertain. This tension warrants
a CNC-off control, not a blanket claim that clipping is wrong.

### Small gates do not mean inactive ligand branches

At the last logged GFGF training step, ligand-encoder and cross-attention gradient
norms are 0.12666 and 0.05954. The mean protein-from-ligand gate is 0.05439 before
the outer 0.25 multiplier, while the corresponding raw delta norm is 54.69.
The raw local-delta absolute logit mean is 5.6875. These paths are not simply
disconnected. The zero gradient of `local_delta_qk_log_scale` is expected when
Q/K normalization is disabled; do not diagnose it as a training failure.

Crucially, the co-attention norm cache averages padded positions, is before the
outer gate, and then averages over blocks. Multiplying the two logged means does
not recover the actual per-residue update/input ratio. Local-delta logit
magnitude can include a class-common shift and is not contact-ranking evidence.
Thus keep the scale-increase proposal conditional on a valid-only, per-block
measurement rather than immediately doubling every gate.

### LiBRe's contrast is positive-class mass, not canonical SupCon

Its released `nt_xent_loss` computes, excluding self-pairs:

```text
loss_i = logsumexp(all other similarities)
       - logsumexp(same-label similarities)
```

This differs from averaging negative log probabilities over every same-label
positive. A toy set of ten orthogonal embeddings, two contact and eight
noncontact, yields released loss 0.640496 versus 2.197224 for the latter
mean-positive-log formulation. Contact-anchor and noncontact-anchor losses are
2.197225 and 0.251315 under the released formulation. These different baselines
arise from class population, not meaningful ligand recognition in the toy.

Therefore name the initial transfer arm `plc_residue_classmass` if borrowing this
objective. Balanced, bounded sampling and stable logsumexp are deliberate PLC
adaptations, not an exact LiBRe training reproduction. A canonical SupCon arm
would be a different experiment. Neither class-based objective is proof of
same-protein ligand assignment; retain real/shuffle and fixed-budget ranking
guardrails.

Verification: 27 local tests passed across the TB summary, matched ranking and
benchmark-analysis suites; both CPU loss probes and Python compilation passed.
The planned readout's extra 82,304 parameters were independently counted on the
proposed PyTorch projection/head modules, not on an implemented full-model arm.

## Recommended controlled trials, not yet submitted

Use the same strict split, TRAIN/VAL label alignment, seeds, preprocessing and
checkpoint-selection policy. Change one item per arm before trying combinations.
Train first-round controls from scratch rather than resuming the selected
benchmark checkpoint; keep training budget and validation stopping rules matched.
The following names are proposals, not existing jobs.

| Priority | Proposed arm | Isolated question |
|---|---|---|
| P0 | `plc_product_readout` plus `plc_additive_readout_control` | Append projected residue x ligand-local-read channels to the existing head; compare with same-width, same-parameter additive channels. Does explicit interaction improve ranking? |
| P0 diagnostic | `plc_cnc_off` | Disable only CNC. Do strict per-ligand negatives improve ranking without losing coverage? Its observed activity is small, so a large gain is not presumed. |
| P1 | `plc_residue_classmass` | Keep architecture fixed and add sampled post-fusion contact/far positive-mass contrast. Does class separation improve AUPRC without losing ligand specificity? |
| Conditional | `plc_coattn_scale_control` | Only after valid-only update/input ratios, raise cross residual scale from 0.25 toward 0.5. Existing nonzero deltas/gradients do not justify treating the branch as dead. |
| Control | `libre_strict_scratch` | Train a clean masked LiBRe-style small model on our strict data, not the publisher split. How much advantage survives matched data/labels? |

For the product arm, use valid-mask ligand reads and normalization before the
product. Keep current local54 and co-attention paths. A parameter-matched
additive control is necessary: adding dimensions alone could explain a gain.
It is an inductive-bias test, not a claim that the current MLP lacks expressivity.
The exact 64D design and staged run order are in
`../refine-logs/libre_transfer_20261010/EXPERIMENT_PLAN.md`.

A dedicated ESM sequence path is a second-stage alternative: apply lightweight
local convolution and sequence context before geometry fusion or as a parallel
readout. This tests preservation of a clean ESM signal, not just adding another
BiMamba to already-fused states. Existing GDN and virtual tokens mean this is not
an obviously missing capability, so do not prioritize it over the small controls.

Keep a binary-loss-only/CNC-off diagnostic separate from the contrastive arm if
budget allows. Do not change contrastive weight, class weights, decay, attention
scales and dataset filtering in the same first probe.

Evaluate strict VAL AUPRC, fixed-budget ranking, sample/protein-balanced F1,
prediction count and candidate coverage before fixing gates. Retain real/shuffle
dependency checks. A larger real-zero gap is not itself success. Benchmark after
selection, without tuning on HOLO/COACH or dropping their small-contact cases.
Reusing these benchmarks for many architecture choices is itself adaptive test
exposure; a final paper should reserve an additional untouched evaluation set.

## Code anchors

- LiBRe protein path: pinned `src/models/libre.py:48`, `:71`, `:88`.
- LiBRe pooling and product: pinned `src/models/libre.py:24`, `:150`, `:159`.
- LiBRe contrastive objective: pinned `src/utils.py:298`; training at `:315`.
- PLC early gated fusion: `src/models/pocket_gvp_fusion_v11.py:1200`.
- PLC co-attention then base classifier: same file `:1331`, `:1369`.
- PLC base plus delta: `src/models/pocket_gvp_fusion_v11m4.py:234`.
- Existing FiLM: `src/models/pocket_gvp_fusion_v11m8.py:288`.
- Group compatibility and readout: `src/models/pocket_gvp_fusion_v11m10.py:305`, `:433`.
- Co-attention scalar formula: `src/models/pocket_gvp_fusion_v12m2.py:849`.
- Focal/Tversky and CNC: `scripts/train/train_v11m3.py:618`,
  `src/losses/confident_negative_clip.py:35`.

Bottom line: stronger features do not guarantee stronger ranking. The actionable
hypotheses are a more direct conditional classifier, better-shaped residue
supervision, and effective use of existing interaction paths. None is proven to
be the cause of LiBRe's advantage until a matched controlled experiment succeeds.
