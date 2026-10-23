# Ligand-conditioned local spatial scanning

Date: 2026-10-24. Status: user approved the G0/S0/S1/T1 scratch pilots.
Implementation and CPU tests are complete; GPU smoke precedes full training.
MoE remains deferred. All dimensions below are starting settings, not measured
optima. Existing experiments and historical plans remain unchanged.

## Problem and hypotheses

Replace repeated whole-protein mixing with shared, ligand-conditioned spatial
operators, while retaining the existing feature information. The task is
three-class residue prediction for a protein-ligand pair, not pose generation.

| Hypothesis | Minimum evidence | Main alternative explanation |
| --- | --- | --- |
| A local decision architecture is sufficient or preferable to repeated global mixing | Strict VAL gains or a predeclared efficiency tradeoff; no interface-contact regression; same-data small-global control | Parameter reduction, different optimization, loss or source composition |
| Ligand-dependent local operators or expert routing add useful conditional specialization | Dynamic beats static local control; learned routing beats the same uniform expert bank; VAL ligand-swap sensitivity is retained | More parameters, ensemble smoothing, molecule-size or data-source shortcuts |

ESM features already contain sequence context. Success would support removing
ADDITIONAL downstream global mixing, not the absence of biological long-range
effects. Cached V4 descriptors also have their original extraction context.

## Data and evaluation contract

- Reuse the expanded joint GLOBAL-SI30 split: TRAIN 139158, VAL 17395, TEST 17395.
- Keep TRAIN-only normalization, existing feature validity masks and fixed
  heavy-atom contact labels at <5 A. Do not change near/far labels or eligibility.
- Keep protein cap 1800 and ligand cap 256 with the existing crop policy during
  this architecture comparison; input-coverage changes require another control.
- Keep every observed chain and all scored residues, including zero-contact
  chains. No GT chain, pocket, ligand position, or contact-mask selection.
- TRAIN and VAL only during development. No TEST/COACH/HOLO-driven choices.
- Report PDBbind and BioLiP separately, plus single/multichain, positive-contact
  size, protein length and cross-chain-contact strata. Source-stratified metrics
  are essential because approximately 90% of this VAL is BioLiP.

The historical full model is V11m13 group_rescompat + GFGF, seed44, VAL-best
epoch25. Its old run stopped on loss-based patience. It is a descriptive anchor,
not a matched F1-stopping control. Before a final claim, rerun the full reference
under the same training/stopping/microbatch policy as the new comparisons.

## Common tensor interface

B = protein-ligand pairs; N = valid/padded residues; M = ligand atoms.
The public feature/collate interface is retained.

| Input | Shape | Use |
| --- | --- | --- |
| ESM2 final features | B,N,1280 | Frozen cache; trainable per-residue projection |
| Protein physicochemistry | B,N,62 | Per-residue scalar projection |
| Protein spatial scalars | B,N,89 | Per-residue scalar projection |
| Protein spatial vectors | B,N,8,3 | Geometry; do not flatten as ordinary invariant scalars |
| Backbone GVP vectors | B,N,4,3 | Combined with spatial vectors, 12 vector channels |
| CA coordinates and residue/chain keys | B,N,3 plus metadata | Protein-only neighborhood construction |
| Ligand atom features | B,M,109 | local54 plus global55; same TRAIN normalization |
| Ligand bonds | sparse edges,6 | Same bond attributes; intraligand distance RBF if enabled |
| Ligand conformer | B,M,3 | Intraligand geometry only; never aligned protein-ligand distances |

No new ESM3, ESMC, MLC, UniMol, artificial glycine linker or feature extraction.

### Protein pointwise stem

```text
ESM2 1280 -- Linear + LN ------------------------> 128
physchem 62 -- Linear + GELU + LN ---------------> 32
spatial scalar 89 -- Linear + GELU + LN ---------> 96
concat ----------------------------------------> h0: B,N,256
12 vector channels -- equivariant vector map ---> V0: B,N,16,3
```

Normalization is per residue/channel, not across all residues of a protein.
Vector norms (used in T1) or Gram features can enter scalar updates; arbitrary Cartesian
components cannot enter invariant scalar classifiers directly. Verify the
coordinate-frame semantics of every existing vector feature before claiming
end-to-end rotation invariance.

### Ligand stem

Use the existing three-layer ligand geometric graph encoder at hidden width
128. Bond features remain 6D with the existing optional 8D intraligand RBF.
Retain a direct local54 feature projection alongside graph context:

```text
109D atoms + bonds + intraligand geometry -> 3 graph layers -> A: B,M,128
local54 -> Linear + LN -----------------------------------> C: B,M,128
concat(A,C) -> Linear(256,128) + LN -----------------------> Z: B,M,128
concat(masked_mean(Z), masked_max(Z)) -> MLP(256,128) ------> gL: B,128
```

Z is a ligand-only atom memory, never updated by pooling the entire protein.
The same ligand encoding is reused over all protein centers. Permuting atoms
and consistently permuting bonds/coordinates must preserve output scores.

### Protein-only sparse neighborhoods

For each valid residue i, select at most 31 nearest OTHER residues within
12 A CA distance; include the center separately, for at most K=32 tokens.
The radius and K are fixed engineering starting values, not values selected
on HOLO. No first-round radius sweep.

Permit spatially adjacent residues from different chains. Edges contain RBF16
distance, a same-chain bit and four within-chain sequence-offset channels
(zero across chains). Directional features use relative protein coordinates.
Do not treat chain ID numbers or concatenation indices as biological distances.
Padding and missing coordinates must be excluded consistently. Count cap hits,
cross-chain edges and isolated centers in diagnostics. Neighbor ties need a
documented deterministic rule; test permutation consistency away from ties.

There is no whole-protein normalization/softmax in the local spatial operator.
This is an implicit shared scan, not an expensive re-encoding of each crop.
With two graph layers, the receptive field can reach two hops (up to roughly
24 A path length), so it is NOT a strict 12 A total receptive field. A strict
patch-isolation implementation would need separate patch-restricted processing.

## Spatial operators

### S0: local static GVP control

Use two GVP message-passing blocks with scalar/vector widths (256,16):

```text
(m_s_ij, m_v_ij) = GVP_message(h_i, h_j, V_i, V_j, edge_ij)
h_i_next = LN(h_i + scalar_update(mean_j(m_s_ij)))
V_i_next = vector_residual_update(V_i, mean_j(m_v_ij))
```

Use vector-norm-preserving normalization and channel-shared vector dropout.
All valid residues share parameters; all centers are predicted. This arm still
uses ligand atoms and gL in the common final readout. It is NOT ligand-blind.

### S1: ligand-conditioned GVP scan

Same graph, two GVP blocks, widths and readout as S0. A separate conditioner in
each layer maps gL to bounded scalar scale/shift and vector-channel scale:

```text
gamma_s, beta_s, gamma_v = MLP_layer(gL)  # 256,256,16 channels
m_s_ij = (1 + 0.2*tanh(gamma_s)) * m_s_ij + 0.2*tanh(beta_s)
m_v_ij = (1 + 0.2*tanh(gamma_v))[...,None] * m_v_ij
```

Apply the shift only to real edges, then masked-mean aggregate. Vector scaling
is identical across xyz components. Initialize conditioner output projections
to zero, so the local operator starts at the static GVP behavior; the shared
readout already supplies ligand-dependent gradients. Output head is not zeroed.
This changes a local representation operator, not a whole-protein logit bias.
Count the additional conditioner parameters; do not call S0/S1 exactly
parameter-matched. If a gain is small, add a static capacity-matched control.

### T1: ligand-conditioned local triangle pair attention

Replace the two GVP spatial blocks with two local pair-attention blocks; do not
stack them on top of S1. Keep the same pointwise stem, ligand stem and readout.
Reduce V0 to invariant norms/Gram features at the pair input. Each layer builds
64D pair states for the same K-neighbor support:

```text
z_ij = PairMLP(h_i, h_j, vector_invariants_i_j, edge_ij)       # 64D
z_ij = bounded_FiLM(z_ij, gL)
a_ijk = softmax_k(Q(z_ij) dot K(z_ik) / sqrt(16)
                 + B(RBF(distance(j,k))))                   # 4 heads
z'_ij = z_ij + Wo(sum_k a_ijk * V(z_ik))
h'_i = LN(h_i + Wpool(masked_mean_j(z'_ij)))                  # 256D
```

The triangle comprises learned center-neighbor states (i,j)/(i,k) and the
known protein neighbor-neighbor edge (j,k). This is local triangle-BIASED pair
attention, not a full AlphaFold pair trunk or hard triangle-inequality solver.
No true ligand pose or protein-ligand distance is used. Protein geometry is
fixed input, not predicted docking geometry.

Cost is O(B*N*K^2) attention, rather than all-protein O(B*N^3) triangle
attention. Process center chunks (initially 128) with activation checkpointing;
chunking alone does not remove saved backward activations. Parameter counts
and measured throughput must accompany S1/T1 comparisons: this also changes
the geometry operator family, not only an attention-bias flag.

## Common atom readout and output head

After the spatial operator, each center reads the same ligand-only memory:

```text
q_i = Linear(256,128)(h_i)
k_a, v_a = Linear(128,128)(Z_a)
w_ia = softmax_over_VALID_ATOMS(q_i dot k_a / sqrt(128))
c_i = sum_a w_ia * v_a                                  # B,N,128
x_i = concat(h_i, c_i, broadcast(gL))                   # B,N,512
logits_i = MLP(512,128,3)(x_i)                          # contact/near/far
p_contact_i = softmax_over_CLASSES(logits_i)[0]
```

Center-chunk the atom readout as needed. Its softmax is over ligand atoms,
NOT all residues or candidate pockets. Do not force one positive pocket.
All inputs use validity masks. Handle absent ligand branches explicitly;
masked means/maxima and all-masked attention must produce finite zero context.

Remove the downstream GFGF stack, four co-attention blocks, whole-protein
virtual tokens and base-plus-late-delta duplicate decision path in local arms.
Do not feed a globally mixed V11 hidden state to this local model and claim
to have tested locality. Reuse feature and loader code, not the full V11
forward. These deliberate deletions mean S1 versus historical V11 is a
strategy comparison; S0/S1 is the cleaner conditioning comparison.

Keep the existing supervised distance-auxiliary objective with a common
readout head in all new local arms. It uses labels only in the loss. Preserve
the three-class primary objective and CNC; do not simultaneously add ranking,
contrastive, distillation, pocket-center or new negative-sampling losses.

## MoE: small compatibility experts, not four full backbones

M0/M1/M2 share the COMPLETE S1 encoder and atom readout. Insert four small
compatibility experts before the SAME classifier:

```text
E_e(x_i): Linear(512,64) -> GELU -> Linear(64,256), e=1..4
route = softmax(MLP(128,64,4)(gL) / 1.0)                  # B,4
h'_i = LN(h_i + 0.2 * sum_e route_e * E_e(x_i))
logits_i = common_classifier(concat(h'_i,c_i,gL))
```

The router receives ligand information only in the first round. Every center
of the same ligand uses the same mixture, but expert outputs depend on the
local protein context and atom readout. These are latent compatibility
experts; do not name them hydrogen-bond/metal/aromatic experts before evidence.
Initialize expert second projections to zero; router learning may initially
lag until expert outputs diverge. Log this behavior rather than mislabeling
initial uniform routing as a failed implementation.

| Arm | Route | Balance |
| --- | --- | --- |
| M0 uniform4 | Fixed 1/4 each, same four experts | None |
| M1 moe4 | Learned dense soft routing | None |
| M2 moe4_bal | Identical learned routing | Weak batch-average penalty |

All four experts execute: this is dense soft-MoE, not a sparse-compute speedup.
No token dropping, distributed expert parallelism or four full geometry trunks.

For M2 define p_bar_e = mean over PAIRS of route_e and
L_balance = 1e-3 * 4 * sum_e (p_bar_e - 1/4)^2. The coefficient is a proposed
starting value, not a literature-established optimum. Do not weight this mean
by protein length, and do not force EACH ligand to use uniform routing.
Small batches and chemical prevalence can favor uneven use; balancing may
hurt legitimate specialization. Compare M1/M2 rather than assume a benefit.

Log expert mass, per-pair entropy, expert-output similarity, contribution
norms and gradients. Analyze routes versus ligand chemistry, atom count and
data source on held-out VAL. Balanced usage is not proof of distinct functions.
Uniform4 controls expert-bank capacity/averaging; a small routing gain still
needs multiple seeds. Routing metrics alone cannot establish chemical causality.

## Run matrix and order

| ID | Variant | Main question | Stage |
| --- | --- | --- | --- |
| R0 | Current full GFGF reference, matched recipe rerun | Mainline comparator | Confirmation required |
| G0 | Full interaction family, d_model=256, same layer counts | Width/capacity control | First pilot |
| S0 | Static local GVP + common ligand readout | Local architecture baseline | First pilot |
| S1 | Ligand-conditioned local GVP | Conditional local operator | First pilot |
| T1 | Local triangle pair attention | Alternative spatial operator | First pilot, after smoke |
| M0 | S1 + uniform4 experts | Capacity/ensemble control | Second stage |
| M1 | S1 + learned moe4 | Ligand-dependent expert mixture | Second stage |
| M2 | S1 + learned moe4 + balance | Balance necessity | Second stage |

G0 changes d_model only; leave layer counts and geometry/ligand input widths
unchanged where compatible. It is a width-reduction control, not exact
parameter matching to the new local family. Do not change depth and width
simultaneously and interpret the result as an isolated width effect.

Pilot seed44; promising variants and their decisive controls then get seeds43
and45. If S1 does not beat/match S0 meaningfully, do not automatically expand
MoE. A triangle-only improvement routes to a matched triangle follow-up, not
uncontrolled stacking with GVP and MoE.

## Training and acceptance

All trainable parts start randomly and are unfrozen; ESM is a frozen feature
cache, not a newly fine-tuned PLM. No resume or old trained trunk. S0/S1 can
share deterministic initialization of common modules for the seed44 pilot.

Use AdamW LR1e-4, weight_decay0.05, five-epoch warmup, 120-epoch cosine horizon,
clip1.0, BF16 and no compile. Reuse the pinned joint loss/augmentation recipe.
Hidden dropout0.15 and classifier dropout0.2 are initial common settings.
Preserve source-dropout semantics across local variants. The old local-delta
gradient scale0.6 does not automatically apply to the new single-path scanner;
new local trainable paths use ordinary gradients, with no invented 0.6 global
multiplier. Report this difference to the historical architecture.

Target common microbatch16/accumulation1 after GPU smoke. If any arm cannot
fit, choose ONE common microbatch/accumulation policy before comparisons,
including a matched R0 rerun. Equal nominal effective batch alone is not
mathematical equivalence for batch-reduced Tversky. Checkpointing is preferred
before reducing the shared microbatch. One Slurm GPU per run if later approved.

Select and early-stop on the same existing best VAL F1 metric, patience30;
save best-F1 and best finite VAL-loss checkpoints only. AUPRC/AUROC are logged,
not separate checkpoint-selection paths. Keep explicitly labeled legacy F1,
sample-macro and residue-pooled metrics rather than switching definitions.

At the selected best-F1 checkpoint report F1/MCC/P/R at0.5 and at that run's
VAL-selected F1 gate, AUPRC/AUROC, Top20 and Top-L/20,L/10,L/5. A fitted VAL
gate is a development measurement, not an independent performance estimate.
After model/protocol freeze, a final TEST gate would remain fixed.

Predeclared pilot screen: roughly +0.003 absolute legacy VAL F1 with no
material ranking/source-specific regression, or near-parity accompanied by
a substantial measured efficiency gain. This is a screening rule, NOT
statistical significance. Confirm with three seeds and paired per-protein or
SI-component uncertainty estimates. Report effect sizes, not just pass/fail.

Use VAL real/shuffle/zero diagnostics for finalists; shuffled ligand is not a
proven nonbinder, and a large zero drop alone is not an optimization objective.
Also audit TRAIN-eval versus VAL using the SAME deterministic evaluation path,
aggregation and fixed TRAIN subset to distinguish capacity effects from
context sensitivity. Do not compare training-time augmented loss to clean
VAL F1 and call that a measured generalization gap.

## Sanity and mechanistic diagnostics

- Verify labels/features/keys stay aligned, all residues are scored, and all
  variants use identical dataset hashes and masks. No nonfinite-score dropping.
- Protein translation/rotation with consistent vector transformations, chain
  permutation, ligand atom permutation and padding changes must be tested.
- Add a spatially separated distractor component while keeping original
  cached features fixed: local geometric operators should preserve original
  outputs outside their receptive field. This is a model intervention, not a
  claim that a physically fused protein necessarily preserves its structure.
- Distinguish fixed-cache context interventions from sequence re-extraction:
  ESM can change after artificial sequence fusion even if the head is local.
- Keep real cross-chain neighbors and inspect interface-contact subsets.
  Nearby-chain deletion is not expected to be invariant.
- Profile 100 TRAIN steps after warmup and one complete VAL pass. Measure
  samples/sec, residues/sec and peak allocated/reserved memory per arm.
- Tiny TRAIN overfit checks, BF16 forward/backward, masks, permutation checks
  and finite gradients precede any full run. Smoke weights are discarded.

## Budget and deferred work

The old full joint model took approximately40 GPU-hours for34 epochs, but this
is NOT an ETA for local models or F1-based stopping. No credible new GPU-hour
estimate exists before smoke/profile. Up to four one-GPU pilots can be queued
after approval; no multi-GPU-per-run allocation is needed. MoE-stage expansion
and three-seed confirmation require a positive decision, not automatic launch.

Defer radius sweeps, sparse top-k experts, global-summary augmentation, TCN,
new PLM caches, artificial linkers, docking-distance features, ligand-induced
pose generation and automatic benchmark selection. A later weak-context arm
can assess whether some VAL strata benefit from remote context.

Do not claim chain independence, an end-to-end local information model, exact
capacity matching, chemical expert identities, guaranteed improvement or
external SI30 separation from this design alone.

## References and local evidence

- [Current full architecture audit](../../docs/libre_architecture_transfer_20261010.md).
- [Historical negative readout probes](../../docs/libre_transfer_results_20261011.md):
  these used an older split and are not numerical comparators for the new joint set.
- [Expanded joint training](../../docs/joint_global_si30_results_20261021.md).
- [Frozen HOLO context diagnostic](../../docs/joint_holo_chain_context_20261023.md):
  motivation only, never a hyperparameter-selection set.
- [GVP, ICLR2021](https://arxiv.org/abs/2009.01411).
- [PeSTo, Nature Communications2023](https://www.nature.com/articles/s41467-023-37701-8).
- [Dynamic Filter Networks, NeurIPS2016](https://arxiv.org/abs/1605.09673).
- [MERA, AAAI2026](https://ojs.aaai.org/index.php/AAAI/article/view/39903):
  retrieval/active-site MoE, not direct evidence for ligand-routed local scanning.
- [HydrAffinity, 2026 preprint](https://www.biorxiv.org/content/10.64898/2026.07.24.740495v1.full):
  affinity prediction, not our residue benchmark.
- [Long-range graph benchmark reassessment](https://arxiv.org/abs/2309.00367).
- [PDZ allosteric maps, Nature Communications2026](https://www.nature.com/articles/s41467-026-71005-x).
- [GS linkers, Biochemistry2017](https://doi.org/10.1021/acs.biochem.7b00902).

## Readiness checklist

- [x] Shared inputs, spatial operators, readout and MoE controls specified.
- [x] Selection, no-test-iteration and threshold-MCC contracts retained.
- [x] Explicit non-claims and historical comparator limitations recorded.
- [ ] Architecture implementation and unit/integration tests.
- [ ] Actual parameter counts and GPU smoke/profile.
- [ ] Frozen run configs, source hashes and deployment approval.
- [ ] Pilot training, then control-matched multi-seed confirmation.
