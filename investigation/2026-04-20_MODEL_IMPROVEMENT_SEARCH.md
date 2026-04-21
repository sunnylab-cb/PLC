# PLC Model Improvement Search

Date: 2026-04-20
Scope: `research-pipeline` Stage 1 only, focused on finding compatible upgrades for the current PLC mainline.

## Current Baseline in Repo Reality

- Current training entry is `scripts/train/train_v10m8.py`, which instantiates `PocketGVPFusionV10m8` rather than `v10m4` or `v10m6`.
- Current config is `configs/pocket_gvp_v10.yaml`.
- Current model already includes:
  - dynamic ligand global token via masked mean pooling, not a fixed learned token
  - protein-only distance-biased self-attention
  - chain-gap RoPE / chain features
  - distance regression auxiliary head
  - confident negative clipping
- Code refs:
  - `scripts/train/train_v10m8.py:216-217`
  - `configs/pocket_gvp_v10.yaml:3`
  - `src/models/pocket_gvp_fusion_v10m8.py:543-554`
  - `src/models/pocket_gvp_fusion_v10m8.py:573-575`
  - `src/models/pocket_gvp_fusion_v10m8.py:858-862`
  - `src/models/pocket_gvp_fusion_v10m8.py:616-628`
  - `src/models/pocket_gvp_fusion_v10m8.py:923-925`

## Hard Constraint

- The main model must remain in a separate-coordinate regime.
- Allowed:
  - protein-only coordinates
  - ligand-only coordinates
  - latent-space interaction
  - matching / compatibility objectives
- Not allowed:
  - protein-ligand shared-frame geometry
  - explicit cross-distance features from aligned complexes
  - joint complex graphs that assume known pose at inference

## Important Repo Observations

- The older summary file is partially stale. Repo reality now includes chain features and the distance auxiliary head.
- There is still a label-threshold inconsistency that should be treated as a blocker before reading too much into future ablations:
  - `configs/pocket_gvp_v10.yaml:6-8` says PoC-aligned `4.0/6.0`
  - `configs/pocket_gvp_v10.yaml:29-31` actually sets `label_contact_thresh: 5.0`, `label_near_thresh: 8.0`
  - `src/data/pocket_dataset_v8.py:549-562` uses those runtime thresholds directly
- If this inconsistency is unresolved, architecture comparisons may be confounded by supervision drift.

## Ranked Upgrade Directions

### P0. Add a ligand-conditioned contrastive matching objective

Why:
- The current model is still mainly trained as residue-wise 3-class classification plus distance regression.
- It has cross-attention, but no explicit global objective saying "this ligand should match this protein / this candidate pocket better than negatives".
- This is the cleanest way to strengthen ligand specificity without violating the separate-coordinate rule.

Why it fits the constraint:
- The objective can be defined purely in latent space.
- It does not require protein-ligand aligned coordinates at inference.

Evidence:
- ConGLUDe (arXiv, submitted 2026-01-14, revised 2026-02-03) aligns ligands with whole-protein and candidate-site embeddings through contrastive learning and supports ligand-conditioned pocket prediction without predefined pockets.
- ProFSA (ICLR 2024) shows pocket-side contrastive pretraining aligned to pseudo-ligands can materially improve downstream pocket tasks.

Minimal implementation sketch:
- Reuse the current ligand global token as the ligand-side anchor.
- Add one protein global token and one pocket candidate token set.
- Build an InfoNCE loss with in-batch negatives:
  - positive: `(protein_i, ligand_i)`
  - negatives: `(protein_i, ligand_j)` for `j != i`
- Optional second head:
  - contrast ligand against top-k residue cluster summaries from the same protein
- Start with loss weight `0.05-0.2`.

Why this is my top recommendation:
- It is the most aligned with your deployment reality.
- It targets the main missing supervision level: compatibility, not just residue labels.

### P1. Upgrade the ligand encoder with a pretrained molecular language model branch

Why:
- The current ligand branch is still dominated by `109D` atom features + local 3D graph encoding.
- The global ligand token is now dynamic, but it is still only a masked mean over current ligand node states.
- That likely underuses large-scale chemistry priors for unseen ligands.

Evidence:
- LABind (Nature Communications, published 2025-08-19) uses MolFormer for ligand representation and explicitly reports improved generalization to unseen ligands in a ligand-aware binding-site setting.
- LABind also uses cross-attention rather than pose-conditioned cross-geometry, so the design pattern is compatible with PLC.

Minimal implementation sketch:
- Precompute one ligand LM embedding per ligand SMILES, e.g. MolFormer-style pooled vector.
- Project it to `d_model`.
- Inject it in one of three cheap ways:
  - replace the current mean-pooled ligand global token
  - concatenate with the current mean-pooled token, then project
  - use it as a conditioning vector for the classifier head
- Keep the existing 3D ligand GNN. Do not replace it initially; fuse it.

### P1. Strengthen the protein side with pretrained structural supervision

Why:
- PLC already has ESM2 and a stronger geometric branch than earlier versions, but it still lacks a stronger protein-only structural prior tailored to binding-site prediction.
- This is especially attractive because it respects the "no shared protein-ligand coordinates" rule.

Evidence:
- AF2BIND (Nature Methods, published 2026-03-11) shows AlphaFold2 pair representations outperform ESM2 and ESM1-IF features for small-molecule binding-site prediction.
- GPSite (eLife, 2024) shows that combining protein language model features with a geometry-aware residue graph and richer geometric featurization improves binding-site prediction.
- ProFSA (ICLR 2024) indicates pocket-focused contrastive pretraining is useful even before downstream supervised fine-tuning.

Lowest-risk implementation path:
- Do teacher distillation before raw feature injection.
- Precompute AF2BIND-like residue pocketness scores or a protein-only binding prior.
- Add an auxiliary KL / regression head to distill this teacher into PLC's protein branch.

Higher-upside but costlier path:
- Precompute pooled AF2 pair features around each residue and add them as an extra protein feature channel.

### P2. Add a multi-task objective for ligand-family or pocketness prediction

Why:
- Current supervision is still dominated by a single residue label space.
- The model has no explicit auxiliary task for "general pocketness" or ligand-family discrimination.
- Multi-task learning can reduce overfitting to one exact label geometry.

Evidence:
- GPSite uses a multi-task framework across multiple binding partner types and reports stronger binding-site prediction.
- AF2BIND demonstrates that protein-only pocketness signals are strong enough to be useful even without ligand input.

Implementation sketch:
- Add one auxiliary head for general pocketness.
- Optionally add a ligand-family head if ligands can be binned by chemistry or source annotation.
- Start with shared trunk + small MLP heads and low auxiliary weights.

### P2. Move from single-scale to multi-scale protein geometry

Why:
- The current protein graph is still a single-radius single-neighbor-budget construction from the GVP branch.
- Binding pockets often require both tight local geometry and broader cavity context.

Evidence:
- GPSite explicitly models residual, relational, and global context.
- Repo-local `DOCs/future_improvements.md` already identifies dual-radius / multi-scale geometry as a plausible next step.

Implementation sketch:
- Keep the current `15A / 30NN` graph as the global branch.
- Add a local branch, e.g. `8A / 16NN`.
- Fuse the two scalar outputs before entering the Transformer.

Why this is not higher:
- It is a real architectural change with higher implementation and debugging cost.
- I would not do this before the contrastive objective and pretrained-encoder upgrades.

## What I Would Not Prioritize

- Do not move toward Docking-style aligned complex models as the PLC mainline.
- That includes explicit protein-ligand cross-distance features, shared-frame triangle updates, or joint complex graphs at inference.
- BindNet-style fine-grained complex interaction learning is useful as inspiration, but its core training signal is much closer to the regime you explicitly want to avoid.

## Recommended Execution Order

1. First fix the label-threshold inconsistency and confirm the actual benchmark-aligned supervision.
2. Add a global ligand-conditioned contrastive loss on top of the current V10m8 model.
3. Add a pretrained ligand LM vector branch.
4. Add a protein-side teacher distillation branch using AF2BIND-like pocketness priors.
5. Only then test multi-task or multi-scale geometry extensions.

## Fastest High-ROI Ablation Set

If you want the shortest path to a meaningful result, I would start with exactly these three experiments:

1. `V10m8 + contrastive protein-ligand matching loss`
2. `V10m8 + MolFormer-style ligand global embedding`
3. `V10m8 + contrastive loss + ligand LM embedding`

This gives one objective upgrade, one representation upgrade, and one combined model.

## Sources

- Repo docs:
  - `DOCs/future_improvements.md`
  - `DOCs/PLC_POC_FAILURE_ANALYSIS_REPORT_ZH.md`
  - `DOCs/MODEL_ARCHITECTURE.md`
- Repo code:
  - `configs/pocket_gvp_v10.yaml`
  - `scripts/train/train_v10m8.py`
  - `src/models/pocket_gvp_fusion_v10m8.py`
  - `src/data/pocket_dataset_v8.py`
- Primary external sources:
  - LABind, Nature Communications, 2025-08-19: https://www.nature.com/articles/s41467-025-62899-0
  - GPSite, eLife, 2024: https://elifesciences.org/articles/93695
  - ProFSA, ICLR 2024: https://proceedings.iclr.cc/paper_files/paper/2024/file/6cc31b44d88dce8380d36e81485cd07f-Abstract-Conference.html
  - ConGLUDe, arXiv, submitted 2026-01-14: https://arxiv.org/abs/2601.09693
  - AF2BIND, Nature Methods, 2026-03-11: https://www.nature.com/articles/s41592-026-03011-2

## Bottom Line

- The current PLC model is no longer missing the old easy wins like dynamic ligand global token or distance auxiliary loss; those are already present in V10m8.
- The best next gain is not "more shared geometry". It is stronger ligand-conditioned matching in latent space plus stronger pretrained priors on both protein and ligand sides.
