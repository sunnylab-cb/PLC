# PLC Future Improvements Roadmap

Generated: 2026-04-11. Updated: 2026-04-13.
Based on code review of v10m4-v10m6 + web search of 2022–2026 literature.
All paper URLs verified. Implementation sketches are for reference, not production-ready code.

---

## Priority Overview

| Priority | Improvement | Effort | Expected Gain | Status |
|----------|------------|--------|---------------|--------|
| ✅ Done | Dynamic ligand global token (mean pool) | Low | High | **V10m5 done** |
| ✅ Done | Batched GVP graph construction | Medium | High (2-4x speed) | **V10m6 done** |
| ✅ Done | Vectorized boundary loss | Low | Medium (eval speed) | **V10m6 done** |
| ✅ Done | Gradient checkpointing bug fix | Low | Correctness | **V10m5 done** |
| ✅ Done | Loss smart gating (weight=0 skip) | Low | Minor speed | **V10m5 done** |
| 🔴 P1 | Per-layer distance bias (vs shared) | Low | High | Partially done (shared only) |
| 🔴 P1 | ESM-2 LoRA fine-tuning | Medium | High | Not implemented |
| 🔴 P1 | Dynamic ligand global token upgrade: PMA | Low | Medium | mean pool done, PMA not yet |
| 🟡 P2 | Distance regression auxiliary head | Low | Medium-High | Not implemented |
| 🟡 P2 | Multi-task binding (MPBind-style) | High | High | Not implemented |
| 🟡 P2 | Contrastive residue representation (MucLiPred) | Medium | Medium-High | Not implemented |
| 🟡 P2 | Multi-scale GVP (dual-radius graph) | High | High | Not implemented |
| 🟢 P3 | SASA auxiliary loss | Low | Medium | Not implemented |
| 🟢 P3 | R-Drop regularization | Very Low | Medium | Not implemented |
| 🟢 P3 | Test-time augmentation (TTA) | Low | Medium | Not implemented |
| 🟢 P3 | Compound loss (Tversky + HausdorffDT) | Medium | Medium | Not implemented |
| 📌 Ref | LABind / GPSite / MPBind as SOTA upper bound | — | — | Benchmark only |

---

## P1-A: Dynamic Ligand Global Token

### Problem
`pocket_gvp_fusion_v10m4.py:574`
```python
self.ligand_global_token = nn.Parameter(torch.randn(1, 1, c['d_model']))
```
This is a **fixed learned constant** — identical for every ligand regardless of molecular structure. All ligands receive the same global context prefix in cross-attention, eliminating ligand-specific conditioning at the global level.

### Evidence
- **GTransCYPs** (J. Cheminformatics, 2024): Attention pooling for molecular graph global readout outperforms mean pooling and fixed tokens for CYP450 inhibitor prediction.
  - URL: https://jcheminf.biomedcentral.com/articles/10.1186/s13321-024-00915-z
- **Physical Pooling Functions in GNNs** (Computers & Chemical Engineering, 2023): Systematic ablation shows attention-based readout (PMA) > mean pool > fixed parameter for molecular property tasks.
  - URL: https://www.sciencedirect.com/science/article/abs/pii/S0098135423000716
- **Self-Attention Graph Pooling** (ICML 2019): Introduced hierarchical attention pooling for graph classification; widely adopted in molecular GNNs.
  - URL: https://arxiv.org/pdf/1904.08082
- **LABind** (Nature Communications, 2025): Uses MolFormer to generate ligand-specific global representations; cross-attention with explicit ligand identity significantly improves cross-ligand generalization.
  - URL: https://www.nature.com/articles/s41467-025-62899-0

### Implementation Sketch
Replace the fixed `nn.Parameter` with **Pooling by Multihead Attention (PMA)**:

```python
# In __init__: replace self.ligand_global_token = nn.Parameter(...)
self.ligand_pool_query = nn.Parameter(torch.randn(1, 1, c['d_model']))
self.ligand_pool_attn  = nn.MultiheadAttention(c['d_model'], c['n_heads'],
                                                batch_first=True,
                                                dropout=c['dropout'])
self.ligand_pool_norm  = nn.LayerNorm(c['d_model'])

# In forward: replace static global_token expand with:
# h_ligand: [B, max_atoms, d_model] (already computed from GNN)
# ligand_mask: [B, max_atoms]
def _pool_ligand_global(self, h_ligand, ligand_mask):
    B = h_ligand.size(0)
    q = self.ligand_pool_query.expand(B, -1, -1)          # [B, 1, d]
    key_pad = ~ligand_mask                                  # True = padding
    pooled, _ = self.ligand_pool_attn(q, h_ligand, h_ligand,
                                       key_padding_mask=key_pad)
    return self.ligand_pool_norm(pooled)                   # [B, 1, d]
```

**Fallback (simpler, still much better than fixed parameter):**
```python
# Masked mean pool — zero cost, zero new parameters
ligand_mask_f = ligand_mask.float().unsqueeze(-1)           # [B, max_atoms, 1]
global_token = (h_ligand * ligand_mask_f).sum(1, keepdim=True) \
               / ligand_mask_f.sum(1, keepdim=True).clamp(min=1)  # [B, 1, d]
```

### Notes
- PMA adds ~2M parameters (one MHA layer). Mean pool adds zero.
- Recommend starting with mean pool to establish baseline, then try PMA.
- This change requires `h_ligand` to be computed *before* prepending the global token, which is already the case in the current forward pass.

---

## P1-B: Per-Layer Distance Bias

### Problem
`pocket_gvp_fusion_v10m4.py:348–367` (RoPEProteinContextEncoder):
```python
# Distance bias computed ONCE and shared across all Transformer layers
attn_bias = self.dist_bias_module(coords, mask=mask)  # computed once
for layer in self.layers:
    x = layer(x, mask=mask, attn_bias=attn_bias, ...)  # same bias every layer
```
AlphaFold2 (Evoformer) and ESMFold both use **per-layer** pair biases. Shallow layers learn local geometry; deep layers learn global context. A shared bias cannot express this hierarchy.

### Evidence
- **AlphaFold2 Evoformer** (Nature, 2021): Each attention layer has its own `pair_bias` projection from the pair representation. Ablations show per-layer bias > shared bias for structure-sensitive tasks.
  - Illustrated breakdown: https://elanapearl.github.io/blog/2024/the-illustrated-alphafold/
- **"Transformers trained on proteins can learn to attend to Euclidean distance"** (arXiv 2502.01533, 2025): Empirically shows that different Transformer layers specialize in different distance ranges. Per-layer biases are necessary to capture this.
  - URL: https://ar5iv.labs.arxiv.org/html/2502.01533
- **ESMFold structure module** (Science, 2022): Uses per-layer pair representation updates to inject 3D geometry at each attention step.
  - Mechanism analysis: https://folding.baulab.info/

### Implementation Sketch
Change `DistanceBias` from a single module to a `ModuleList` inside `RoPEProteinContextEncoder`:

```python
# In RoPEProteinContextEncoder.__init__:
# Replace: self.dist_bias_module = DistanceBias(...)
# With:
if use_dist_bias:
    self.dist_bias_modules = nn.ModuleList([
        DistanceBias(n_heads, n_rbf=dist_bias_n_rbf, max_dist=dist_bias_max_dist)
        for _ in range(n_layers)
    ])
else:
    self.dist_bias_modules = None

# In RoPEProteinContextEncoder.forward:
# Replace shared bias loop with per-layer:
for i, layer in enumerate(self.layers):
    attn_bias = None
    if self.dist_bias_modules is not None and coords is not None:
        attn_bias = self.dist_bias_modules[i](coords, mask=mask)
    x = layer(x, mask=mask, attn_bias=attn_bias, position_ids=position_ids)
```

### Notes
- Extra parameter count: `n_layers × n_rbf × n_heads` = `4 × 16 × 8` = **512 parameters** — negligible.
- Current `DistanceBias` already has zero-init on `proj`, so this is safe to add without training instability.

---

## P2-A: Distance Regression Auxiliary Head

### Problem
Current supervision is a hard 3-class label (Contact/Near/Far). Residues at 4.9Å and 5.1Å receive completely different gradients despite nearly identical binding relevance. A continuous distance regression auxiliary task provides smoother supervision.

### Evidence
- **"Protein-Ligand Binding Representation Learning from Fine-grained Interactions"** (ICLR 2024): Introduces Atom-level Distance Prediction (ADP) as auxiliary task. Combined with masked ligand reconstruction, achieves significant improvement on DUD-E, LIT-PCBA, and CASF benchmarks.
  - URL: https://proceedings.iclr.cc/paper_files/paper/2024/file/2c28efa5a86dca4b603a36c08f49f240-Paper-Conference.pdf
- **"Normalized Protein-Ligand Distance Likelihood Score for Blind Docking"** (J. Chem. Inf. Model., 2024): Distance-based supervision outperforms binary contact labels for localization quality.
  - URL: https://pubs.acs.org/doi/10.1021/acs.jcim.4c01014
- **"Multi-task Bioassay Pre-training for Protein-Ligand Binding Affinity"** (Briefings in Bioinformatics, 2024): Multi-task learning with distance regression improves binding affinity prediction.
  - URL: https://academic.oup.com/bib/article/25/1/bbad451/7469349
- **"Using Autoregressive-Transformer for Protein-Ligand Binding Site Prediction"** (bioRxiv, 2025): Distance-based auxiliary supervision consistently improves residue-level binding site localization.
  - URL: https://www.biorxiv.org/content/10.1101/2025.03.11.642700v1.full

### Implementation Sketch
```python
# In __init__: add alongside the existing classifier
self.dist_regression_head = nn.Sequential(
    nn.Linear(c['d_model'], 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.ReLU()          # distance >= 0
)

# In forward: add after h_prot_fused is computed
pred_dist = self.dist_regression_head(h_prot_fused).squeeze(-1)  # [B, L]

# In training loss (train_v10m4.py):
# true_dist: [B, L] from residue_ligand_distances_dcenter.pt (already loaded)
# Weight nearby residues more heavily
def weighted_dist_loss(pred, true, mask, cutoff=10.0):
    w = torch.exp(-true.clamp(max=cutoff) / cutoff)  # near residues have higher weight
    loss = F.huber_loss(pred, true, delta=2.0, reduction='none')
    return (loss * w * mask.float()).sum() / mask.float().sum().clamp(min=1)

loss_dist = weighted_dist_loss(pred_dist, true_dist, protein_mask)
total_loss = main_loss + 0.1 * loss_dist
```

### Notes
- Labels already exist in `data/residue_ligand_distances_dcenter.pt` — no new data needed.
- Use Huber loss (delta≈2.0Å) rather than MSE to reduce sensitivity to outliers (very distant residues).
- Auxiliary weight 0.1 is a reasonable starting point; search 0.05–0.2.
- Drop the head at inference (no cost).

---

## P2-B: Multi-Scale GVP (Dual-Radius Graph)

### Problem
Current graph: `radius=15Å, k=30`. A single radius cannot simultaneously capture tight local backbone geometry (needs ~6–8Å) and broader pocket context (needs ~18–20Å).

### Evidence
- **GearNet** (ICLR 2023): Multi-relational protein graph using sequential edges + radius edges + KNN edges. Outperforms single-scale GVP by ~3–5% on EC number prediction and fold classification.
  - URL: https://arxiv.org/pdf/2203.06125
  - GitHub: https://github.com/DeepGraphLearning/GearNet
- **MGPPI** (Frontiers in Genetics, 2024): Multi-scale graph for protein-protein interaction prediction; dual-scale consistently outperforms single-scale.
  - URL: https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2024.1440448/full
- **GPSite** (eLife, 2024): Uses hierarchical graph construction; best SOTA results on genome-scale binding site annotation.
  - URL: https://elifesciences.org/articles/93695
- **"Towards Multiscale Graph-based Protein Learning"** (NeurIPS 2025): Formal analysis showing single-radius graphs miss important structural features at different scales.
  - URL: https://www.math.ucdavis.edu/~strohmer/papers/2025/NeurIPS_GNN.pdf

### Implementation Sketch
```python
# In PocketGVPFusionV10m4.__init__: add second GVP encoder
self.gvp_encoder_local = GVPEncoder(   # r=8Å, tight local contacts
    node_in_dim=(c['d_spatial_scalar'], c['n_gvp_vectors']),
    node_h_dim=(c['d_gvp_internal'] // 2, c['n_gvp_vector_hidden'] // 2),
    edge_in_dim=(_edge_s_dim, _edge_v_dim), n_layers=c['n_gvp_layers'],
)
self.gvp_encoder_global = GVPEncoder(  # r=20Å, broader context
    node_in_dim=(c['d_spatial_scalar'], c['n_gvp_vectors']),
    node_h_dim=(c['d_gvp_internal'] // 2, c['n_gvp_vector_hidden'] // 2),
    edge_in_dim=(_edge_s_dim, _edge_v_dim), n_layers=c['n_gvp_layers'],
)
self.gvp_fusion = nn.Linear(c['d_gvp_internal'], c['d_gvp_internal'])

# In forward: build two graphs, run both encoders, fuse
edge_local  = build_protein_graph(..., k=16, radius=8.0,  ...)
edge_global = build_protein_graph(..., k=30, radius=20.0, ...)
h_s_local,  h_v_local  = self.gvp_encoder_local(...)
h_s_global, h_v_global = self.gvp_encoder_global(...)
h_s_gvp = self.gvp_fusion(torch.cat([h_s_local, h_s_global], dim=-1))
```

### Notes
- **High implementation effort** — requires pipeline changes (two graph builds per sample, per batch item).
- GearNet's ablation suggests the local (short-range) encoder contributes most to binding site tasks.
- Can be introduced as a config flag `use_multiscale_gvp: false` to maintain backward compat.

---

## P3-A: SASA Auxiliary Loss

### Problem
The model already receives SASA as input (part of the 89D `spatial_scalar_features`), but doesn't explicitly learn to reconstruct it. Adding an auxiliary SASA prediction head encourages the encoder to maintain surface-exposure information through the representation layers.

### Evidence
- **GeoNet** (Structure / Cell, 2024): Model spontaneously learned RSA-like representations; explicit SASA supervision was shown to be an effective inductive bias for binding site encoders.
  - URL: https://www.sciencedirect.com/science/article/abs/pii/S0969212624004465
- **E-pRSA** (J. Mol. Biol., 2024): Embedding-based RSA prediction; shows RSA is predictable from sequence/structure representations and is useful for downstream binding tasks.
  - URL: https://www.sciencedirect.com/science/article/abs/pii/S0022283624000664
- **"Evaluating Representation Learning on the Protein Structure Universe"** (2024): Auxiliary structure denoising and SASA prediction consistently improve GNN protein representations.
  - URL: https://arxiv.org/html/2406.13864v1

### Implementation Sketch
```python
# In __init__:
self.sasa_aux_head = nn.Sequential(
    nn.Linear(c['d_model'], 64),
    nn.ReLU(),
    nn.Linear(64, 1),
    nn.Sigmoid()    # RSA is normalized to [0, 1]
)

# In forward (after protein_context_encoder):
pred_rsa = self.sasa_aux_head(h_prot_global).squeeze(-1)  # [B, L]

# In training loss:
# true_rsa: extract from protein_spatial_scalar[:, :, sasa_dim_index]
# (need to identify which dimension of 89D is RSA — check feature extraction code)
loss_sasa = F.mse_loss(pred_rsa[protein_mask], true_rsa[protein_mask])
total_loss = main_loss + 0.1 * loss_sasa
```

### Notes
- RSA (Relative Solvent Accessibility) is more stable than absolute SASA — normalize by max SASA per residue type.
- Need to identify the correct SASA dimension in the 89D `spatial_scalar_features` vector by checking `preparation/protein/` feature extraction scripts.
- Expected benefit is moderate since ESM-2 already implicitly encodes some surface exposure information. Most useful for improving boundary residue (Contact/Near interface) precision.

---

## P3-B: R-Drop Regularization

### Problem
Current dropout is applied independently each forward pass, but there's no explicit constraint ensuring the model produces consistent predictions under different dropout masks for the same input.

### Evidence
- **R-Drop: Regularized Dropout for Neural Networks** (NeurIPS 2021, Liang et al.): Two forward passes on the same batch + bidirectional KL divergence loss. Improves over 18 NLP datasets with 0.5–1.5% average gain. The improvement is larger when base dropout is higher.
  - arXiv: https://arxiv.org/abs/2106.14448
  - GitHub: https://github.com/dropreg/R-Drop
- **DropConn** (IEEE TNNLS, 2023): Applies R-Drop-style consistency regularization to molecular GNN edges. Improves molecular property prediction on MoleculeNet benchmarks.
  - URL: https://ieeexplore.ieee.org/document/10164235/

### Implementation Sketch
```python
# In train_epoch (train_v10m4.py), replace single forward with:
def rdrop_train_step(model, inputs, labels, mask, alpha=0.1, **loss_kwargs):
    # Two forward passes with different dropout masks
    out1 = model(**inputs)
    out2 = model(**inputs)
    logits1, logits2 = out1['logits'], out2['logits']

    # Standard task loss (averaged over both passes)
    loss1, _, _ = compute_combined_loss_v4(logits1, labels, mask, **loss_kwargs)
    loss2, _, _ = compute_combined_loss_v4(logits2, labels, mask, **loss_kwargs)
    task_loss = 0.5 * (loss1 + loss2)

    # R-Drop KL consistency loss (only over valid residues)
    p1 = F.softmax(logits1[mask], dim=-1)
    p2 = F.softmax(logits2[mask], dim=-1)
    kl = 0.5 * (F.kl_div(p1.log(), p2, reduction='batchmean') +
                 F.kl_div(p2.log(), p1, reduction='batchmean'))

    return task_loss + alpha * kl
```

### Notes
- Doubles the forward pass cost per batch step (but NOT backward — only one backward on total loss).
- Recommended starting alpha: 0.1 for protein tasks (NLP uses 0.5, but protein tasks have smaller datasets relative to model capacity).
- Most effective when the model already has high dropout rates — less applicable after the regularization reduction in v10m4.1.
- Enable via config flag: `use_rdrop: false`, `rdrop_alpha: 0.1`.

---

## P1-C: ESM-2 LoRA Fine-Tuning

### Problem
PLC uses frozen ESM-2 embeddings (1280d). The pretrained embeddings are general-purpose — they are not tuned for binding site semantics. Fine-tuning the full ESM-2 is prohibitively expensive (650M+ parameters), but LoRA (Low-Rank Adaptation) inserts small trainable matrices into attention layers while keeping 99%+ of weights frozen.

### Evidence
- **ESMBind / QBind** (bioRxiv, 2023): LoRA fine-tuning of ESM-2 (650M) for per-residue binding site prediction. LoRA+QLoRA mitigate overfitting better than full fine-tuning, and larger ESM-2 models can be trained on modest hardware (single GPU).
  - URL: https://www.biorxiv.org/content/10.1101/2023.11.13.566930v1.full
  - HuggingFace blog: https://huggingface.co/blog/AmelieSchreiber/esmbind
- **InstructPLM-mu** (arXiv, 2025): 1-hour fine-tuning of ESM-2 beats ESM-3 on protein mutation predictions. Token-wise concat + LoRA achieves SOTA with minimal compute.
  - URL: https://arxiv.org/html/2510.03370v1
- **SeqProFT** (arXiv, 2024): Systematic evaluation of LoRA fine-tuning for sequence-only protein property prediction. Shows rank-4 or rank-8 LoRA is sufficient for most tasks.
  - URL: https://arxiv.org/html/2411.11530v1/
- **Medium-sized PLMs** (Scientific Reports, 2025): Even medium ESM-2 models (150M) with fine-tuning perform well on realistic datasets; larger is not always necessary.
  - URL: https://www.nature.com/articles/s41598-025-05674-x

### Implementation Sketch
```python
# Replace frozen ESM-2 embeddings with LoRA-adapted ESM-2
from peft import LoraConfig, get_peft_model
from transformers import EsmModel

esm_model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
lora_config = LoraConfig(
    r=8, lora_alpha=16, target_modules=["query", "key", "value"],
    lora_dropout=0.1, bias="none"
)
esm_model = get_peft_model(esm_model, lora_config)
# Trainable params: ~2M out of 650M (0.3%)

# In forward: compute embeddings on-the-fly instead of loading from disk
esm_out = esm_model(input_ids=token_ids, attention_mask=token_mask)
esm_embeddings = esm_out.last_hidden_state  # [B, L, 1280]
```

### Notes
- Requires changing data pipeline: load raw sequences + tokenize instead of pre-computed embeddings.
- Training will be slower (ESM-2 forward pass per batch), but representation quality should improve significantly.
- Start with rank=4 or 8; higher ranks risk overfitting on ~15K training samples.
- Can freeze LoRA for first N warmup epochs, then unfreeze.

---

## P2-C: Multi-Task Binding Prediction (MPBind-style)

### Problem
PLC trains on a single protein-ligand pair at a time. Recent work shows that jointly predicting binding sites for multiple partner types (protein, DNA, RNA, ligand, ion) improves individual-task performance through representation sharing.

### Evidence
- **MPBind** (Bioinformatics, 2025): Multitask protein binding site predictor using PLMs + 4-layer EGNN. Predicts 5 binding partner types simultaneously. Multitask learning improves per-task metrics vs single-task baselines.
  - URL: https://academic.oup.com/bioinformatics/article/41/11/btaf589/8300842
  - GitHub: https://github.com/jianlin-cheng/MPBind
- **GPSite** (eLife, 2024): 10-ligand-type joint training with shared encoder + per-type MLP heads. Simpler architecture than PLC but higher DCC on standard benchmarks.
  - URL: https://elifesciences.org/articles/93695
- **LABind** (Nature Communications, 2025): Ligand-aware binding site prediction with Ankh PLM + MolFormer + cross-attention. 4.7% MCC improvement over GPSite.
  - URL: https://www.nature.com/articles/s41467-025-62899-0

### Implementation Sketch
```python
# Add per-ligand-type classifier heads alongside existing 3-class head
self.classifier_heads = nn.ModuleDict({
    'small_molecule': nn.Linear(c['d_model'], 3),  # Contact/Near/Far
    'protein': nn.Linear(c['d_model'], 3),
    'dna': nn.Linear(c['d_model'], 3),
    'rna': nn.Linear(c['d_model'], 3),
    'ion': nn.Linear(c['d_model'], 3),
})
# Route to correct head based on ligand_type metadata
# Shared encoder learns general binding geometry; heads specialize
```

### Notes
- Requires dataset expansion: current dataset is small-molecule only. Need BioLiP or similar multi-partner dataset.
- High effort due to data pipeline changes, but evidence strongly supports multi-task benefit.
- Can start with ligand-type embedding as a cheaper alternative (encode ligand category as an input feature).

---

## P2-D: Contrastive Residue Representation (MucLiPred-style)

### Problem
The model only receives supervised signal from the 3-class classification loss. There's no explicit constraint that binding residues should cluster together in representation space while non-binding residues stay separate.

### Evidence
- **MucLiPred** (J. Chem. Inf. Model., 2024): Dual contrastive learning for multi-ligand binding residue prediction. Residue-level paradigm differentiates binding vs non-binding; type-level paradigm distinguishes ligand types. Significant improvement over non-contrastive baselines.
  - URL: https://pubs.acs.org/doi/10.1021/acs.jcim.3c01471
- **CLAPE** (PNAS, 2023): Contrastive learning in protein language space for drug-protein interaction. Pre-trained encoder + contrastive fine-tuning achieves SOTA on multiple binding benchmarks.
  - URL: https://www.pnas.org/doi/10.1073/pnas.2220778120
- **Multi-Level Contrastive Learning for Protein-Ligand Binding** (bioRxiv, 2023): Residue-level contrastive loss alongside classification loss improves binding residue F1.
  - URL: https://www.biorxiv.org/content/10.1101/2023.12.06.570503v1

### Implementation Sketch
```python
# Supervised contrastive loss on residue representations
def residue_contrastive_loss(h_prot, labels, mask, temperature=0.1):
    """Pull Contact residues together, push apart from Far residues."""
    h = F.normalize(h_prot[mask], dim=-1)           # [N_valid, d]
    y = labels[mask]                                  # [N_valid]
    sim = torch.mm(h, h.t()) / temperature            # [N_valid, N_valid]
    # Same-class pairs = positives, different-class = negatives
    pos_mask = (y.unsqueeze(0) == y.unsqueeze(1))
    # SupCon loss (Khosla et al. 2020)
    ...
```

### Notes
- Medium effort: requires implementing SupCon loss + tuning temperature.
- Most useful when Contact examples are scarce — contrastive loss amplifies their signal.
- Can be combined with R-Drop for even stronger regularization.

---

## P3-C: Test-Time Augmentation (TTA)

### Problem
At inference, only a single forward pass is used. Given that PLC already has coordinate noise and dropout infrastructure, running multiple stochastic forward passes and averaging predictions can improve robustness.

### Evidence
- **EnsemPPIS** (Research, 2024): Transformer-based ensemble for protein-protein interaction site prediction. Ensemble of diverse models achieves highest performance across benchmarks.
  - URL: https://spj.science.org/doi/10.34133/research.0240
- **AlphaFold2 subsampled MSA** (Nature Communications, 2024): Multiple runs with different MSA subsamples predict conformational distributions with >80% accuracy.
  - URL: https://www.nature.com/articles/s41467-024-46715-9
- **Protein data augmentation benchmark** (arXiv, 2024): Systematic evaluation of augmentation strategies for protein tasks; coordinate perturbation and dropout-based TTA consistently help.
  - URL: https://arxiv.org/html/2403.00875v1

### Implementation Sketch
```python
# At inference time:
def predict_with_tta(model, inputs, n_runs=5, coord_noise=0.1):
    model.eval()
    all_probs = []
    for _ in range(n_runs):
        # Apply small coord noise
        noisy_inputs = {**inputs}
        noisy_inputs['protein_coords'] = inputs['protein_coords'] + \
            torch.randn_like(inputs['protein_coords']) * coord_noise
        with torch.no_grad():
            logits = model(**noisy_inputs)['logits']
            all_probs.append(F.softmax(logits, dim=-1))
    return torch.stack(all_probs).mean(dim=0)  # Average probabilities
```

### Notes
- Zero training cost; only inference-time overhead (n_runs × forward pass).
- Should also enable dropout at test time (`model.train()` for dropout layers only) for MC Dropout.
- Recommended n_runs: 5-10 for good accuracy/speed tradeoff.
- Can provide uncertainty estimates (variance across runs) as bonus.

---

## P3-D: Compound Loss (Tversky + Boundary Distance)

### Problem
Current loss combines Focal + Tversky. Recent work shows that adding boundary-aware distance terms (e.g., Hausdorff distance transform) to region-based losses (Tversky) yields SOTA results in highly imbalanced segmentation.

### Evidence
- **Compound loss survey** (arXiv, 2023): Systematic evaluation of 25+ loss functions. Tversky + HausdorffDT or adaptive TverskyCE achieves best Dice and surface distance across varying class imbalances.
  - URL: https://arxiv.org/html/2312.05391v1
  - GitHub: https://github.com/JunMa11/SegLossOdyssey
- **Unified Focal Loss** (Computerized Medical Imaging and Graphics, 2022): Generalizes Dice and CE losses; makes Focal Tversky a special case. Reduces hyperparameter search space.
  - URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC8785124/
- **SPix-WCE** (Scientific Reports, 2025): Context-aware loss that identifies complicated areas (analogous to binding site boundaries) and focuses training on them.
  - URL: https://www.nature.com/articles/s41598-025-08234-5

### Implementation Sketch
```python
# Hausdorff Distance Transform loss (boundary-aware)
def hausdorff_dt_loss(pred_probs, labels, mask, coords):
    """Penalize predictions that are spatially far from true boundaries."""
    # For Contact class: compute distance transform from true Contact boundary
    contact_true = (labels == 0).float() * mask.float()
    contact_pred = pred_probs[:, :, 0]
    # Use pre-computed boundary mask (already vectorized in V10m6)
    boundary = compute_boundary_mask(labels, coords, mask)
    # Weight loss by distance to nearest boundary
    ...
```

### Notes
- Medium effort: needs careful implementation of distance transform on irregular protein graphs (not regular grids).
- Alternative: use the already-implemented `boundary_loss.py` with higher weight, since it's now vectorized in V10m6.
- Consider Unified Focal Loss as a drop-in replacement for separate Focal + Tversky.

---

## SOTA Reference Models (as Upper Bound Benchmarks)

### LABind (Nature Communications, 2025)
**Full name**: LABind: Identifying Protein Binding Ligand-Aware Sites via Multimodal Transformer
- URL: https://www.nature.com/articles/s41467-025-62899-0
- GitHub: https://github.com/ljquanlab/LABind
**Key techniques PLC lacks**:
- Uses **MolFormer** (molecular language model) to generate ligand-specific global representations
- Protein encoder: **Ankh** (a protein language model) + geometric features
- Cross-attention with explicit ligand type identity encoding
- Multi-ligand training (DNA, RNA, peptide, small molecule, ion) with shared encoder + type-specific heads

### GPSite (eLife, 2024)
**Full name**: GPSite: Genome-Scale Annotation of Protein Binding Sites via Protein Language Model and Geometric Deep Learning
- URL: https://elifesciences.org/articles/93695
- GitHub: https://github.com/biomed-AI/GPSite
**Key techniques PLC lacks**:
- Simultaneously predicts **10 ligand types** (ATP, ADP, AMP, GTP, HEM, NAD, FAD, SAM, DNA, RNA) with shared geometry encoder + per-ligand classifier heads
- Multi-task binding improves individual-task performance through representation sharing
- Simpler architecture than PLC but achieves higher DCC on COACH420/HOLO4K

### GeoNet (Structure / Cell, 2024)
**Full name**: GeoNet: Accurate Prediction of Protein-Ligand Binding Sites through Interpretable Geometric Deep Learning
- URL: https://www.sciencedirect.com/science/article/abs/pii/S0969212624004465
**Key techniques PLC lacks**:
- Local spherical coordinate-invariant representation (stronger than CA-based GVP for pocket geometry)
- Multi-scale neighborhood aggregation
- Explicit interpretability: outputs attention maps aligned with known binding residue chemistry
- Spontaneously learns RSA and hydrophobicity — suggests these are important inductive biases

### MPBind (Bioinformatics, 2025)
**Full name**: MPBind: A Multitask Protein Binding Site Predictor Using Protein Language Models and Equivariant GNNs
- URL: https://academic.oup.com/bioinformatics/article/41/11/btaf589/8300842
- GitHub: https://github.com/jianlin-cheng/MPBind
**Key techniques PLC lacks**:
- 4-layer EGNN (equivariant to rotation/translation) — PLC's GVP is SE(3)-equivariant too but EGNN is simpler
- Multitask prediction for 5 binding partner types (protein, DNA/RNA, ligand, lipid, ion) with shared encoder
- Per-type MLP classifier heads; multitask learning improves individual-task metrics

### ScanNet (Nature Methods, 2022)
**Full name**: ScanNet: An Interpretable Geometric Deep Learning Model for Structure-Based Protein Binding Site Prediction
- URL: https://www.nature.com/articles/s41592-022-01490-7
**Key techniques**:
- Purely structure-based (no ESM) yet competitive — strong baseline for geometric inductive bias
- Uses atomic-level geometry (not residue-level CA only)
- Highly interpretable: each neuron corresponds to a learnable geometric pattern

---

## Bugs / Latent Issues Found in Code Review

These are not improvements but **known bugs** that should be fixed:

### ~~Bug 1: Gradient Checkpointing Drops chain-gap RoPE and Distance Bias~~ ✅ FIXED in V10m5
Fixed via closure wrapping `coords` and `position_ids` into `checkpoint()` call.

### Bug 2: `ligand_to_protein` is always a zero tensor, only used for shape
**Location**: `collate.py:638`, `pocket_gvp_fusion_v10m4.py:810`
```python
# Allocates B × max_atoms × 512 zeros every batch — only B and max_atoms are used
'ligand_to_protein': torch.zeros(B, batch['ligand_features'].shape[1], 512)
```
Replace with reading B and max_atoms directly from `ligand_mask`.

### Bug 3: NMA augmentation is a no-op
`nma_n_confs: 1` in config means only one conformation is ever loaded.
Framework exists but never activates.

### Bug 4: Cluster-weighted loss never activates
`splits.txt` has no 4th column for cluster sizes → `load_splits_with_cluster_size` always falls back → `cluster_sizes=None` → `ClusterWeightedLoss` never instantiated.

---

## Implementation Order Recommendation

**v10m5 ✅ DONE** — low effort, high confidence:
1. ✅ Dynamic ligand global token (mean pool)
2. ✅ Gradient checkpointing bug fix
3. ✅ Loss smart gating
4. ✅ Label threshold aligned to PoC (4/6Å)

**v10m6 ✅ DONE** — speed optimization:
5. ✅ Batched GVP graph construction (2-4x speedup)
6. ✅ Vectorized boundary loss (eval 2-5x speedup)

**v10m7 (next)** — model quality, low-medium effort:
7. Per-layer distance bias → change shared to ModuleList (P1-B, 10 lines)
8. Upgrade ligand global token: mean pool → PMA (P1-C, ~30 lines)
9. Distance regression auxiliary head (P2-A, add head + loss term)

**v10m8** — representation quality, medium effort:
10. ESM-2 LoRA fine-tuning (P1-C, pipeline change needed)
11. Contrastive residue representation (P2-D, SupCon loss)
12. SASA auxiliary loss (P3-A)

**v10m9** — high effort, architectural:
13. Multi-scale GVP (P2-B, dual-radius graph building)
14. Multi-task binding (P2-C, requires dataset expansion)

**Inference-time (no training change):**
15. Test-time augmentation (P3-C, coord noise + MC dropout)
