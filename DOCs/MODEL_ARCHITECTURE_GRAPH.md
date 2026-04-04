# PocketGVPFusion (current): Architecture Graph

Source of truth: `src/models/pocket_gvp_fusion.py` (`PocketGVPFusion`).

```mermaid
flowchart TD
  %% =========================
  %% Inputs
  %% =========================
  subgraph P[Protein Inputs]
    ESM[ESM embeddings<br/>`esm_embeddings`<br/>[B, L, d_esm=1280]]
    PS[Physchem scalars<br/>`protein_scalar`<br/>[B, L, d_phys=35]]
    PV[Vector features<br/>`protein_vectors`<br/>[B, L, n_vec_in=4, 3]]
    PC[Coordinates<br/>`protein_coords`<br/>[B, L, 3]]
    PM[Mask<br/>`protein_mask`<br/>[B, L]]
  end

  subgraph L[Ligand Inputs]
    LX[Atom features<br/>`ligand_x`<br/>[N, d_ligand_atom=64]]
    LEI[Edge index<br/>`ligand_edge_index`<br/>[2, E]]
    LEA[Edge features<br/>`ligand_edge_attr`<br/>[E, edge_dim=6]]
    LB[Batch ids<br/>`ligand_batch`<br/>[N]]
    LM[Mask (padded)<br/>`ligand_mask`<br/>[B, M]]
  end

  %% =========================
  %% Protein geometric encoder
  %% =========================
  KNN[kNN graph build<br/>`build_knn_graph`<br/>k=20, radius=10Å]
  GVP[GVPEncoder<br/>W_s_in/W_v_in + 3× GVPConv<br/>→ LayerNorm]
  HS[h_s_gvp<br/>[B, L, d_gvp_internal=256]]
  HV[h_v_gvp<br/>[B, L, n_vec_hidden=16, 3]]

  PC --> KNN
  PS --> GVP
  PV --> GVP
  KNN --> GVP
  GVP --> HS
  GVP --> HV

  %% =========================
  %% Bridge + early injection
  %% =========================
  PB[PairwiseBridge<br/>Gram upper-tri<br/>16→136 dims]
  VINV[v_inv<br/>[B, L, 136]]
  HGEO[h_geo = concat<br/>(h_s_gvp, v_inv, protein_scalar)<br/>[B, L, 427]]
  GPROJ[gvp_proj<br/>Linear 427→256<br/>+ LayerNorm + Dropout]

  EDROP[Modality dropout (train only)<br/>drops ESM with p=esm_dropout_prob]
  EPROJ[esm_proj<br/>Linear 1280→256<br/>+ LayerNorm + Dropout]

  HSEQ[h_seq = concat<br/>(h_gvp_proj, h_esm_proj)<br/>[B, L, d_model=512]]

  HV --> PB --> VINV --> HGEO --> GPROJ
  PS --> HGEO
  ESM --> EDROP --> EPROJ
  GPROJ --> HSEQ
  EPROJ --> HSEQ

  %% =========================
  %% Protein context encoder
  %% =========================
  PCE[ProteinContextEncoder<br/>PosEnc + 4× GatedTransformerBlock<br/>→ LayerNorm]
  HPROT[h_prot_global<br/>[B, L, 512]]

  HSEQ --> PCE --> HPROT
  PM --> PCE

  %% =========================
  %% Ligand encoder + padding
  %% =========================
  LENC[LigandEncoder<br/>node/edge proj → 3× GINEConv (residual)<br/>→ out_proj+norm]
  LNODE[h_ligand_nodes<br/>[N, 512]]
  LSCAT[Scatter to padded tensor<br/>naive per-batch loop]
  LPAD[h_ligand (padded)<br/>[B, M, 512]]

  LX --> LENC
  LEI --> LENC
  LEA --> LENC
  LB --> LENC
  LENC --> LNODE --> LSCAT --> LPAD

  %% =========================
  %% Cross-attention fusion
  %% =========================
  XATTN[CrossAttentionBlock × n_cross_attention_layers<br/>(protein queries ← ligand kv)]
  HFUSED[h_fused<br/>[B, L, 512]]

  HPROT --> XATTN --> HFUSED
  LPAD --> XATTN
  PM --> XATTN
  LM --> XATTN

  %% =========================
  %% Late injection + classifier
  %% =========================
  XFIN[x_final = concat(h_fused, protein_scalar)<br/>[B, L, 547]]
  CLS[Classifier<br/>Linear 547→256 → ReLU → Dropout → Linear 256→3]
  LOGITS[logits<br/>[B, L, n_classes=3]]

  HFUSED --> XFIN --> CLS --> LOGITS
  PS --> XFIN
```

Notes:
- This graph matches the code defaults (`d_esm=1280`, `d_phys=35`, `d_model=512`). See `PocketGVPFusion.DEFAULT_CONFIG` in `src/models/pocket_gvp_fusion.py`.
- `DOCs/MODEL_ARCHITECTURE.md` contains older dimension values (e.g., ESM-3/1536D); use this file for the current implementation.

