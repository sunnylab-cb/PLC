# PocketGVPFusion V10m2 架构分析与改进方案

## 当前架构图

```
输入层 (独立坐标系)
├─ 蛋白质分支 (蛋白质局部坐标系)
│  ├─ ESM Embeddings [B, L, 1280]
│  ├─ PhysChem Features [B, L, 62]
│  ├─ Spatial Scalar [B, L, 89]
│  ├─ Spatial Vectors [B, L, 8, 3]
│  ├─ Runtime GVP Vectors [B, L, 4, 3]
│  └─ Protein Coords [B, L, 3]
│
└─ 配体分支 (配体局部坐标系)
   ├─ Atom Features [N_atoms, 109]
   ├─ Ligand Pos [N_atoms, 3]
   ├─ Edge Index [2, E]
   └─ Edge Attr [E, 6]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

蛋白质编码流程
┌────────────────────────────────────────────────────────────────────────┐
│ Module 1: PhysChem MLP Encoder                                         │
│   PhysChem [B,L,62] → MLP → [B,L,128]                                  │
│   + Modality Dropout (p=0.2)                                           │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────┐
│ Module 2: GVP Encoder (3D 局部几何编码)                                │
│   Input: Spatial Scalar [89D] + Vectors [12×3D] + Coords [3D]         │
│   KNN Graph: k=20, radius=10Å (蛋白质局部坐标系)                       │
│   GVP Layers: 3层 → h_s_gvp [B,L,256], h_v_gvp [B,L,20,3]             │
│   ⚠️ 问题: 整条蛋白质序列 [L] 进入后续 Transformer → O(L²) 复杂度      │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
                            b_gvp = h_s_gvp (Block 0)
                                    ↓
┌────────────────────────────────────────────────────────────────────────┐
│ Module 3: Pairwise Bridge + Gating Fusion                             │
│   h_v_gvp → Gram Matrix → Upper Triangle → [B,L,210]                  │
│   h_geo = [h_s_gvp, v_inv, physchem] → [B,L,594]                      │
│   ESM [1280] → Proj → [256]  (+ Modality Dropout p=0.1)               │
│   GVP [594] → Proj → [256]                                             │
│   Gate = σ(MLP([h_gvp_proj, h_esm_proj]))                             │
│   h_seq = [Gate·h_gvp_proj, (1-Gate)·h_esm_proj] → [B,L,512]         │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
                            b0 = h_seq (Block 1)
                                    ↓
┌────────────────────────────────────────────────────────────────────────┐
│ Module 4: RoPE Protein Context Encoder                                │
│   ⚠️ 核心问题: 整条蛋白质 [B, L, 512] 进入 Self-Attention              │
│   4层 RoPE Transformer:                                                │
│     - RoPE: 相对位置编码 (基于1D序列位置)                              │
│     - Gated Multi-Head Attention                                       │
│     - SwiGLU FFN                                                       │
│     - Stochastic Depth (线性递增)                                      │
│   复杂度: O(L²) per layer × 4 layers                                   │
│   问题分析:                                                            │
│     1. L 可达 500-1000 残基 → 注意力矩阵 [L×L] 巨大                   │
│     2. 结合位点通常只占 5-15% 残基 → 85-95% 是背景噪声                │
│     3. RoPE 基于1D序列位置，但蛋白质折叠后 3D 空间才是关键             │
│     4. 序列远端残基(N端/C端)在3D空间可能紧邻，共同构成口袋             │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
                        b1 = h_prot_global (Block 2)
                                    ↓
配体编码流程
┌────────────────────────────────────────────────────────────────────────┐
│ Module 5: Ligand 3D Encoder                                            │
│   Input: Atom Features [N_atoms, 109] + Pos [N_atoms, 3]              │
│   Radius Graph: r=5Å, max_neighbors=32 (配体局部坐标系)                │
│   3层 GVP → h_ligand_nodes [N_atoms, 512]                             │
│   Batch化 → [B, Max_M, 512]                                            │
│   + Global Token [B, 1, 512]                                           │
│   → h_ligand_with_global [B, Max_M+1, 512]                            │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
                        b_lig0 = h_ligand_with_global (Ligand Block 0)
                                    ↓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

交叉注意力模块 (4层循环)
┌────────────────────────────────────────────────────────────────────────┐
│ Module 6: Bidirectional Cross-Attention with AttnRes                  │
│                                                                        │
│ For i in [0, 1, 2, 3]:                                                │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │ 蛋白侧 Inter-Block AttnRes (V10m2: 3 blocks)                 │   │
│   │   h_prot = AttnRes([b_gvp, b0, b1], partial=h_prot)          │   │
│   │   Softmax attention over 4 sources (3 blocks + partial)      │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                           ↓                                            │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │ 配体侧 Inter-Block AttnRes (V10m2: 1 block)                  │   │
│   │   h_lig = AttnRes([b_lig0], partial=h_lig)                   │   │
│   │   Softmax attention over 2 sources (1 block + partial)       │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                           ↓                                            │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │ Protein → Ligand Cross-Attention                             │   │
│   │   Q: h_prot [B,L,512]  KV: h_lig [B,M+1,512]                 │   │
│   │   Attention [B,8,L,M+1] → h_prot_updated                     │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                           ↓                                            │
│   ┌──────────────────────────────────────────────────────────────┐   │
│   │ Ligand → Protein Cross-Attention                             │   │
│   │   Q: h_lig [B,M+1,512]  KV: h_prot_updated [B,L,512]         │   │
│   │   Attention [B,8,M+1,L] → h_lig_updated                      │   │
│   └──────────────────────────────────────────────────────────────┘   │
│                           ↓                                            │
│   h_prot = h_prot_updated                                             │
│   h_lig = h_lig_updated                                               │
└────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌────────────────────────────────────────────────────────────────────────┐
│ Module 7: Classifier                                                   │
│   x_final = [h_prot_fused, physchem_encoded] → [B,L,640]              │
│   MLP: 640 → 256 → 3 (背景/边界/核心)                                  │
│   Output: logits [B, L, 3]                                             │
└────────────────────────────────────────────────────────────────────────┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## 关键验证：坐标系独立性 ✅

**蛋白质坐标系:**
- `protein_coords [B, L, 3]` 用于 GVP KNN 图构建 (k=20, r=10Å)
- 仅在蛋白质内部计算相对距离和方向向量
- 不与配体坐标混合

**配体坐标系:**
- `ligand_pos [N_atoms, 3]` 用于配体 Radius 图构建 (r=5Å)
- 仅在配体内部计算相对距离
- 不与蛋白质坐标混合

**交叉注意力:**
- 仅基于特征空间 [512D] 的点积相似度
- 无显式 3D 空间距离计算
- ✅ 符合要求：两个坐标系完全独立

