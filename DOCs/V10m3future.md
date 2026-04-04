好的，架构图已完成。现在直接在这里分析改进方案。

  核心问题分析

  当前架构的致命缺陷：

  Module 4 的 Transformer 将整条蛋白质序列 [B, L, 512] 输入 Self-Attention：

  1. 计算复杂度爆炸: L=500-1000 → O(L²) = 250K-1M 注意力矩阵元素
  2. 注意力稀释: 结合位点仅占 5-15% 残基，85-95% 是无关背景
  3. 1D 序列偏见: RoPE 基于序列位置，但蛋白质折叠后，序列远端（N端/C端）在 3D 空间可能紧邻共同构成口袋
  4. 全局均匀处理: 所有残基平等对待，无法聚焦潜在结合区域

  ---
  改进方案 A：3D 空间聚类 + 局部-全局 Transformer（推荐）

  核心思路

  不按 1D 序列切分，而是基于 3D 空间距离 将蛋白质划分为多个局部 patch，先在 patch 内做局部 Self-Attention，再通过全局 token
  聚合。

  具体实现

  Step 1: 3D Spatial Clustering (在 GVP 之后)
  # 在 Module 2 GVP 输出后，基于 protein_coords 做 3D 聚类
  # 方法1: KNN-based Patching
  def create_3d_patches(coords, features, k_patch=64, overlap=16):
      """
      coords: [B, L, 3]
      features: [B, L, D]
      返回: patches [B, N_patches, patch_size, D], patch_masks
      """
      # 使用 FPS (Farthest Point Sampling) 选择 patch 中心
      # 每个 patch 包含距离中心最近的 k_patch 个残基
      # overlap 允许边界残基属于多个 patch

  Step 2: Local Self-Attention (patch 内)
  # 替换当前的 4层全局 Transformer
  class LocalGlobalTransformer(nn.Module):
      def __init__(self, d_model, n_local_layers=2, n_global_layers=2):
          # Local: 在每个 patch [patch_size, D] 内做 Self-Attention
          self.local_transformer = nn.ModuleList([...])

          # Global: patch 级别的 Self-Attention
          # 每个 patch 用一个 learnable token 代表 [N_patches, D]
          self.global_transformer = nn.ModuleList([...])

  优势:
  - 局部复杂度: O(k_patch²) × N_patches，k_patch=64 → 4K per patch
  - 全局复杂度: O(N_patches²)，N_patches ≈ L/k_patch ≈ 10-20
  - 总复杂度: O(L·k_patch + N_patches²) << O(L²)
  - 3D 空间邻近的残基在同一 patch，符合口袋的空间连续性

  ---
  改进方案 B：3D Distance-Biased Attention（轻量级）

  核心思路

  保持全局 Transformer，但在 Attention 计算时加入 3D 空间距离偏置，强制模型关注空间邻近区域。

  具体实现

  class SpatialBiasedAttention(nn.Module):
      def __init__(self, d_model, n_heads, max_dist=20.0):
          super().__init__()
          self.attn = RoPEGatedMultiHeadAttention(...)
          # 距离偏置: 将 3D 距离映射为 attention bias
          self.dist_bias = nn.Sequential(
              nn.Linear(1, 64), nn.ReLU(),
              nn.Linear(64, n_heads)  # 每个 head 独立的 bias
          )
          self.max_dist = max_dist
      def forward(self, x, coords, mask=None):
          # coords: [B, L, 3]
          # 计算 pairwise 3D 距离矩阵
          dist = torch.cdist(coords, coords)  # [B, L, L]

          # 距离归一化 + 映射为 bias
          dist_norm = (dist / self.max_dist).clamp(0, 1).unsqueeze(-1)  # [B,L,L,1]
          bias = self.dist_bias(dist_norm)  # [B, L, L, n_heads]
          bias = bias.permute(0, 3, 1, 2)  # [B, n_heads, L, L]

          # 标准 Attention + 距离 bias
          # attn_scores = Q@K^T / sqrt(d) + bias
          # 距离近 → bias 大 → attention 权重高

  优势:
  - 实现简单，仅修改 Attention 计算
  - 保持 O(L²) 复杂度，但通过 bias 引导模型关注空间邻近区域
  - 距离远的残基 attention 权重被抑制

  劣势:
  - 仍需计算完整 [L×L] 距离矩阵和 attention 矩阵
  - 复杂度未降低，仅改变注意力分布

  ---
  改进方案 C：配体引导的动态 Masking（任务导向）

  核心思路

  利用配体位置信息，动态计算每个蛋白质残基与配体的最近距离，仅对 配体附近的残基 做精细 Self-Attention。

  具体实现

  def compute_ligand_proximity_mask(protein_coords, ligand_pos, ligand_batch, threshold=15.0):
      """
      protein_coords: [B, L, 3]
      ligand_pos: [N_atoms, 3]
      ligand_batch: [N_atoms] 指示每个原子属于哪个 batch
      threshold: 距离阈值 (Å)
   
      返回: mask [B, L] — True 表示该残基在配体 threshold Å 范围内
      """
      B, L = protein_coords.shape[:2]
      mask = torch.zeros(B, L, dtype=torch.bool, device=protein_coords.device)

      for b in range(B):
          prot_b = protein_coords[b]  # [L, 3]
          lig_b = ligand_pos[ligand_batch == b]  # [M, 3]

          # 计算每个残基到所有配体原子的最小距离
          dist = torch.cdist(prot_b, lig_b)  # [L, M]
          min_dist = dist.min(dim=1)[0]  # [L]
          mask[b] = min_dist < threshold

      return mask

  # 在 Transformer 中应用
  class LigandGuidedTransformer(nn.Module):
      def forward(self, h_seq, protein_coords, ligand_pos, ligand_batch):
          # 计算配体邻近 mask
          proximity_mask = compute_ligand_proximity_mask(
              protein_coords, ligand_pos, ligand_batch, threshold=15.0
          )

          # 仅对邻近残基做 Self-Attention
          # 远离配体的残基跳过或用简化处理

  优势:
  - 任务导向：直接聚焦结合位点附近区域
  - 大幅减少有效序列长度：L_eff ≈ 0.2L (假设 20% 残基在 15Å 内)
  - 复杂度: O(L_eff²) ≈ O(0.04L²)

  劣势:
  - 需要配体坐标，推理时必须提供配体
  - 可能遗漏远端但通过变构影响结合的残基

  ---
  推荐方案优先级

  1. 方案 A (3D Spatial Clustering) — 最优，复杂度降低 + 符合 3D 几何
    - 实现成本：中等（需要重构 Transformer 模块）
    - 预期收益：+2-4% IoU，训练速度提升 3-5×
  2. 方案 C (Ligand-Guided Masking) — 次优，任务导向
    - 实现成本：低（仅修改 forward 逻辑）
    - 预期收益：+1-3% IoU，训练速度提升 2-3×
    - 限制：推理时需要配体
  3. 方案 B (Distance-Biased Attention) — 保守，复杂度未降低
    - 实现成本：低
    - 预期收益：+0.5-1.5% IoU，无速度提升