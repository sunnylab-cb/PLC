NMA 数据增强实现方案分析

  1. 核心设计决策

  问题 1：预计算 vs 动态计算？

  方案 A：预计算所有 NMA 变体
  - ✅ 优点：训练时零开销，直接加载
  - ✅ 优点：可以预先验证质量（RMSD、clash 检测）
  - ❌ 缺点：存储开销大（19037 个蛋白 × 10 个变体 × 坐标）
  - ❌ 缺点：灵活性差，无法动态调整振幅

  方案 B：预计算 NMA 模式，训练时动态扰动
  - ✅ 优点：存储小（只存 top-K 模式的特征向量）
  - ✅ 优点：灵活性高（可动态调整振幅、随机组合模式）
  - ✅ 优点：可以根据 cluster_size 动态决定是否增强
  - ❌ 缺点：训练时有轻微计算开销（但很小，只是向量加法）

  推荐：方案 B

  ---
  问题 2：如何与 Cluster Weighting 结合？

  策略：Cluster-Size Adaptive Augmentation
  # 小簇 → 高增强概率
  if cluster_size <= 5:
      augmentation_prob = 0.9  # 90% 概率使用 NMA
      n_modes = 10             # 使用更多模式
  elif cluster_size <= 50:
      augmentation_prob = 0.5
      n_modes = 5
  else:
      augmentation_prob = 0.1  # 大簇很少增强
      n_modes = 3

  这样可以：
  - 孤儿蛋白（cluster_size=1）几乎每次都增强
  - 大簇（cluster_size>1000）基本不增强
  - 与 loss weighting 形成双重机制

  ---
  2. 技术实现细节

  NMA 计算流程（使用 ProDy）
  # 伪代码
  import prody as pd

  # 1. 加载蛋白结构
  protein = pd.parsePDB(pdb_file)
  calphas = protein.select('calpha')

  # 2. 计算 ANM（Anisotropic Network Model）
  anm = pd.ANM('protein')
  anm.buildHessian(calphas)
  anm.calcModes(n_modes=20)  # 计算前 20 个模式

  # 3. 保存模式（特征向量）
  modes = anm.getEigvecs()  # [N_ca, 3, n_modes]
  eigenvalues = anm.getEigvals()  # [n_modes]

  # 4. 训练时动态扰动
  for mode_idx in selected_modes:
      amplitude = random.uniform(-2.0, 2.0)  # Å
      displacement = modes[:, :, mode_idx] * amplitude
      perturbed_coords = original_coords + displacement

  ---
  存储方案
  data/nma_modes/
  ├── 1a0q_nma.npz
  │   ├── eigenvectors: [N_ca, 3, 20]  # 前 20 个模式
  │   ├── eigenvalues: [20]
  │   ├── ca_indices: [N_ca]           # Cα 原子索引
  │   └── residue_mapping: [N_ca]      # 残基编号映射
  ├── 1a1b_nma.npz
  └── ...

  每个文件约 10-50 KB，总存储 ~500 MB（远小于预计算完整坐标）

  ---
  3. 数据流设计

  训练时的数据流
  1. Dataset.__getitem__(idx)
     ↓
  2. 加载原始特征（ESM, GVP, coords, labels）
     ↓
  3. 查询 cluster_size → 决定是否增强
     ↓
  4. 如果增强：
     - 加载 NMA modes
     - 随机选择 1-3 个低频模式
     - 随机采样振幅（-2Å 到 +2Å）
     - 扰动 Cα 坐标
     - 重新计算 GVP features（基于新坐标）
     ↓
  5. 返回增强后的样本（labels 不变）

  关键点：标签一致性
  - NMA 只扰动坐标，不改变残基序列
  - 标签（Contact/Interface/Non-binding）保持不变
  - 距离计算基于新坐标（配体位置不变）

  ---
  4. 物理合理性控制

  振幅控制
  # 低频模式 → 大振幅（全局运动）
  # 高频模式 → 小振幅（局部振动）

  amplitude = base_amplitude / sqrt(eigenvalue)

  # 例如：
  # Mode 1 (eigenvalue=0.01): amplitude = 2.0 / sqrt(0.01) = 20 Å (太大，需要限制)
  # Mode 10 (eigenvalue=1.0): amplitude = 2.0 / sqrt(1.0) = 2.0 Å (合理)

  # 实际使用：
  amplitude = clip(base_amplitude / sqrt(eigenvalue), min=0.5, max=3.0)

  质量检查
  - RMSD < 3.0 Å（相对原结构）
  - 无原子碰撞（clash detection）
  - 保持二级结构完整性

  ---
  5. 实现优先级

  Phase 1：最小可行方案（MVP）
  1. 创建 scripts/precompute_nma.py
    - 对所有 19037 个蛋白计算 NMA
    - 保存前 20 个模式到 data/nma_modes/
    - 只针对 cluster_size <= 50 的样本
  2. 修改 PocketDatasetV6
    - 添加 use_nma_augmentation 参数
    - 在 __getitem__ 中根据 cluster_size 决定是否增强
    - 简单策略：随机选 1 个模式，振幅 ±1.5 Å
  3. 验证效果
    - 可视化增强前后的结构
    - 检查 RMSD 分布
    - 训练小模型测试是否有帮助

  Phase 2：优化方案
  1. Cluster-adaptive augmentation probability
  2. 多模式组合（同时使用 2-3 个模式）
  3. 动态振幅调整（基于训练进度）
  4. 与 Cluster Weighting 联合调优

  ---
  6. 潜在问题与解决方案

  问题 1：ProDy 计算慢
  - 解决：预计算所有模式（一次性成本）
  - 备选：使用 C++ 加速版本（如 MMTK）

  问题 2：配体位置如何处理？
  - 方案 A：配体不动，只扰动蛋白（简单）
  - 方案 B：配体随最近的蛋白原子一起移动（复杂但更真实）
  - 推荐：先用方案 A

  问题 3：GVP features 需要重新计算
  - GVP 依赖坐标（边、向量）
  - 必须在增强后重新计算 GVP graph
  - 开销：每个样本 ~10ms（可接受）

  问题 4：ESM embeddings 是否需要更新？
  - 不需要！序列未变，ESM 保持不变
  - 只需更新 GVP 和坐标相关特征

  ---
  7. 预期效果

  定量指标
  - 小簇蛋白（cluster_size < 10）的测试 F1 提升 5-10%
  - 整体泛化性能提升 2-3%
  - 训练时间增加 <10%（因为只对小簇增强）

  定性效果
  - 模型学会对构象变化的鲁棒性
  - 减少对特定晶体结构的过拟合
  - 提高对 flexible loops 的预测能力

  ---
  我的推荐方案

  最佳实践：两阶段实现

  Stage 1：预计算 NMA 模式（离线）
  python scripts/precompute_nma.py \
      --pdb_root /home/husrcf/Code/ProtBind/New \
      --split_file configs/splits/mmseqs_splits_strict.csv \
      --output_dir data/nma_modes \
      --n_modes 20 \
      --cluster_size_threshold 50  # 只计算小簇

  Stage 2：训练时动态增强（在线）
  - 修改 PocketDatasetV6.__getitem__
  - 根据 cluster_size 自适应增强
  - 轻量级：只需向量加法，无需重新运行 ProDy

  配置参数
  data:
    use_nma_augmentation: true
    nma_modes_dir: "data/nma_modes"
    nma_augmentation_prob:
      small_cluster: 0.9   # cluster_size <= 5
      medium_cluster: 0.5  # 5 < cluster_size <= 50
      large_cluster: 0.1   # cluster_size > 50
    nma_amplitude_range: [-1.5, 1.5]  # Å
    nma_n_modes_sample: [1, 3]  # 随机选 1-3 个模式

  这个方案平衡了效果、效率和实现复杂度。你觉得这个设计如何？有什么需要调整的吗？