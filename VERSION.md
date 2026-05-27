# Pocket-GVP-Fusion 版本说明

本文档记录了Pocket-GVP-Fusion模型的各个版本及其主要变化。

---

## 总体简介（如何阅读/如何维护）

- **目的**：用"面向维护"的方式记录每次版本迭代的**动机**、**改动点**、**影响范围**与**复现实例**，方便回溯与对齐训练/评测脚本。
- **顺序**：按时间从新到旧（最新版本放最上面）。
- **最小必填**（每个版本建议都包含）：`主要特性`、`关键改进/变化`、`文件结构`、`配置变化`、`使用方法`。
- **以代码为准**：如文档与实现不一致，请在对应版本条目里标注"以哪些文件为准"，并给出路径（例如：`configs/...`、`src/models/...`、`scripts/...`）。
- **关联文档**：模型结构图见 `MODEL_ARCHITECTURE_GRAPH.md`（如需更新，请同步更新对应版本条目中的"架构改进"描述）。

## 总体结论（版本演进主线）

- 迭代主线通常围绕三件事：**标签/物理定义更准确**（例如距离阈值）、**特征更丰富**（例如配体/蛋白输入特征）、**模态融合更强**（例如交叉注意力从单向到双向）。
- 维护建议：每次升级尽量明确回答三问：**为什么改**（问题/瓶颈）、**改了什么**（模块/超参/数据）、**怎么验证**（复现实验命令+预期变化）。

## 维护模板（复制即可）

> 新版本建议直接复制以下模板，填完后放到本文档最上方（在当前最新版本之前）。

````markdown
## V? - YYYY-MM-DD

### 主要特性
- ✅ ...

### 关键改进/变化
1. ...

### 影响范围（改了哪些东西）
- 配置：`configs/...`
- 代码：`src/...`
- 脚本：`scripts/...`
- 数据/标签：`data/...`

### 文件结构
```
configs/
  └── ...
src/
  └── ...
scripts/
  └── ...
```

### 配置变化
```yaml
model:
  # ...
data:
  # ...
```

### 使用方法（最小可复现）
```bash
# 训练
python scripts/train_?.py --config configs/?.yaml --amp

# 评测
python scripts/evaluate.py --config configs/?.yaml --ckpt outputs/.../best.pt
```

### 兼容性/迁移说明（可选）
- 与 V? checkpoint 是否兼容：是/否（如否，给出转换脚本/原因）
- 需要重新生成哪些中间产物：...

### 已知问题（可选）
- ...

---
````

---

## V11m6 - 2026-05-27

### 主要特性
- ✅ `V11m6`: V11m4t stable trunk + local54/graph evidence delta-only branch
- ✅ 不再创建旧 `pair_delta_*` 模块，也不把 pair evidence 注入 cross-attention logits
- ✅ local54 同时进入 delta branch 的 key/value，直接形成 residue-level `delta_logits`
- ✅ global55 + UniMol CLS 继续只做 delta amplitude calibration gate
- ✅ 默认 `local_delta_detach_protein: true`，保护 V11 protein-side decision floor

### 关键改进/变化
1. **动机**：依赖评估显示 `V11m5t-balanced` 入口更干净、UniMol 有小贡献，但 `Δshuffle` 没提升；主线仍应围绕 local54/ligand graph → residue decision，而不是强 pair-bias 接管 trunk。
2. **结构约束**：`logits = base_logits + scale * global_gate(global55, UniMol) * local_delta_logits`；delta branch 不修改 protein/ligand trunk，只做 late correction。
3. **信息路径**：local54 atom chemistry 直接参与 local delta attention 的 key/value；ligand hidden state 仍提供 graph/self-attn 后的 atom memory。
4. **稳定性**：delta head final projection zero-init，初始 logits 与 base path 等价；GPU0 dummy FWD/BWD smoke 通过，finite logits/grad。

### 影响范围
- 模型：`src/models/pocket_gvp_fusion_v11m6.py`
- 训练：`scripts/train/train_v11m6.py`
- 配置：`configs/pocket_gvp_v11m6_hpc.yaml`
- Slurm：`scripts/slurm/run_v11m6_local_delta.slurm`
- 推理：`scripts/infer/infer_v11_internal_ligdep.py`

### 使用方法
```bash
sbatch scripts/slurm/run_v11m6_local_delta.slurm

# 或指定配置
sbatch --export=ALL,CONFIG_FILE=configs/pocket_gvp_v11m6_hpc.yaml \
  scripts/slurm/run_v11m6_local_delta.slurm

# 对照矩阵：no_gate / local54_only / lig_hidden_only / no_detach_protein
sbatch scripts/slurm/run_v11m6_ablation_matrix.slurm
```

### 兼容性/迁移说明
- 默认从头训练，便于和 V11m4t/V11m5t-balanced 公平比较。
- 可选 `training.init_from_checkpoint` 做 `strict=False` 初始化；新增 `local_delta_*` / `global_delta_gate.*` 允许 missing。

---

## V11m5t-balanced - 2026-05-26

### 主要特性
- ✅ 归档旧 V11m5 pair-bias 前移路线：100 epoch 趋势未证明优于 V11m4/V11m4t
- ✅ 当前 `V11m5t` 改为 `V11m4t + group-wise balanced ligand input adapters`
- ✅ 新增 `V11m4tp` probe-only 对照：参数/logits 与 V11m4t 相同，只记录旧入口 contribution RMS
- ✅ local54/global55/UniMol 分组 `LayerNorm -> Linear -> GELU -> LayerNorm`，再 concat 给 Ligand3DEncoder
- ✅ 新增入口平衡诊断：`scripts/diagnostics/check_v11_input_balance.py`
- ✅ TB 新增 ligand group RMS / group scale / adapter grad 监控

### 关键改进/变化
1. **动机**：维度压缩只能避免 foundation embedding 天然维度压制，但不能证明入口贡献平衡；真正需要看 `RMS(W_local x_local)`、`RMS(W_global x_global)`、`RMS(W_unimol x_unimol)`。
2. **平衡策略**：local54、global55、UniMol CLS 各自先过 adapter，初始化时 projected RMS 在同一数量级；group scale 是可学习参数，不是固定同权约束。
3. **保留稳定主线**：继承 V11m4t 的 late local pair-delta 与 global55/UniMol delta gate，不再使用 P→L pair-bias attention injection。
4. **诊断覆盖 protein 端**：probe 同时输出 protein `h_gvp_proj` / `h_esm_proj` RMS，用于判断 ESM/GVP 入口是否被压制。

### 影响范围
- 模型：`src/models/pocket_gvp_fusion_v11m5t.py`
- Probe 对照：`src/models/pocket_gvp_fusion_v11m4tp.py`
- 训练：`scripts/train/train_v11m5t.py`、`scripts/train/train_v11m4tp.py`、`scripts/train/train_v11m3.py`
- 配置：`configs/pocket_gvp_v11m5t_hpc.yaml`、`configs/pocket_gvp_v11m4tp_hpc.yaml`
- Slurm：`scripts/slurm/run_v11m5t_balanced_input.slurm`、`scripts/slurm/run_v11m4tp_probe.slurm`
- 推理/诊断：`scripts/infer/infer_v11_internal_ligdep.py`、`scripts/diagnostics/check_v11_input_balance.py`

### 配置变化
```yaml
model:
  use_ligand_groupwise_input_balance: true
  ligand_source_atom_dim: 109
  ligand_local_dim: 54
  ligand_global_dim: 55
  ligand_group_local_proj_dim: 96
  ligand_group_global_proj_dim: 64
  ligand_group_unimol_proj_dim: 64
  ligand_group_output_norm: true
  ligand_group_scale_init: 1.0
```

### 使用方法
```bash
sbatch scripts/slurm/run_v11m4tp_probe.slurm
sbatch scripts/slurm/run_v11m5t_balanced_input.slurm

python scripts/diagnostics/check_v11_input_balance.py \
  --checkpoint outputs/.../best_f1_c0_model.pt \
  --ligand_mode real \
  --split test \
  --batch_size 24 \
  --max_batches 16 \
  --output outputs/infer/input_balance_v11m5t
```

### 兼容性/迁移说明
- 旧 V11m5/V11m5t pair-bias checkpoint 仍可通过 git 历史回溯，但当前主线不再继续该结构。
- 新 V11m5t 与旧 V11m5t checkpoint 不结构兼容；应从头训练或从 V11m4t/V11m4 checkpoint 以 `strict=False` 初始化新增 adapter。

---

## V11m4t / V11m5t-light - 2026-05-26

### 主要特性
- ✅ `V11m4t`: 在 V11m4 late local pair-delta 上加入 global55 + UniMol delta calibration gate
- ✅ `V11m5t-light`: 保留 V11m5t 结构，但将 `pair_bias_beta_init` 从 `0.1` 降到 `0.03`
- ✅ 推理兼容：`scripts/infer/infer_v11_internal_ligdep.py` 可自动识别 `v11m4t/v11m5/v11m5t`

### 关键改进/变化
1. **隔离 global gate 贡献**：`V11m4t` 不使用 P→L pair-bias 前移，只验证 `delta_logits = global_gate(global55, UniMol) * local_delta_logits` 是否能改善 V11m4。
2. **测试 pair-bias 强度**：`V11m5t-light` 用更弱的 pair-bias 初始化，检验 V11m5/V11m5t 中 pair-bias 前移是否扰动 trunk。
3. **checkpoint 可评估**：ligand dependency inference loader 已支持新版本 checkpoint。

### 影响范围
- 模型：`src/models/pocket_gvp_fusion_v11m4t.py`
- 训练：`scripts/train/train_v11m4t.py`
- 配置：`configs/pocket_gvp_v11m4t_hpc.yaml`、`configs/pocket_gvp_v11m5t_light_hpc.yaml`
- Slurm：`scripts/slurm/run_v11m4t_global_gate.slurm`、`scripts/slurm/run_v11m5t_light_pair_bias_gate.slurm`
- 推理：`scripts/infer/infer_v11_internal_ligdep.py`

### 使用方法
```bash
sbatch scripts/slurm/run_v11m4t_global_gate.slurm
sbatch scripts/slurm/run_v11m5t_light_pair_bias_gate.slurm
```

---

## V11m4 / V11m5 / V11m5t - 2026-05-25

### 主要特性
- ✅ `V11m4`: V11m3 stable trunk + coordinate-free local54 pair-delta head
- ✅ `V11m5`: 将 local pair evidence 前移为 P→L cross-attention logit bias
- ✅ `V11m5t`: 在 V11m5 上用 global55 + UniMol CLS gate 校准 delta logits
- ✅ 训练选择器修正：V11m4+ train loop 保存 `best_f1_c0_model.pt` 和 `best_dice_c0_model.pt`

### 关键改进/变化
1. **V11m4 late delta**：保留 V11m3/V11m2 trunk，新增 local54-driven residue-level `delta_logits`，以保护 protein-side floor。
2. **V11m5 pair-bias 前移**：`PairEvidence(hP_i, hL_j, local54_j)` 加到 P→L attention logits，不引入任何 protein-ligand cross-distance。
3. **V11m5t calibration**：global55/UniMol 不进入 pair evidence，仅控制 delta amplitude，验证其作为 precision/calibration signal 的价值。
4. **Ligand dependency 诊断**：V11m4 epoch60 test F1_c0 约 `0.6000`；主要 dependency 来自 ligand graph/local54，UniMol 单独贡献较小。

### 影响范围
- 模型：`src/models/pocket_gvp_fusion_v11m4.py`、`src/models/pocket_gvp_fusion_v11m5.py`、`src/models/pocket_gvp_fusion_v11m5t.py`
- 训练：`scripts/train/train_v11m4.py`、`scripts/train/train_v11m5.py`、`scripts/train/train_v11m5t.py`
- 配置：`configs/pocket_gvp_v11m4_hpc.yaml`、`configs/pocket_gvp_v11m5_hpc.yaml`、`configs/pocket_gvp_v11m5t_hpc.yaml`
- Slurm：`scripts/slurm/run_v11m4_pair_delta.slurm`、`scripts/slurm/run_v11m5_pair_bias.slurm`、`scripts/slurm/run_v11m5t_pair_bias_gate.slurm`

---

## V12 - 2026-05-15

### 主要特性
- ✅ Interleaved Co-Attention：以 `N=6` 个 `CoAttentionBlock` 取代 V11 中分离的 protein context encoder、ligand self-attention block 与 cross-attention loop
- ✅ 标准 CrossAttentionBlock：移除跨模态几何 bias 路线，不再使用 `CrossModalDistanceBias` / `BiasedCrossAttentionBlock`
- ✅ Uni-Mol conditioning：`unimol_embed [B,512]` 同时用于每个 co-attention block 的 protein FiLM，以及 ligand global token augmentation
- ✅ RSA / SASA surface prior：新增 raw RSA + mask 输入，支持 co-attention 前 RSA input injection 与 classifier 前 SASA residual gate

### 关键改进/变化
1. **架构重排为 co-evolution blocks**：`src/models/pocket_gvp_fusion_v12.py` 保留 V11 的 GVP + ESM fusion 与 ligand GNN encoder，但将后续融合改成多层 interleaved co-attention。每个 block 顺序为 Uni-Mol FiLM、protein self-attention、ligand self-attention、P→L cross-attention、L→P cross-attention。
2. **移除跨坐标系距离偏置**：V12 明确不计算 Cα ↔ ligand atom 的跨模态距离，因为 protein / ligand 坐标系不保证处于 holo 对齐态。cross-attention 仅使用标准 `src/models/transformer_layers.py::CrossAttentionBlock`；几何 bias 只保留在 protein 内部 Cα-Cα 与 ligand 内部 atom-atom self-attention。
3. **Uni-Mol global token injection**：新增 `FiLMLayer` 与 `unimol_to_global`。若 `unimol_embed` 缺失，模型退化为零向量输入；相关投影 zero-init，因此初始行为接近 identity / 原 mean-pool global token。
4. **RSA / SASA 输入契约**：`src/data/collate.py` 在 `pocket_collate_fn_v10` 中 padding `protein_rsa_raw` 与 `protein_rsa_mask`，并新增 `prepare_model_inputs_v12()`。V12 模型用 `rsa_input_proj` 在 co-attention 前注入 RSA，用 `SASAResidualGate` 在 classifier 前做 bounded residual gate；`protein_rsa_mask` 防止旧 `.pt` 缺失 RSA 被误判为 RSA=0。
5. **V11 相关实际 diff**：当前 tracked diff 中 `scripts/train/train_v11.py` 只把文档示例的 `torchrun --nproc_per_node=4` 改为 `--nproc_per_node=2`；未看到新的 V11 train loop 行为改动。`src/data/collate.py` 的本轮 tracked 改动主要是 V12 RSA 输入与 `prepare_model_inputs_v12`。

### 影响范围（改了哪些东西）
- 模型：`src/models/pocket_gvp_fusion_v12.py`
- 数据/输入：`src/data/collate.py`
- Uni-Mol 预处理：`preparation/ligand/extract_unimol_embeddings.py`
- 推理辅助：`scripts/infer/infer_v11.py`（V11/V11m1/V11m2 checkpoint auto-detect，支持 HOLO4K/COACH420 与 ligand ablation modes）
- 训练脚本：`scripts/train/train_v11.py`（使用示例改为 2 GPU）
- 计划/状态记录：`TODO.md`

### 文件结构
```text
src/
  ├── models/
  │   └── pocket_gvp_fusion_v12.py       # V12 co-attention + Uni-Mol + RSA/SASA
  └── data/
      └── collate.py                     # protein_rsa_raw/mask + prepare_model_inputs_v12
preparation/
  └── ligand/
      └── extract_unimol_embeddings.py   # 离线提取 Uni-Mol CLS embedding
scripts/
  ├── train/
  │   └── train_v11.py                   # 示例 nproc_per_node 更新为 2
  └── infer/
      └── infer_v11.py                   # V11-family benchmark / ablation inference
```

### 配置变化
```yaml
model:
  d_unimol: 512
  n_co_attn_blocks: 6
  use_film: true
  use_prot_dist_bias: true
  use_lig_dist_bias: true
  sasa_gate_amplitude: 0.5
```

### 使用方法（当前状态）
```bash
# Uni-Mol embedding 离线提取
conda run -n AIAA python preparation/ligand/extract_unimol_embeddings.py \
  --source_root /home/husrcf/Code/ProtBind/New \
  --output_root data/unimol_embeddings \
  --batch_size 64 --workers 4

# V11-family 推理 / ablation
conda run -n AIAA python scripts/infer/infer_v11.py \
  --checkpoint outputs/.../best_model.pt \
  --set holo4k \
  --ligand_mode real \
  --output results/holo4k_real/
```

### 兼容性/迁移说明
- 与 V11 checkpoint **不兼容**：V12 class、fusion path、co-attention block 与新增 Uni-Mol/RSA/SASA 参数均改变。
- V12 forward 仍可在缺少 `unimol_embed` 时运行（零向量 fallback），缺少 `protein_rsa_raw` 时会跳过 RSA/SASA 路径。
- 旧数据若没有 `protein_rsa_mask`，collate 默认 mask=0；这会让 SASA/RSA 注入保持 identity，避免把缺失值当成真实 buried signal。

### 已知问题
- 当前工作区尚未看到 `configs/pocket_gvp_v12.yaml` 与 `scripts/train/train_v12.py`。
- `TODO.md` 记录完整 forward smoke test 仍待环境问题（`torch_scatter` / ROCm 兼容性）修复后验证。

---

## V11m3 - 2026-05（补录）

### 主要特性
- ✅ V11m2 + Uni-Mol ligand atom input：将 Uni-Mol CLS embedding `[B,512]` 投影到 64D 后按 `ligand_batch` 拼接到每个 ligand atom feature
- ✅ 独立实验分支：只扩展 `PocketGVPFusionV11m3` 的 ligand input dim，不改变 V11 / V11m1 / V11m2 的行为
- ✅ Uni-Mol cache 严格校验：训练脚本禁止缺失、全零、非有限或 shape 错误的 Uni-Mol embedding 继续训练
- ✅ 训练监控增强：额外记录 Uni-Mol coverage、raw norm、ligand delta norm、alpha、channel ratio 以及 Uni-Mol projection / alpha 梯度

### 关键改进/变化
1. **Uni-Mol → atom feature augmentation**：`src/models/pocket_gvp_fusion_v11m3.py` 继承 `PocketGVPFusionV11m2`，在 `Ligand3DEncoder` 前调用 `_augment_ligand_x()`。若 `use_unimol_ligand_input=true`，模型内部把 `d_ligand_atom` 从原始 109 扩展为 `109 + d_unimol_ligand_proj`。
2. **小幅可学习注入强度**：`unimol_ligand_proj = LayerNorm(512) → Linear(512,64) → GELU → LayerNorm(64)`，再乘以 `unimol_ligand_alpha`；默认 `unimol_ligand_alpha_init=0.02`，避免 Uni-Mol 在 step 0 直接压过原始 ligand descriptors。
3. **输入兼容但训练严格**：模型 forward 允许 `unimol_embed=None` 并用零向量 fallback，方便 smoke / 兼容调用；但 `scripts/train/train_v11m3.py` 的正式训练会通过 `load_unimol_cache()` 和 `get_batch_unimol()` 要求 cache 完整且有限。
4. **继承 V11m2 结构**：保留 ligand self-attention distance bias、`b_lig1` AttnRes anchor、ordinal label smoothing、distance auxiliary head、CNC 与 V11 的蛋白/配体主干。

### 影响范围（改了哪些东西）
- 模型：`src/models/pocket_gvp_fusion_v11m3.py`
- 训练脚本：`scripts/train/train_v11m3.py`
- 配置：`configs/pocket_gvp_v11m3_hpc.yaml`
- 输入准备：继续使用 `src/data/collate.py::pocket_collate_fn_v10` 与 `prepare_model_inputs_v11()`；`unimol_embed` 由训练脚本按 `pdb_ids` 额外传入模型

### 文件结构
```text
src/
  └── models/
      └── pocket_gvp_fusion_v11m3.py
scripts/
  └── train/
      └── train_v11m3.py
configs/
  └── pocket_gvp_v11m3_hpc.yaml
```

### 配置变化
```yaml
data:
  unimol_embeddings_dir: "unimol_embeddings"

model:
  use_unimol_ligand_input: true
  d_unimol: 512
  d_unimol_ligand_proj: 64
  unimol_ligand_alpha_init: 0.02
  use_lig_dist_bias: true
  use_lig_attnres_b1: true
  n_ligand_self_attn_layers: 4
```

### 使用方法（最小可复现）
```bash
python scripts/train/train_v11m3.py --config configs/pocket_gvp_v11m3_hpc.yaml
torchrun --nproc_per_node=2 scripts/train/train_v11m3.py --config configs/pocket_gvp_v11m3_hpc.yaml
```

### 兼容性/迁移说明
- 与 V11 / V11m1 / V11m2 checkpoint **不直接兼容**：V11m3 的 ligand encoder input dim 增加了 Uni-Mol projection channels。
- 数据 collate 仍与 V11 family 共用；不需要把 Uni-Mol 写入 batch tensor。Uni-Mol embedding 由训练脚本从 `data/{unimol_embeddings_dir}` 一次性载入 RAM 后按 `pdb_ids` 取 batch。
- 当前仓库只看到 `configs/pocket_gvp_v11m3_hpc.yaml`，未看到非 HPC 命名的 `configs/pocket_gvp_v11m3.yaml`。

### 已知问题
- 如果 `use_unimol_ligand_input=true`，正式训练不允许缺失 Uni-Mol cache；缺失 PDB embedding 会直接报错停止。
- 模型层面的零向量 fallback 只适合兼容 / smoke test，不应作为正式训练路径。

---

## V11m2 - 2026-05（补录）

### 主要特性
- ✅ Ligand self-attention 3D distance bias：在 ligand internal self-attention 内加入 atom-atom 距离 RBF additive bias
- ✅ `b_lig1` AttnRes anchor：将 post-self-attention 的 ligand representation 作为第二个 ligand residual anchor
- ✅ Ordinal label smoothing：训练时只在相邻类别之间平滑 Contact↔Near、Near↔Far，避免 Contact↔Far 的非相邻平滑

### 关键改进/变化
1. **LigandDistanceBias**：`src/models/pocket_gvp_fusion_v11.py` 中已有 `LigandDistanceBias` 与 `LigandSelfAttentionEncoder` 的可选 dist-bias 路径；`src/models/pocket_gvp_fusion_v11m2.py` 通过默认配置打开 `use_lig_dist_bias=true`、`lig_dist_bias_n_rbf=8`、`lig_dist_bias_max_dist=8.0`。
2. **Ligand AttnRes 双锚点**：V11 基础路径只用 `b_lig0`（raw ligand encoder + global token）作为 ligand-side AttnRes anchor；V11m2 打开 `use_lig_attnres_b1=true` 后，cross-attention 前每层可在 `[b_lig0, h_lig_ctx]` 与当前 partial 间做 per-position weighted merge。
3. **训练侧 ordinal smoothing**：`scripts/train/train_v11m2.py` 中的训练 loss 对三分类 ordinal label 做相邻类别平滑；Tversky、distance auxiliary loss 与 metrics 仍使用 hard labels。

### 影响范围（改了哪些东西）
- 模型：`src/models/pocket_gvp_fusion_v11m2.py`（薄 subclass，默认打开 V11m2 flags）
- 共享实现：`src/models/pocket_gvp_fusion_v11.py`（`LigandDistanceBias`、`LigandSelfAttentionEncoder`、`use_lig_attnres_b1` forward 逻辑）
- 训练脚本：`scripts/train/train_v11m2.py`
- 配置：`configs/pocket_gvp_v11m2.yaml`
- 输入准备：`src/data/collate.py::prepare_model_inputs_v11()`

### 文件结构
```text
src/
  └── models/
      ├── pocket_gvp_fusion_v11.py       # 共享 ligand dist-bias / AttnRes 实现
      └── pocket_gvp_fusion_v11m2.py     # 默认打开 V11m2 flags
scripts/
  └── train/
      └── train_v11m2.py
configs/
  └── pocket_gvp_v11m2.yaml
```

### 配置变化
```yaml
model:
  n_ligand_self_attn_layers: 2
  use_lig_dist_bias: true
  lig_dist_bias_n_rbf: 8
  lig_dist_bias_max_dist: 8.0
  use_lig_attnres_b1: true

training:
  ordinal_label_smoothing: true
```

### 使用方法（最小可复现）
```bash
python scripts/train/train_v11m2.py --config configs/pocket_gvp_v11m2.yaml
torchrun --nproc_per_node=2 scripts/train/train_v11m2.py --config configs/pocket_gvp_v11m2.yaml
```

### 兼容性/迁移说明
- 与 V11 checkpoint **大部分参数可复用但不保证严格兼容**：V11m2 打开 ligand dist-bias 后新增 per-layer bias projection 参数；`b_lig1` AttnRes 不新增参数，但改变 forward 信息流。
- 不引入 protein-ligand 跨坐标几何距离；ligand distance bias 只使用 ligand 自身坐标系下的 atom-atom 距离。

### 已知问题
- 依赖 `ligand_batch` 在 PyG batch 中按 sample 非递减排序；`PocketGVPFusionV11.forward()` 内部有 assert 检查。
- `configs/pocket_gvp_v11m2.yaml` 注释中 `label_contact_thresh: 5.0` 旁仍有旧注释 “Contact: d_min < 4.0Å”，实际值以 YAML key 为准。

---

## V11m1 - 2026-05（补录）

### 主要特性
- ✅ Ligand Global Gate：用 cross-attended ligand global token 对每个 protein feature channel 做 sigmoid gate
- ✅ Classifier input dim 不变：仍为 `d_model + d_phys_encoded = 640`
- ✅ 薄 subclass：除 final re-classification 外，主体完全复用 V11 forward

### 关键改进/变化
1. **Global ligand-conditioned gating**：`src/models/pocket_gvp_fusion_v11m1.py` 继承 V11，先运行 `super().forward()` 得到 `h_fused`、`h_ligand_fused` 与 `protein_physchem_encoded`，再用 `h_ligand_fused[:,0,:]` 生成 `gate = sigmoid(W lig_global)`。
2. **重算 logits**：V11m1 用 `h_prot_gated = h_fused * gate[:,None,:]` 替换原始 fused protein representation，然后复用同一个 `classifier_input_norm` 与 `classifier` 覆盖 `result['logits']`。
3. **半开初始化**：`lig_gate_proj.weight/bias` zero-init，因此 `sigmoid(0)=0.5`，训练起点是稳定的半开门控，而非随机强调某些通道。

### 影响范围（改了哪些东西）
- 模型：`src/models/pocket_gvp_fusion_v11m1.py`
- 训练脚本：`scripts/train/train_v11m1.py`
- 配置：`configs/pocket_gvp_v11m1.yaml`
- 输入准备：`src/data/collate.py::prepare_model_inputs_v11()`

### 文件结构
```text
src/
  └── models/
      └── pocket_gvp_fusion_v11m1.py
scripts/
  └── train/
      └── train_v11m1.py
configs/
  └── pocket_gvp_v11m1.yaml
```

### 配置变化
```yaml
experiment:
  name: "pocket_gvp_fusion_v11m1"

model:
  n_ligand_self_attn_layers: 2
```

### 使用方法（最小可复现）
```bash
python scripts/train/train_v11m1.py --config configs/pocket_gvp_v11m1.yaml
torchrun --nproc_per_node=2 scripts/train/train_v11m1.py --config configs/pocket_gvp_v11m1.yaml
```

### 兼容性/迁移说明
- 与 V11 checkpoint **不严格兼容**：V11m1 增加 `lig_gate_proj` 参数，并覆盖 logits 计算。
- 不改变 collate / dataset 契约；仍使用 V11 family 的 flat PyG ligand graph 输入。

### 已知问题
- 当前配置文件未看到专门的 `use_lig_global_gate` 开关；是否使用 V11m1 由训练脚本导入的模型类决定。
- Gate 初始为 0.5 倍 protein fused representation，严格来说不是 identity；这是实现中的稳定半开策略。

---

## V11 - 2026-05（补录）

### 主要特性
- ✅ Ligand Internal Self-Attention：在 ligand GNN encoder + dynamic global token 后、cross-attention 前加入 ligand self-attention
- ✅ Dynamic ligand global token：用 masked mean pool 生成 ligand global token，不再依赖固定 learnable global parameter
- ✅ AttnRes cross-attention loop：protein side 使用 `[b_gvp, b0, b1]` 三锚点，ligand side 使用 `b_lig0` 锚点
- ✅ 继承 V10m8：protein per-layer distance bias、distance regression auxiliary head、CNC、GVP output dropout、coordinate noise、RoPE、SwiGLU / gated attention

### 关键改进/变化
1. **Ligand self-attention 插入位置**：`src/models/pocket_gvp_fusion_v11.py` 在 `Ligand3DEncoder` 输出和 masked mean global token 拼接后运行 `LigandSelfAttentionEncoder`，让 ligand atoms 在进入 protein-ligand cross-attention 前获得分子内上下文。
2. **Cross-attention 前 AttnRes**：每个 bidirectional cross-attention layer 前，protein representation 先通过 `InterBlockAttnRes([b_gvp,b0,b1], partial)`，ligand representation 通过 `InterBlockAttnRes([b_lig0], partial)`；这提供 weighted merge，而不是简单残差相加。
3. **V11 输入收口**：`prepare_model_inputs_v11()` 不再传旧的 `ligand_to_protein` placeholder；模型签名仍保留 `ligand_to_protein: Optional` 但注释为 legacy ignored。
4. **Distance auxiliary head**：若 `use_dist_aux=true`，forward 返回 `pred_dist`；`scripts/train/train_v11.py` 在有 `raw_distances` 时加入 Huber/proximity-style distance auxiliary loss。

### 影响范围（改了哪些东西）
- 模型：`src/models/pocket_gvp_fusion_v11.py`
- 训练脚本：`scripts/train/train_v11.py`
- 配置：`configs/pocket_gvp_v11.yaml`
- 输入准备：`src/data/collate.py::pocket_collate_fn_v10` 与 `prepare_model_inputs_v11()`

### 文件结构
```text
src/
  ├── models/
  │   └── pocket_gvp_fusion_v11.py
  └── data/
      └── collate.py
scripts/
  └── train/
      └── train_v11.py
configs/
  └── pocket_gvp_v11.yaml
```

### 配置变化
```yaml
model:
  n_ligand_self_attn_layers: 2
  use_dist_aux: true
  use_dist_bias: true
  use_sdpa_protein_self_attn: true
  use_chain_features: true
  gvp_dropout_prob: 0.1
  coord_noise_std: 0.2

data:
  use_spatial_clamp: true
  spatial_clamp_dims: [62, 76, 78, 79]
  spatial_clamp_range: [-5.0, 5.0]
```

### 使用方法（最小可复现）
```bash
python scripts/train/train_v11.py --config configs/pocket_gvp_v11.yaml
torchrun --nproc_per_node=2 scripts/train/train_v11.py --config configs/pocket_gvp_v11.yaml
```

### 兼容性/迁移说明
- V11 与 V10m8 **不严格兼容**：V11 新增 ligand self-attention 模块，并改变 cross-attention 前的 ligand context。
- `prepare_model_inputs_v11()` 仍兼容 `pocket_collate_fn_v10` 输出的 flat ligand graph；旧 `ligand_to_protein` placeholder 不再需要。
- 不使用 protein-ligand 跨模态坐标距离；protein distance bias 只基于 Cα-Cα，ligand edge RBF 只基于 ligand atom-atom。

### 已知问题
- `ligand_batch` 必须按 sample 连续排序，否则 vectorized scatter 会触发 assert。
- `use_dist_aux=true` 需要 dataset/collate 提供 `raw_distances`；若 batch 中没有该字段，训练脚本会跳过 dist aux。
- `configs/pocket_gvp_v11.yaml` 注释中 `label_contact_thresh: 5.0` 旁仍有旧注释 “Contact: d_min < 4.0Å”，实际训练阈值以 YAML 数值为准。

---

## V10m8 - 2026-04-14

### 主要特性
- ✅ Per-layer distance bias: 每层 Transformer 使用独立的 DistanceBias 模块（AlphaFold2/ESMFold style），浅层学习局部几何，深层学习全局上下文
- ✅ Distance regression auxiliary head: 连续 d_min 预测辅助头，与 3-class 分类并行。Huber loss + proximity weighting，在类别边界处提供更平滑的梯度信号

### 关键改进/变化
1. **Per-layer distance bias**: 将 `RoPEProteinContextEncoder` 中共享的单个 `DistanceBias` 替换为 `ModuleList`，每个 Transformer layer 拥有独立的 DistanceBias module。参数量微增（16 RBF × 8 heads × 4 layers = 512 params）
2. **Distance regression aux head**: `Softplus` 激活确保预测距离 ≥ 0。训练时使用 proximity-weighted Huber loss（近距残基权重更高），推理时可忽略
3. **Dataset 支持**: `PocketDatasetV8` 现在输出 `raw_distances` 字段，`pocket_collate_fn_v10` 支持 padding

### 影响范围
- 模型：`src/models/pocket_gvp_fusion_v10m8.py`
- 数据：`src/data/pocket_dataset_v8.py`（新增 raw_distances 输出）
- 数据：`src/data/collate.py`（新增 raw_distances padding）
- 脚本：`scripts/train/train_v10m8.py`
- 配置：`configs/pocket_gvp_v10.yaml`

### 配置变化
```yaml
model:
  use_dist_aux: true              # 启用距离回归辅助头

training:
  dist_aux:
    enabled: true
    weight: 0.1                   # 辅助损失权重
    max_dist: 20.0                # clamp 距离上限
```

### 使用方法
```bash
python scripts/train/train_v10m8.py --config configs/pocket_gvp_v10.yaml
torchrun --nproc_per_node=4 scripts/train/train_v10m8.py --config configs/pocket_gvp_v10.yaml
```

---

## V10m7 - 2026-04-13

### 主要特性
- ✅ **Confident Negative Clipping (CNC)**：對 c_2 (far) 標籤的 noisy negative 進行置信度感知的 loss 降權

### 关键改进/变化
- **動機**：PDBbind 中同一蛋白有多配體 entry，殘基被標為 c_2 (far) 僅代表「不接觸此配體」，不代表「永不結合」。模型偵測到 binding-capable 區域但被錯誤懲罰。
- **機制**：當模型高置信（≥ clip_thresh）預測 c_2 標籤殘基為 positive 時，該殘基的 loss 權重降至 clip_ratio
- **CNC 同時作用於三個 loss**：
  - Focal Loss：per-residue loss × cnc_weights
  - Tversky Loss：FP 項 × cnc_weights（降低 clipped 殘基的 false positive 懲罰）
  - Weighted Dice Loss：pred_sum × cnc_weights（降低 clipped 殘基的分母貢獻）
- **防 collapse 機制**：warmup epochs、max_clip_frac 上限、clip_ratio > 0 保留微弱約束
- **理論基礎**：Confident Learning (Northcutt et al., JAIR 2021) + Bootstrap Loss (Reed et al., 2015)

### 新增文件
- `src/losses/confident_negative_clip.py`：CNC 獨立模組
- `scripts/train/train_v10m7.py`：訓練腳本（基於 v10m6）
- `configs/pocket_gvp_v10m7.yaml`：配置文件

### 配置参数
```yaml
cnc:
  enabled: true
  clip_thresh: 0.85       # positive 置信度閾值
  clip_ratio: 0.1         # 降權乘數
  warmup_epochs: 5        # 前 N epochs 不啟用
  max_clip_frac: 0.3      # 每 sample 最多 clip 30% 殘基
```

### TensorBoard 监控
- `CNC/batch_clip_fraction`：每 batch 被 clip 的殘基比例
- `CNC/epoch_clip_fraction`：每 epoch 被 clip 的殘基比例

### 复现命令
```bash
python scripts/train/train_v10m7.py --config configs/pocket_gvp_v10m7.yaml
```

### 模型架構
- 無架構改動，使用 V10m6 模型（`PocketGVPFusionV10m6`）
- 改動純在 loss 層面

---

## V10m6 - 2026-04-12

### 主要特性
- ✅ **Batched GVP Graph Construction**：per-sample for-loop → 單次 batched GVP forward pass（預估 2-4x 訓練加速）
- ✅ **向量化 Boundary Loss**：雙層 Python 迴圈 → 全 tensor 操作（預估 eval 2-5x 加速）

### 关键改进/变化

**模型層面（model file）：**
1. GVP 處理流程重構：
   - 舊：`for b in range(B)` 逐 sample 建圖 + 逐 sample 跑 GVP encoder
   - 新：所有 sample 的 valid residues flatten 成一個 flat tensor，per-sample 建圖後 offset edge indices 合併，**一次** GVP encoder forward pass，再 scatter 回 `[B, L, d]`
   - `build_protein_graph` 仍 per-sample 調用（KNN + backbone edges 需要 sample 隔離），但 GVP encoder 只跑一次
2. 繼承 V10m5 全部改動（Dynamic Ligand Global Token、gradient checkpointing fix、loss smart gating、cache_intermediates config）

**Loss 層面（boundary_loss.py，所有版本受益）：**
1. `compute_boundary_mask` 向量化重寫：
   - 舊：`for b in range(B): for i in valid_indices:` — O(B × N²) Python 迴圈
   - 新：`torch.cdist` → boolean mask → `.any(dim=2)` — 零 Python 迴圈
   - 輸出完全一致，純加速

### 影響範圍
- 配置：`configs/pocket_gvp_v10m6.yaml`
- 模型：`src/models/pocket_gvp_fusion_v10m6.py`（新建）
- 腳本：`scripts/train/train_v10m6.py`（新建）
- Loss：`src/losses/boundary_loss.py`（原地修改，所有版本受益）

### 文件结构
```
configs/
  └── pocket_gvp_v10m6.yaml
src/models/
  └── pocket_gvp_fusion_v10m6.py   # 新建，Batched GVP
src/losses/
  └── boundary_loss.py             # 原地修改，向量化
scripts/train/
  └── train_v10m6.py               # 新建
```

### 配置变化
```yaml
# 無新增配置項，僅 experiment.name 更新
experiment:
  name: "pocket_gvp_fusion_v10m6"
```

### 使用方法（最小可复现）
```bash
# 訓練
python scripts/train/train_v10m6.py --config configs/pocket_gvp_v10m6.yaml

# 分布式訓練
torchrun --nproc_per_node=4 scripts/train/train_v10m6.py --config configs/pocket_gvp_v10m6.yaml
```

### 兼容性/迁移说明
- 與 V10m5 checkpoint **不兼容**（class 名稱改為 `PocketGVPFusionV10m6`）
- 模型權重結構完全相同，僅 forward 計算路徑改變（數值等價）
- 無需重新生成特徵資料或距離標籤

### 已知問題
- `build_protein_graph` 仍 per-sample 調用（含 KNN），完全向量化需改為 batched KNN（PyG `knn_graph` 支援 `batch` 參數），留待後續
- `boundary_loss` 向量化版本在 N 很大時（>2000）會分配 `[B, N, N]` 記憶體，可能 OOM；建議 `max_seq_len ≤ 1800`

---

## V10m5 - 2026-04-12

### 主要特性
- ✅ **Dynamic Ligand Global Token**：固定 `nn.Parameter` → masked mean pool over ligand GNN outputs
- ✅ **Gradient Checkpointing Bug Fix**：啟用 checkpointing 時不再遺失 `coords` 和 `position_ids`
- ✅ **Loss Smart Gating**：weight=0 的 loss 分支完全跳過計算（含 softmax/one_hot）
- ✅ **可配置 cache_intermediates**：TensorBoard Health 診斷的中間 tensor cache 改為 config 控制
- ✅ **Label 閾值對齊 PoC**：Contact < 4.0Å / Near 4.0–6.0Å / Far > 6.0Å

### 关键改进/变化

**模型層面（model file）：**
1. Dynamic Ligand Global Token：
   - 舊：`self.ligand_global_token = nn.Parameter(torch.randn(1, 1, d_model))` — 所有配體共享同一固定向量
   - 新：`global_token = masked_mean_pool(h_ligand, ligand_mask)` — 每個配體有獨立的 global representation
   - 效果：cross-attention 第一層就能區分不同配體，零新增參數
2. Gradient Checkpointing Fix：
   - 舊：`checkpoint(encoder, h_seq, mask)` — 遺失 `coords`（distance bias）和 `position_ids`（chain-gap RoPE）
   - 新：用 closure 包裝 `coords` 和 `position_ids`，完整傳入
3. `_cached_*` 改為 `cache_intermediates` config 控制（預設 `false`，省記憶體）

**Loss 層面（train script）：**
1. `compute_combined_loss_v4` smart gating：
   - `focal_weight=0` → 跳過整個 focal 計算
   - `tversky_weight=0` → 跳過（原有）
   - `weighted_dice_weight=0` → 跳過（原有）
   - 共用 tensor（log_softmax、one_hot）lazy init，多個 loss 啟用時不重複計算

**標籤定義：**
- `label_contact_thresh`: 5.0 → 4.0（對齊 PoC）
- `label_near_thresh`: 8.0 → 6.0（對齊 PoC）

### 影響範圍
- 配置：`configs/pocket_gvp_v10m5.yaml`
- 模型：`src/models/pocket_gvp_fusion_v10m5.py`（新建）
- 腳本：`scripts/train/train_v10m5.py`（新建）

### 文件结构
```
configs/
  └── pocket_gvp_v10m5.yaml
src/models/
  └── pocket_gvp_fusion_v10m5.py   # 新建
scripts/train/
  └── train_v10m5.py               # 新建
```

### 配置变化
```yaml
model:
  cache_intermediates: false  # NEW: 控制 _cached_* 中間 tensor（TensorBoard Health 用）

data:
  label_contact_thresh: 4.0   # 5.0 → 4.0（對齊 PoC）
  label_near_thresh: 6.0      # 8.0 → 6.0（對齊 PoC）
```

### 使用方法（最小可复现）
```bash
# 訓練
python scripts/train/train_v10m5.py --config configs/pocket_gvp_v10m5.yaml

# 分布式訓練
torchrun --nproc_per_node=4 scripts/train/train_v10m5.py --config configs/pocket_gvp_v10m5.yaml
```

### 兼容性/迁移说明
- 與 V10m4 checkpoint **不兼容**（class 名稱改為 `PocketGVPFusionV10m5`；`ligand_global_token` 參數被移除）
- 標籤閾值改變（4/6 vs 5/8），**必須重新訓練**，不可直接 finetune V10m4 checkpoint
- 無需重新生成特徵資料；距離標籤檔案不變（閾值在 dataloader 動態計算）

---

## V10m4 - 2026-04-05

### 主要特性
- ✅ 修復 Dice Loss 主導訓練導致泛化災難（global batch Dice → per-sample Tversky）
- ✅ 新增 GVP Output Dropout（sample-level，對稱於 ESM dropout）
- ✅ 新增 CA Coordinate Noise（結構擾動增強，影響 KNN 圖拓撲）
- ✅ 三層 GVP 正則化機制完整化

### 关键改进/变化

**Loss 層面（train script）：**
1. 移除 global batch Dice Loss（根因：全局聚合使少數 Contact class 梯度被 Far 稀釋，訓練不穩定）
2. 新增 `compute_tversky_loss`：per-sample、Contact-only、smooth=1.0、alpha=0.3/beta=0.7
   - alpha < beta：允許適量 FP，強調 recall（不漏 binding site）
   - smooth=1.0：訓練早期梯度穩定，杜絕梯度爆炸
3. 新增 `compute_weighted_dice_loss`：per-sample、三 class、class_weights 正規化加權（可選）
4. 統一 `compute_combined_loss_v4`，支援 sample_weights（兼容 cluster dedup）
5. label_smoothing: 0.15 → 0.05（過高 smoothing 對少數 Contact class 更不利）
6. focal_gamma: 2.0 → 2.5（加強 hard example mining）
7. class_weights: [7,3,2] → [8,3,1]（Contact↑、Far↓）

**模型層面（model file）：**
1. GVP Output Dropout（L2）：訓練時以 `gvp_dropout_prob` 機率將整個樣本的 GVP 輸出 mask 為零
   - 迫使模型不過度依賴幾何特徵，與 ESM dropout 對稱
2. Coordinate Noise（L3）：在 `build_protein_graph` 前對 CA/N/C 座標加相同 Gaussian noise
   - 同時改變 KNN 圖拓撲、邊距離和 local backbone frame，結構正則化更強
   - 僅訓練期間啟用

**蛋白 GVP 邊特徵真實化（`build_protein_graph`）：**
- 舊：`edge_s [E,1]` = 原始距離；`edge_v [E,1,3]` = 全局方向
- 新：`edge_s [E,20]` = RBF(16) + seq offset 4-class；`edge_v [E,2,3]` = global + local backbone frame
- 強制保留 backbone i±1 邊（coord noise 下不丟失）
- Sequential offset 4-class: backbone / short / medium / long range → 模型能區分肽鍵與遠程接觸
- N-CA-C local frame（第 2 個 edge vector）：模型學 frame-relative 方向，有利 H-bond 幾何

**配體側邊特徵增強：**
- 舊：RDKit 化學邊 6D（bond type/ring/aromaticity），缺乏 3D 距離信息
- 新：化學邊 6D + RBF(原子間距, 8 bins, 0-5Å) = 14D
- 在 forward 中在線計算，無需重新生成 ligand feature 資料

**3D Distance-Biased Attention（蛋白自注意力）：**
- 在 protein context encoder（4 層 RoPE Transformer）的 self-attention 加入空間距離偏置
- `DistanceBias` 模塊：CA-CA pairwise distances → RBF(16) → Linear → per-head bias `[B, n_heads, L, L]`
- 零初始化：訓練起點等同標準 attention，偏置逐步學習（訓練穩定）
- 距離矩陣預計算一次，所有 4 層共享
- **無洩題**：僅使用蛋白質自身 CA 座標（protein-internal）；不涉及配體位置

**三層 GVP 正則化完整體系：**
```
L1: internal GVP Dropout  (layer-level scalar+vector channel dropout, drop_rate=dropout=0.1)
L2: gvp_dropout_prob=0.1  (output-level sample mask, NEW in V10m4)
L3: coord_noise_std=0.5Å  (input coord Gaussian noise, NEW in V10m4)
```

### 影響範圍

- 配置：`configs/pocket_gvp_v10m4.yaml`
- 模型：`src/models/pocket_gvp_fusion_v10m4.py`（新建，繼承 V10m3 架構）
- 腳本：`scripts/train/train_v10m4.py`（新建）

### 文件结构
```
configs/
  └── pocket_gvp_v10m4.yaml
src/models/
  └── pocket_gvp_fusion_v10m4.py   # 新建，架構同 V10m3 + GVP dropout
scripts/train/
  └── train_v10m4.py               # 新建，新 loss 函數
```

### 配置变化
```yaml
model:
  gvp_dropout_prob: 0.1       # NEW: GVP output sample-level dropout
  coord_noise_std: 0.5        # NEW: CA coord noise (Å)
  gvp_edge_rbf_bins: 16       # NEW: protein edge RBF bins
  gvp_edge_rbf_dmax: 20.0     # NEW: protein edge RBF max dist (Å)
  ligand_edge_rbf_bins: 8     # NEW: ligand edge RBF bins (appended to chemical bond 6D)

training:
  focal_gamma: 2.5          # 2.0 → 2.5
  class_weights: [8.0, 3.0, 1.0]  # [7,3,2] → [8,3,1]
  focal_weight: 0.7
  tversky_weight: 0.3       # NEW: per-sample Contact-only Tversky
  tversky_alpha: 0.3
  tversky_beta: 0.7
  tversky_smooth: 1.0
  weighted_dice_weight: 0.0 # optional, disabled by default
  label_smoothing: 0.05     # 0.15 → 0.05
  dice_weight: 0.0          # deprecated
```

### 使用方法（最小可复现）
```bash
# 訓練
python scripts/train/train_v10m4.py --config configs/pocket_gvp_v10m4.yaml

# 純 Focal 基線對比（在 config 中設置 tversky_weight: 0.0, focal_weight: 1.0）
python scripts/train/train_v10m4.py --config configs/pocket_gvp_v10m4.yaml

# 評測（使用 V10m3 推理腳本，模型架構相同）
python scripts/infer/infer_v10m3.py --checkpoint outputs/.../best_model.pt ...
```

### 兼容性/迁移说明
- 與 V10m3 checkpoint **不兼容**（class 名稱從 `PocketGVPFusionV10m3` 改為 `PocketGVPFusionV10m4`）
- 需重新訓練；無需重新生成特徵資料或距離標籤

### 設計動機（技術細節）
global batch Dice 的失效機制：
1. 全 batch 聚合 → Contact 5% tokens 對 Loss 貢獻遠小於 Far 70% tokens
2. 初期隨機預測 → Dice(Contact)≈0 → 巨大梯度 → 模型瘋狂預測 Contact
3. train Dice↑ 但 precision 崩潰 → test 嚴重 over-prediction → 泛化災難

Tversky 修復：per-sample + Contact-only + smooth=1.0 → 梯度穩定、方向正確

---

## V10m3 - 2026-04-04

### 主要特性
- ✅ **d_center 标签体系回归**: 从 VDW 距离标签切换回经典重原子中心距离（d_center）
- ✅ **主流 benchmark 对齐**: Contact < 5.0Å / Near 5.0–8.0Å / Far > 8.0Å，与 BioLiP / P2Rank / COACH 一致
- ✅ **TensorBoard Val/ 修复**: 移除 `_original` 路由逻辑，d_center 指标直接写入 `Val/`
- ✅ **Health/ 诊断恢复**: 补全 `_cached_*` 属性赋值，contribution_rate 和 gate_openness 正常记录

### 关键改进/变化

#### 1. 标签体系切换
- **动机**: VDW 距离标签（V10m2）与主流 benchmark 不对齐，难以横向比较
- **改动**: `PocketDatasetV8` 新增 `label_contact_thresh` / `label_near_thresh` 参数（默认保持旧值兼容）
- **V10m3 阈值**: Contact < 5.0Å，Near 5.0–8.0Å，Far > 8.0Å
- **距离文件**: `data/residue_ligand_distances_dcenter.pt`（重原子中心最小距离，19029 对）

#### 2. 距离文件重新生成
- **问题**: `residue_ligand_distances_v2.pt` 只有 13374 条（脚本硬编码了错误路径 `P-L/`）
- **修复**: 新建 `preparation/tag/calculate_distances_dcenter.py`，`--pl-dir` 默认指向 `New/`

#### 3. TensorBoard 修复
- **问题**: `Val/` 命名空间被 `if 'loss_original' in val_metrics` 门控，v10m3 无 `distances_file_original` 时永远为空
- **修复**: 移除 `_original` 分支，d_center 指标直接写入 `Val/`

#### 4. Health/ 诊断恢复
- **问题**: v10m2/m3 的 `forward` 未设置 `self._cached_*` 属性，训练脚本 `hasattr` 检查静默跳过
- **修复**: 在 `forward` 中补全 `_cached_gvp_scalar/vector/h_geo/current_esm/h_gvp_proj/h_esm_proj/gate`

### 影响范围
- 配置：`configs/pocket_gvp_v10m3.yaml`（新增）
- 模型：`src/models/pocket_gvp_fusion_v10m3.py`（新增，含 `_cached_*` 修复）
- 脚本：`scripts/train/train_v10m3.py`（新增）、`scripts/infer/infer_v10m3.py`（新增）
- 数据：`preparation/tag/calculate_distances_dcenter.py`（新增）
- 数据集：`src/data/pocket_dataset_v8.py`（新增 `label_contact_thresh/label_near_thresh` 参数）

### 文件结构
```
configs/
  └── pocket_gvp_v10m3.yaml
src/
  ├── models/
  │   └── pocket_gvp_fusion_v10m3.py
  └── data/
      └── pocket_dataset_v8.py          # 新增阈值参数
scripts/
  ├── train/
  │   └── train_v10m3.py
  └── infer/
      └── infer_v10m3.py
preparation/tag/
  └── calculate_distances_dcenter.py
data/
  └── residue_ligand_distances_dcenter.pt  # 需先运行生成脚本
```

### 配置变化
```yaml
data:
  distances_file: "data/residue_ligand_distances_dcenter.pt"
  label_contact_thresh: 5.0   # Contact: d_center < 5.0Å
  label_near_thresh: 8.0      # Near: 5.0–8.0Å; Far: >8.0Å

training:
  class_weights: [4.0, 1.0, 2.0]   # Contact 类更大，权重相应调整
```

### 使用方法（最小可复现）
```bash
# 1. 生成 d_center 距离文件（首次需要）
python preparation/tag/calculate_distances_dcenter.py --workers 64

# 2. 训练
python scripts/train/train_v10m3.py --config configs/pocket_gvp_v10m3.yaml --amp

# 3. 推理（评估模式）
python scripts/infer/infer_v10m3.py \
    --checkpoint outputs/xxx/best_model.pt \
    --split_file configs/splits/splits.txt \
    --evaluate --output results/

# 4. 全局推理（test set 分类）
python scripts/infer/infer_v10m3.py \
    --checkpoint outputs/xxx/best_model.pt \
    --split_file configs/splits/splits.txt \
    --global_infer --output results/
```

### 兼容性/迁移说明
- 与 V10m2 checkpoint **不兼容**（标签体系不同，需重新训练）
- 需要先运行 `calculate_distances_dcenter.py` 生成距离文件
- `PocketDatasetV8` 新参数有默认值，V10m1/m2 训练脚本无需修改

---

## V10m2 - 2026-04-03

### 主要特性
- ✅ **RoPE 维度 Bug 修复**: `cos/sin` 切片到 `head_dim//2` 后再 unsqueeze，消除 [32 vs 64] 广播错误
- ✅ **b_gvp 维度修复**: GVP 内部输出 256D → 投影到 d_model 512D，消除 `stack` size mismatch
- ✅ **Health/ 诊断恢复**: 补全 `_cached_*` 属性赋值

### 关键改进/变化

#### 1. RoPE 维度修复
- **错误**: `apply_rotary_pos_emb` 中 `cos [L, 64]` 直接 unsqueeze 后与 `x1 [B,H,L,32]` 广播失败
- **修复**: `half = x1.shape[-1]; cos = cos[:, :half].unsqueeze(0).unsqueeze(0)`

#### 2. b_gvp 投影修复
- **错误**: `InterBlockAttnRes.forward` 中 `torch.stack([b_gvp, b0, b1])` 要求三者最后维度相同，但 `b_gvp = h_s_gvp` 为 256D，b0/b1 为 512D
- **修复**: `__init__` 中新增 `self.b_gvp_proj = nn.Linear(d_gvp_internal, d_model, bias=False)`

### 影响范围
- 模型：`src/models/pocket_gvp_fusion_v10m2.py`（修复 + `_cached_*`）
- 脚本：`scripts/train/train_v10m2.py`

### 文件结构
```
src/models/pocket_gvp_fusion_v10m2.py
scripts/train/train_v10m2.py
```

### 使用方法（最小可复现）
```bash
python scripts/train/train_v10m2.py --config configs/pocket_gvp_v10.yaml --amp
```

### 兼容性/迁移说明
- 与 V10m1 checkpoint **不兼容**（新增 `b_gvp_proj` 参数）
- 使用与 V10m1 相同的 VDW 距离标签和配置文件

---

## V10m1 - 2026-04-01

### 主要特性
- ✅ **RoPE (Rotary Position Embedding)**: Transformer 中引入相对位置编码
- ✅ **Ligand-side AttnRes**: 配体侧也引入 Inter-Block Attention Residuals
- ✅ **b_gvp 额外 block**: GVP 纯几何输出作为第三个 AttnRes block，几何信号直达后期交叉注意力

### 关键改进/变化

#### 1. RoPE
- 替换 `ProteinContextEncoder` 中的标准 MHA 为 `RoPEGatedMultiHeadAttention`
- 使用 `RotaryEmbedding` 对 Q/K 施加旋转位置编码，支持相对位置感知

#### 2. Ligand-side AttnRes
- V10 只有蛋白侧 AttnRes（b0, b1）；V10m1 新增配体侧 `lig_attn_res`（b_lig0）
- 每层交叉注意力前，配体表征也回溯早期编码器输出

#### 3. b_gvp block
- Block 划分：b_gvp（GVP 输出）、b0（融合后）、b1（Transformer 后）
- 蛋白侧 AttnRes 从 2 个 block 扩展到 3 个

### 影响范围
- 模型：`src/models/pocket_gvp_fusion_v10m1.py`（新增）
- 脚本：`scripts/train/train_v10m1.py`（新增）

### 文件结构
```
src/models/pocket_gvp_fusion_v10m1.py
scripts/train/train_v10m1.py
```

### 配置变化
```yaml
# 复用 configs/pocket_gvp_v10.yaml，无新增配置项
```

### 使用方法（最小可复现）
```bash
python scripts/train/train_v10m1.py --config configs/pocket_gvp_v10.yaml --amp
```

### 兼容性/迁移说明
- 与 V10t checkpoint **不兼容**（新增 `lig_attn_res`、`b_gvp_proj` 参数）

---

## V10t - 2026-03-28

### 主要特性
- ✅ **Inter-Block Attention Residuals (AttnRes)**: 蛋白侧跨 block 注意力残差（参考 Kimi 2025）
- ✅ **BF16 训练支持**: 替代 FP16，在 AMD GPU 上更稳定
- ✅ **梯度累积**: 支持 `accumulation_steps` 配置
- ✅ **异步数据传输**: `non_blocking=True` + `pin_memory`
- ✅ **Tensor 累积器**: 减少 CPU-GPU 同步次数，提升训练吞吐

### 关键改进/变化

#### 1. Inter-Block AttnRes
- 每层交叉注意力前，蛋白表征通过 `InterBlockAttnRes` 融合 b0（融合后）和 b1（Transformer 后）
- 使用可学习的 softmax 权重，初始化为零（保守初始化）
- 允许模型在不同抽象层次的表征间动态选择

#### 2. 训练效率优化
- BF16：`torch.bfloat16` 替代 `torch.float16`，数值范围更大，溢出风险更低
- 梯度累积：等效扩大 batch size，适合显存受限场景
- 异步传输：DataLoader `pin_memory=True` + `.to(device, non_blocking=True)`

### 影响范围
- 模型：`src/models/pocket_gvp_fusion_v10t.py`（新增）
- 脚本：`scripts/train/train_v10t.py`（新增）
- 配置：`configs/pocket_gvp_v10.yaml`（新增）

### 文件结构
```
configs/
  └── pocket_gvp_v10.yaml
src/models/
  └── pocket_gvp_fusion_v10t.py
scripts/train/
  └── train_v10t.py
```

### 配置变化
```yaml
training:
  use_bf16: false          # V10: BF16 支持（默认关闭，CLI --amp 开启）
  accumulation_steps: 1    # V10: 梯度累积步数
  pin_memory: true         # V10: 异步数据传输
  prefetch_factor: 2
  cudnn_benchmark: false
```

### 使用方法（最小可复现）
```bash
# 单 GPU，BF16
python scripts/train/train_v10t.py --config configs/pocket_gvp_v10.yaml --amp

# 多 GPU
torchrun --nproc_per_node=4 scripts/train/train_v10t.py --config configs/pocket_gvp_v10.yaml
```

### 兼容性/迁移说明
- 与 V9m2 checkpoint **不兼容**（新增 AttnRes 模块）
- 继承 V9m2 的所有特征（complete_features_v4、NMA、VDW 标签）

---

## V9m2 - 2026-03-02

### 主要特性
- ✅ **Flash Attention**: 使用triton版本加速训练（ROCm兼容），预期加速1.5-2x
- ✅ **Enhanced Label Smoothing**: 从0.1提升到0.15，提高泛化能力
- ✅ **Boundary Loss**: 对结合位点边界残基施加额外权重，提升边界预测精度

### 关键改进/变化

#### 1. Flash Attention集成
- **动机**: 加速训练，减少显存占用
- **实现**: 在`GatedMultiHeadAttention`中添加Flash Attention支持
- **特点**:
  - 使用triton版本，兼容ROCm
  - 自动fallback到标准attention（当mask存在或cross-attention时）
  - 仅在训练时启用，推理时使用标准attention
- **性能**: 预期训练速度提升1.5-2x，显存占用减少~20%

#### 2. Enhanced Label Smoothing
- **动机**: 提高模型泛化能力，减少过拟合
- **改进**: V9的0.1 → V9m2的0.15
- **原理**: 软化标签分布 `y_smooth = (1-ε)*y_hard + ε/K`，防止模型对预测过于自信

#### 3. Boundary Loss
- **动机**: 结合位点边界残基最难预测，需要额外关注
- **实现**:
  - 识别边界残基：在8Å范围内存在不同类别邻居的残基
  - 对边界残基施加2x权重
  - 损失组合：`Total = 0.4*Dice + 0.4*Focal + 0.2*Boundary`
- **预期**: 边界预测F1提升2-3%

### 影响范围（改了哪些东西）

#### 新增文件
- `src/models/pocket_gvp_fusion_v9m2.py` - V9m2模型
- `src/losses/boundary_loss.py` - 边界损失实现
- `scripts/train/train_v9m2.py` - V9m2训练脚本（内置默认参数）
- `scripts/test_v9m2_setup.py` - 设置测试脚本
- `TODO.md` - 未来优化方向（10大类，40+具体建议）

#### 修改文件
- `src/models/transformer_layers.py` - 添加Flash Attention支持
  - `GatedMultiHeadAttention`: 新增`use_flash_attn`参数
  - `GatedTransformerBlock`: 传递`use_flash_attn`参数
  - `ProteinContextEncoder`: 传递`use_flash_attn`参数
- `src/losses/__init__.py` - 导出boundary loss函数

#### 配置说明
- V9m2**复用v9配置文件** (`configs/pocket_gvp_v9.yaml`)
- V9m2特性通过训练脚本的默认参数启用
- 要启用完整V9m2功能，需在v9配置中添加相应参数

### 文件结构
```
configs/
  └── pocket_gvp_v9.yaml            # V9m2使用v9配置
src/
  ├── models/
  │   ├── pocket_gvp_fusion_v9m2.py # V9m2模型
  │   └── transformer_layers.py     # 添加Flash Attention支持
  └── losses/
      ├── boundary_loss.py          # 边界损失
      └── __init__.py               # 更新导出
scripts/
  ├── train/
  │   └── train_v9m2.py             # V9m2训练脚本（内置默认参数）
  └── test_v9m2_setup.py            # 设置测试
TODO.md                              # 未来优化方向
```

### 配置变化
```yaml
# V9m2使用v9配置文件: configs/pocket_gvp_v9.yaml
# V9m2的新功能通过训练脚本的默认参数启用

# 在train_v9m2.py中的默认值:
use_boundary_loss: false (默认)      # 设置为true启用boundary loss
boundary_weight: 0.2                 # boundary loss权重
boundary_multiplier: 2.0             # 边界残基权重倍数
boundary_distance_threshold: 8.0     # 边界判定距离阈值(Å)
label_smoothing: 0.15                # 从0.1提升到0.15
use_flash_attn: true                 # 在model配置中设置

# 要启用V9m2的完整功能，在v9配置中添加:
model:
  use_flash_attn: true               # 启用Flash Attention

training:
  label_smoothing: 0.15              # 增强label smoothing
  use_boundary_loss: true            # 启用boundary loss
  boundary_weight: 0.2
  boundary_multiplier: 2.0
  boundary_distance_threshold: 8.0
  dice_weight: 0.4                   # 调整以平衡boundary loss
  focal_weight: 0.4
```

### 使用方法（最小可复现）
```bash
# 测试设置
python scripts/test_v9m2_setup.py

# 单GPU训练（使用v9配置）
python scripts/train/train_v9m2.py --config configs/pocket_gvp_v9.yaml

# 多GPU训练 (DDP)
torchrun --nproc_per_node=4 scripts/train/train_v9m2.py --config configs/pocket_gvp_v9.yaml

# V9m2默认启用以下功能（通过脚本默认参数）:
# - label_smoothing: 0.15
# - 其他V9m2功能需要在配置文件中显式启用

# 要启用完整V9m2功能，修改configs/pocket_gvp_v9.yaml:
# model:
#   use_flash_attn: true
# training:
#   label_smoothing: 0.15
#   use_boundary_loss: true
#   boundary_weight: 0.2
#   dice_weight: 0.4
#   focal_weight: 0.4
```

### 测试结果
```
✓ Flash Attention 已安装 (v2.8.3)
✓ V9m2 模型实例化成功 (23.8M参数)
✓ Boundary Loss 计算成功
✓ 配置文件加载成功
✓ Transformer Layers 测试通过
```

### 预期性能提升
- **训练速度**: +50-100% (Flash Attention)
- **显存占用**: -20% (Flash Attention)
- **边界预测**: +2-3% F1 (Boundary Loss)
- **泛化能力**: +1-2% (Enhanced Label Smoothing)
- **整体性能**: 预期 +3-5% F1

### 兼容性/迁移说明
- 与 V9 checkpoint **不完全兼容**（模型结构相同，但配置不同）
- 可以加载V9权重，但需要调整配置文件
- 不需要重新生成特征（使用相同的complete_features_v3和complete_features_v3_nma）
- 不需要重新生成标签（使用相同的VDW距离标签）

### 已知问题
- Flash Attention在极长序列（>2000残基）时可能不稳定，建议禁用
- Boundary Loss计算对于大batch可能较慢，可考虑降低boundary_weight或禁用

### 下一步优化方向
详见 `TODO.md`，优先级建议：
- **短期** (1-2天): 启用AMP、Layer-wise LR Decay、Per-residue Confidence
- **中期** (1周): Contrastive Loss、NMA质量控制、快照集成
- **长期** (2周+): 半监督学习、Multi-scale Fusion、难样本挖掘

---

## V9 - 2026-02-24

### 主要特性
- ✅ **Stochastic Depth (Drop Path)**: Transformer和Cross-Attention层线性缩放的drop_path
- ✅ **Separate Attention Dropout**: 注意力权重矩阵独立dropout（与FFN dropout分离）
- ✅ **Classifier Dropout**: 分类头dropout提升至0.2
- ✅ **Weight Decay优化**: 从0.01提升到0.05，并通过param groups排除LayerNorm/bias
- ✅ **NMA数据增强**: 每个蛋白质随机采样多个构象（Normal Mode Analysis）

### 关键改进/变化

#### 1. Stochastic Depth (Drop Path)
- **动机**: 提高模型正则化，减少过拟合
- **实现**:
  - Transformer层：线性缩放drop_path_prob从0到0.1（按层递增）
  - Cross-Attention层：同样线性缩放（修复了V9初版的bug）
- **原理**: 训练时随机跳过残差路径，类似Dropout但作用于整个层

#### 2. Separate Attention Dropout
- **动机**: 注意力权重和FFN输出需要不同的dropout策略
- **实现**:
  - `attn_dropout`: 专门用于attention权重矩阵（默认0.1）
  - `dropout`: 用于FFN和其他地方（默认0.1）
- **效果**: 更精细的正则化控制

#### 3. Classifier Dropout增强
- **动机**: 分类头容易过拟合，需要更强的正则化
- **改进**: 从0.1提升到0.2

#### 4. Weight Decay优化
- **动机**: 更强的权重衰减，但排除不应该衰减的参数
- **实现**:
  ```python
  no_decay_keywords = ['bias', 'norm.weight', 'norm.bias', 'LayerNorm']
  param_groups = [
      {'params': decay_params, 'weight_decay': 0.05},
      {'params': no_decay_params, 'weight_decay': 0.0}
  ]
  ```

#### 5. NMA数据增强
- **动机**: 增加蛋白质构象多样性，提高模型对柔性的鲁棒性
- **实现**:
  - 使用ProDy ANM生成10个构象（conf0-conf9）
  - 训练时每个蛋白随机采样3个构象
  - conf0始终包含（原始构象），conf1-9随机采样
  - ESM/Ligand/标签不变，只换蛋白质坐标相关特征
- **数据**: `complete_features_v3_nma/` 目录
- **效果**: 训练集扩大3倍，验证/测试集仍用conf0

### 影响范围（改了哪些东西）

#### 新增文件
- `src/models/pocket_gvp_fusion_v9.py` - V9模型
- `configs/pocket_gvp_v9.yaml` - V9配置
- `scripts/train/train_v9.py` - V9训练脚本
- `preparation/protein/extract_complete_features_v3_nma.py` - NMA特征提取

#### 修改文件
- `src/models/transformer_layers.py` - 添加drop_path和attn_dropout支持
- `src/data/pocket_dataset_v8.py` - NMA数据加载支持

### 文件结构
```
configs/
  └── pocket_gvp_v9.yaml
src/
  ├── models/
  │   ├── pocket_gvp_fusion_v9.py
  │   └── transformer_layers.py      # 添加drop_path, attn_dropout
  └── data/
      └── pocket_dataset_v8.py       # NMA支持
scripts/
  └── train/
      └── train_v9.py
preparation/protein/
  └── extract_complete_features_v3_nma.py
data/
  └── complete_features_v3_nma/      # NMA特征
      └── YEAR_RANGE/
          └── PDB_ID/
              ├── {pdb_id}_protein_conf0.pt  # 原始
              ├── {pdb_id}_protein_conf1.pt
              └── ...
              └── {pdb_id}_protein_conf9.pt
```

### 配置变化
```yaml
# configs/pocket_gvp_v9.yaml

experiment:
  name: "pocket_gvp_fusion_v9"

data:
  nma_features_dir: "complete_features_v3_nma"  # V9: NMA特征目录
  nma_n_confs: 3                                # V9: 每个蛋白采样3个构象
  nma_max_conf: 9                               # V9: 最大构象索引

model:
  # V9: 正则化增强
  attn_dropout: 0.1                             # V9: 独立attention dropout
  classifier_dropout: 0.2                       # V9: 分类头dropout (从0.1提升)
  stochastic_depth_prob: 0.1                    # V9: 最大drop_path概率

training:
  weight_decay: 0.05                            # V9: 从0.01提升 (排除LayerNorm/bias)
  label_smoothing: 0.1                          # 继承V8
```

### 使用方法（最小可复现）
```bash
# 生成NMA特征（如果还没有）
python preparation/protein/extract_complete_features_v3_nma.py \
  --pdb_dir data/pdbs \
  --output_dir data/complete_features_v3_nma \
  --n_modes 10 \
  --n_confs 9 \
  --amplitude 1.0

# 训练
python scripts/train/train_v9.py --config configs/pocket_gvp_v9.yaml

# 多GPU训练
torchrun --nproc_per_node=4 scripts/train/train_v9.py --config configs/pocket_gvp_v9.yaml
```

### NMA特征生成参数
```python
# ProDy ANM参数
n_modes = 10          # 使用前10个低频振动模式
n_confs = 9           # 生成9个扰动构象（加上conf0共10个）
amplitude = 1.0       # 振幅缩放因子
cutoff = 15.0         # ANM cutoff距离(Å)
```

### 兼容性/迁移说明
- 与 V8 checkpoint **不兼容**（模型结构相同，但需要NMA数据）
- 需要生成NMA特征：运行`extract_complete_features_v3_nma.py`
- 不需要重新生成标签（使用相同的VDW距离标签）
- 可以禁用NMA回退到V8模式：设置`nma_n_confs: 1`

### 已知问题
- NMA特征生成较慢（每个蛋白~10-30秒）
- 需要大量存储空间（10x原始特征）
- 某些蛋白质结构可能导致ANM失败（自动跳过）

### 验证结果
- NMA覆盖率: 100% (19,029/19,029 proteins)
- 每个蛋白平均构象数: 10.0
- 数据集扩展: 19,029 → 57,087 samples (3x)
- Drop path线性缩放验证: [0.0, 0.033, 0.067, 0.1] ✓

---

## V8 - 2026-02-24

### 主要特性
- ✅ **Complete Features V3 Integration**: 使用 V3 特征提取（62D PhysChem + 72D Spatial Scalar + 6×3D Spatial Vectors）
- ✅ **Enhanced GVP with 10 Vectors**: 将 6 个 V3 化学矢量与 4 个 GVP 矢量结合（总共 10 个矢量）
- ✅ **Separated Feature Categories**: 物理化学、空间标量、空间矢量三类特征分离处理
- ✅ **Increased GVP Capacity**: GVP 矢量隐藏维度从 16 增至 20

### 关键改进/变化
1. **特征分离与专用处理**:
   - 62D 物理化学特征 → PhysChem MLP Encoder
   - 72D 空间标量特征 → GVP node_s
   - 6×3D 空间矢量特征 + 4×3D GVP 矢量 → GVP node_v (总共 10 个矢量)

2. **GVP 架构增强**:
   - `n_gvp_vectors`: 4 → 10 (增加 6 个 V3 化学矢量)
   - `n_gvp_vector_hidden`: 16 → 20 (增加矢量处理能力)
   - Bridge 输出: 136 → 210 维 (20 个矢量通道)

3. **V3 特征的 6 个空间矢量**:
   - Sidechain Base/CoM Vector: 侧链方向
   - Aromatic Normal Vector: 芳香环法向量
   - H-bond Donor/Acceptor Vector: 氢键方向
   - Cavity Pointing Vector: 空腔指向矢量

### 影响范围（改了哪些东西）
- 配置：`configs/pocket_gvp_v8.yaml`
- 模型：`src/models/pocket_gvp_fusion_v8.py`
- 数据集：`src/data/pocket_dataset_v7.py`
- 特征提取：`preparation/protein/extract_complete_features_v3.py`
- 文档：`DOCs/V3_FEATURE_EXTRACTION.md`

### 文件结构
```
configs/
  └── pocket_gvp_v8.yaml                    # V8 模型配置
src/
  ├── models/
  │   └── pocket_gvp_fusion_v8.py          # V8 模型实现
  └── data/
      └── pocket_dataset_v7.py              # V7 数据集（加载 V3 特征）
preparation/protein/
  ├── extract_complete_features_v3.py      # V3 特征提取脚本
  └── test_v3_features.py                  # V3 特征验证脚本
```

### 配置变化
```yaml
model:
  d_phys: 62                        # V8: 物理化学特征（从 94D 降至 62D）
  d_spatial_scalar: 72              # V8: 空间标量特征（新增）
  n_gvp_vectors: 10                 # V8: 10 个矢量（4 GVP + 6 V3）
  n_gvp_vector_hidden: 20           # V8: 矢量隐藏维度（从 16 增至 20）

data:
  protein_features_dir: "complete_features_v3"  # V8: 使用 V3 特征
  ligand_features_dir: "ligand_features_v4"     # 继续使用 V4 配体特征
```

### 使用方法（最小可复现）

#### 1. 提取 V3 蛋白质特征
```bash
# 批量提取整个目录
python preparation/protein/extract_complete_features_v3.py \
  --input data/processed/ \
  --output data/complete_features_v3/ \
  --pattern "*_protein.pdb" \
  --workers 32

# 验证提取的特征
python preparation/protein/test_v3_features.py \
  --input data/complete_features_v3/1981-2000/1a0q/1a0q_protein.pt
```

#### 2. 训练 V8 模型
```bash
# 基础训练
python scripts/train/train_v8.py \
  --config configs/pocket_gvp_v8.yaml

# 多 GPU 训练
torchrun --nproc_per_node=4 scripts/train/train_v8.py \
  --config configs/pocket_gvp_v8.yaml \
  --distributed
```

#### 3. 评估 V8 模型
```bash
python scripts/evaluate.py \
  --config configs/pocket_gvp_v8.yaml \
  --checkpoint outputs/pocket_gvp_fusion_v8/best.pt \
  --split test
```

### 兼容性/迁移说明
- **与 V7 checkpoint 不兼容**：GVP 输入维度不同（4 → 10 vectors）
- **需要重新提取 V3 蛋白质特征**：使用 `extract_complete_features_v3.py`
- **配体特征可复用**：继续使用 V4 配体特征（无需重新提取）
- **距离标签可复用**：继续使用 VDW-based 距离标签

### 已知问题
- ~~训练脚本待创建~~: ✅ `scripts/train/train_v8.py` 已创建
- **参数量增加**: Bridge 维度增大，需要监控过拟合风险
- **计算复杂度增加**: 10 个矢量 vs 4 个矢量，训练速度可能下降

---

## V7m3 - 2026-02-23

### 主要特性
- ✅ **Cluster-Size Inverse Weighting Loss**: 基于 MMseqs2 聚类大小的动态样本权重
- ✅ **Sample-wise Loss Weighting**: 对小簇（孤儿蛋白）施加更高的 Loss 惩罚
- ✅ **Improved Generalization**: 提高模型对低相似度样本的泛化能力
- ✅ **MMseqs2 30% Sequence Identity Clustering**: 使用更严格的数据划分避免数据泄露

### 关键改进/变化

1. **Cluster-Weighted Loss (V7m3)**
   - 权重公式: `weight = 1 / log(cluster_size + ε)`
   - 小簇（cluster_size 小）获得更高权重
   - 大簇（cluster_size 大）获得较低权重
   - 默认 ε = 2.0 防止权重过大

2. **数据划分更新**
   - 使用 `configs/splits/mmseqs_splits_strict.csv`
   - MMseqs2 参数: `--min-seq-id 0.3 -c 0.8`
   - 新增 cluster_size 列用于权重计算
   - 数据集: Train 15229, Val 1903, Test 1905 (Total 19037)

3. **实现细节**
   - 新增 `ClusterWeightedLoss` 类管理样本权重
   - 新增 `load_splits_with_cluster_size()` 函数读取聚类信息
   - 新增 `compute_combined_loss_with_sample_weights()` 函数
   - 向后兼容：如果没有 cluster_size 信息，自动回退到标准 loss

### 影响范围（改了哪些东西）
- 配置：`configs/pocket_gvp_v7.yaml` (split_file 路径更新)
- 代码：
  - `src/data/pocket_dataset.py` (新增 `load_splits_with_cluster_size`)
  - `scripts/train/train_v7m3.py` (新增 ClusterWeightedLoss 类和相关逻辑)
- 数据：`configs/splits/mmseqs_splits_strict.csv` (4列格式)

### 文件结构
```
configs/
  ├── pocket_gvp_v7.yaml          # 配置文件（split_file 更新）
  └── splits/
      └── mmseqs_splits_strict.csv  # 新格式：split\tpdb_id\trepresentative\tcluster_size
src/
  └── data/
      └── pocket_dataset.py        # 新增 load_splits_with_cluster_size()
scripts/
  └── train/
      └── train_v7m3.py            # V7m3 训练脚本
```

### 配置变化
```yaml
data:
  split_file: "configs/splits/mmseqs_splits_strict.csv"  # 更新为 CSV 格式
```

### 使用方法（最小可复现）
```bash
# 单GPU训练
python scripts/train/train_v7m3.py --config configs/pocket_gvp_v7.yaml

# 单GPU + torch.compile
python scripts/train/train_v7m3.py --config configs/pocket_gvp_v7.yaml --compile

# 多GPU分布式训练
torchrun --nproc_per_node=4 scripts/train/train_v7m3.py \
    --config configs/pocket_gvp_v7.yaml --compile
```

### 权重统计示例
```
Cluster-Weighted Loss Statistics:
  Total samples: 19037
  Weight range: [0.2891, 1.4427]
  Mean weight: 0.4523

  Top 5 highest-weighted samples (smallest clusters):
    1xyz: weight=1.4427, cluster_size=1  (孤儿蛋白)
    2abc: weight=1.2345, cluster_size=2
    ...

  Top 5 lowest-weighted samples (largest clusters):
    3ov1: weight=0.2891, cluster_size=2236  (大簇代表)
    ...
```

### 兼容性/迁移说明
- 与 V7m2 checkpoint 兼容（模型结构未变）
- 与 V7 checkpoint 兼容（模型结构未变）
- 需要使用新的 split 文件：`configs/splits/mmseqs_splits_strict.csv`
- 向后兼容：如果使用旧的 split 文件（无 cluster_size），自动回退到标准 loss

### 已知问题
- 无

---
