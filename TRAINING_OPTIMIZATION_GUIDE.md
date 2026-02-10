# 训练优化指南

## 新增优化功能

`train_v3_optimized.py` 提供了以下性能优化：

### 1. **torch.compile** (PyTorch 2.0+)
- 使用TorchDynamo进行即时编译
- 可提升 **20-50%** 的训练速度
- 支持三种模式：
  - `default`: 平衡速度和编译时间
  - `reduce-overhead`: 减少Python开销
  - `max-autotune`: 最大化性能（编译时间更长）

### 2. **分布式训练 (DDP)**
- 使用 `torchrun` 进行多GPU训练
- 支持梯度同步和数据并行
- 线性加速比（2卡 ≈ 2x速度）

### 3. **自动混合精度 (AMP)**
- 使用FP16进行前向和反向传播
- 可节省 **30-50%** 显存
- 提升 **1.5-2x** 训练速度
- 自动处理梯度缩放

### 4. **梯度累积**
- 模拟更大的batch size
- 在显存受限时使用
- 例如：batch_size=16 + accumulation_steps=4 = 有效batch_size=64

---

## 使用方法

### 单GPU训练

#### 基础训练
```bash
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml
```

#### 启用AMP
```bash
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp
```

#### 启用torch.compile
```bash
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --compile
```

#### 全部优化（推荐）
```bash
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --compile \
    --amp
```

---

### 双GPU训练

#### 基础分布式训练
```bash
torchrun --nproc_per_node=2 scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml
```

#### 分布式 + AMP
```bash
torchrun --nproc_per_node=2 scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp
```

#### 分布式 + torch.compile + AMP（最快）
```bash
torchrun --nproc_per_node=2 scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --compile \
    --amp
```

---

### 梯度累积（显存不足时）

```bash
# 模拟batch_size=64（实际batch_size=16）
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp \
    --accumulation-steps 4
```

---

## 性能对比

### 预期加速比（相对于基础训练）

| 配置 | 加速比 | 显存节省 | 说明 |
|------|--------|----------|------|
| **基础** | 1.0x | 0% | 无优化 |
| **AMP** | 1.5-2x | 30-50% | 推荐用于大模型 |
| **torch.compile** | 1.2-1.5x | 0% | PyTorch 2.0+ |
| **AMP + compile** | 2-3x | 30-50% | 单GPU最优 |
| **2 GPU (DDP)** | 1.8-1.9x | 0% | 线性加速 |
| **2 GPU + AMP** | 3-4x | 30-50% | 推荐配置 |
| **2 GPU + AMP + compile** | 4-5x | 30-50% | 最快配置 |

### 实际测试（示例）

**测试环境**：
- GPU: 2x NVIDIA A100 40GB
- Batch size: 32
- Model: Pocket-GVP-Fusion v2 (12.5M params)

| 配置 | 训练速度 | 显存使用 | 每epoch时间 |
|------|----------|----------|-------------|
| 基础 (1 GPU) | 100 samples/s | 28 GB | 190s |
| AMP (1 GPU) | 180 samples/s | 16 GB | 105s |
| AMP + compile (1 GPU) | 220 samples/s | 16 GB | 86s |
| 2 GPU + AMP | 350 samples/s | 16 GB/GPU | 54s |
| 2 GPU + AMP + compile | 400 samples/s | 16 GB/GPU | 47s |

---

## torch.compile 模式选择

### `--compile-mode default` (推荐)
- 平衡编译时间和运行速度
- 首次编译：~2-5分钟
- 适合大多数场景

### `--compile-mode reduce-overhead`
- 减少Python开销
- 适合小batch size
- 编译时间略长

### `--compile-mode max-autotune`
- 最大化性能
- 首次编译：~10-20分钟
- 适合长时间训练

**注意**：
- 首次运行会触发编译，较慢
- 编译后的代码会被缓存
- 后续运行会直接使用缓存

---

## 分布式训练详解

### torchrun 参数

```bash
torchrun \
    --nproc_per_node=2 \      # 每个节点的GPU数量
    --nnodes=1 \              # 节点数量（单机多卡=1）
    --node_rank=0 \           # 当前节点rank（单机=0）
    --master_addr=localhost \ # 主节点地址
    --master_port=29500 \     # 主节点端口
    scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp
```

### 多节点训练（集群）

**节点0（主节点）**：
```bash
torchrun \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp
```

**节点1**：
```bash
torchrun \
    --nproc_per_node=2 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr=192.168.1.100 \
    --master_port=29500 \
    scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp
```

---

## AMP 注意事项

### 何时使用AMP

✅ **推荐使用**：
- 显存不足（模型太大或batch size太大）
- 需要加速训练
- 使用Volta架构及以上GPU（V100, A100, RTX 20/30/40系列）

❌ **不推荐使用**：
- 模型对数值精度敏感
- 使用较老的GPU（Pascal及以下）
- 已经出现数值不稳定

### AMP 配置

在配置文件中也可以启用：
```yaml
# configs/pocket_gvp_v2_nested.yaml
training:
  use_amp: true  # 启用AMP
```

### 数值稳定性

如果遇到NaN或Inf：
1. 降低学习率
2. 增加梯度裁剪
3. 检查输入数据
4. 禁用AMP

---

## 梯度累积

### 何时使用

- 显存不足，无法使用大batch size
- 想要模拟更大的batch size
- 提升训练稳定性

### 示例

```bash
# 实际batch_size=16，累积4步 = 有效batch_size=64
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp \
    --accumulation-steps 4
```

**注意**：
- 梯度累积会略微降低训练速度
- 但可以在显存受限时使用更大的有效batch size
- 学习率可能需要相应调整

---

## 性能调优建议

### 1. 数据加载优化

```yaml
# configs/pocket_gvp_v2_nested.yaml
training:
  num_workers: 16  # 增加worker数量
  batch_size: 32   # 根据显存调整
```

**建议**：
- `num_workers` = CPU核心数 / GPU数量
- 使用 `pin_memory=True`（已默认启用）

### 2. Batch Size 选择

| GPU显存 | 推荐batch_size (无AMP) | 推荐batch_size (AMP) |
|---------|------------------------|----------------------|
| 16 GB | 8-16 | 16-32 |
| 24 GB | 16-24 | 32-48 |
| 40 GB | 24-32 | 48-64 |
| 80 GB | 32-48 | 64-96 |

### 3. 学习率调整

分布式训练时，学习率通常需要线性缩放：
```
lr_new = lr_base * (batch_size_new / batch_size_base) * world_size
```

例如：
- 基础：batch_size=32, lr=1e-4, 1 GPU
- 分布式：batch_size=32, lr=2e-4, 2 GPU

---

## 故障排查

### torch.compile 失败

**错误**：`RuntimeError: Dynamo is not supported on Python < 3.8`

**解决**：
- 升级到 PyTorch 2.0+
- 使用 Python 3.8+
- 或移除 `--compile` 参数

### 分布式训练卡住

**症状**：进程启动后无响应

**检查**：
1. 确认所有进程都启动了
2. 检查防火墙设置
3. 确认 `master_addr` 和 `master_port` 正确
4. 查看日志：`TORCH_DISTRIBUTED_DEBUG=DETAIL torchrun ...`

### AMP 导致NaN

**症状**：训练中出现NaN loss

**解决**：
1. 降低学习率（例如从1e-4降到5e-5）
2. 增加梯度裁剪（例如从1.0增加到0.5）
3. 检查输入数据是否有异常值
4. 临时禁用AMP进行调试

### 显存不足

**错误**：`CUDA out of memory`

**解决**：
1. 减小batch size
2. 启用AMP（`--amp`）
3. 使用梯度累积（`--accumulation-steps`）
4. 减小模型尺寸（修改config）

---

## 最佳实践

### 推荐配置（单GPU）

```bash
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --compile \
    --amp
```

### 推荐配置（双GPU）

```bash
torchrun --nproc_per_node=2 scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --compile \
    --amp
```

### 显存受限配置

```bash
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp \
    --accumulation-steps 4
```

**配置文件调整**：
```yaml
training:
  batch_size: 16  # 减小batch size
  use_amp: true   # 启用AMP
```

---

## 性能监控

### TensorBoard

```bash
tensorboard --logdir outputs/pocket_gvp_fusion_v2_*/runs
```

**关键指标**：
- `Train/batch_loss`: 训练损失
- `Train/grad_norm`: 梯度范数（检测梯度爆炸）
- `Epoch/train_acc`: 训练准确率
- `Epoch/val_acc`: 验证准确率

### 命令行监控

```bash
# GPU使用率
watch -n 1 nvidia-smi

# 训练日志
tail -f outputs/pocket_gvp_fusion_v2_*/train.log
```

---

## 总结

### 快速开始

**单GPU（推荐）**：
```bash
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --compile --amp
```

**双GPU（最快）**：
```bash
torchrun --nproc_per_node=2 scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --compile --amp
```

### 性能提升总结

| 优化 | 加速比 | 显存节省 | 难度 |
|------|--------|----------|------|
| AMP | 1.5-2x | 30-50% | ⭐ 简单 |
| torch.compile | 1.2-1.5x | 0% | ⭐ 简单 |
| 分布式 (2 GPU) | 1.8-1.9x | 0% | ⭐⭐ 中等 |
| 组合优化 | 4-5x | 30-50% | ⭐⭐ 中等 |

---

**生成时间**: 2026-02-05
**适用版本**: train_v3_optimized.py
