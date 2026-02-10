# V5 Resume 功能改进总结

## ✅ 已完成的改进

### 1. 实验目录管理

**修改前**:
- 总是创建新的带时间戳的目录
- Resume 时会创建新目录，导致文件分散

**修改后**:
```python
def setup_experiment(config: Dict, rank: int, resume_path: str = None) -> Path:
    # If resuming, use the original experiment directory
    if resume_path:
        resume_path = Path(resume_path)
        if resume_path.exists():
            exp_dir = resume_path.parent  # 使用原始目录
            return exp_dir

    # Create new experiment directory (only for new training)
    exp_dir = Path(config['experiment']['output_dir']) / f"{exp_name}_{timestamp}"
    ...
```

**效果**:
- ✅ Resume 时继续在原始目录运行
- ✅ 所有 checkpoint 保存在同一目录
- ✅ TensorBoard 日志在同一目录（覆盖模式）

---

### 2. Checkpoint 保存策略

**三种 Checkpoint**:

1. **Latest** (`checkpoint_latest.pt`)
   - 每个 epoch 后保存
   - 用于意外中断后快速恢复

2. **Periodic** (`checkpoint_epoch_N.pt`)
   - 每 N 个 epoch 保存一次
   - 用于回溯到特定 epoch

3. **Best** (`best_model.pt`)
   - 验证集最佳性能时保存
   - 用于最终评估和部署

**保存内容**:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),  # AMP
    'best_val_loss': best_val_loss,
    'patience_counter': patience_counter,
    'train_metrics': train_metrics,
    'val_metrics': val_metrics,
    'config': config
}
```

---

### 3. Resume 恢复内容

**完整恢复**:
- ✅ 模型权重
- ✅ 优化器状态（包括动量）
- ✅ 学习率调度器状态
- ✅ AMP scaler 状态
- ✅ 训练状态（epoch、best_val_loss、patience_counter）
- ✅ 实验目录（继续在原目录）

**代码**:
```python
if args.resume:
    checkpoint = torch.load(args.resume, map_location=device)

    # Load model
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load optimizer
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Load AMP scaler
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

    # Load training state
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    patience_counter = checkpoint.get('patience_counter', 0)
```

---

### 4. TensorBoard 行为

**覆盖模式**（按用户要求）:
- Resume 时 TensorBoard 会覆盖原有日志
- 如果需要保留历史，用户可以手动复制文件夹

**优点**:
- ✅ 简单直接
- ✅ 避免日志混乱
- ✅ 用户可以通过复制文件夹来保存历史

---

## 📁 目录结构示例

### 新训练
```
outputs/
└── pocket_gvp_fusion_v5_20260208_143022/
    ├── config.yaml
    ├── splits.txt
    ├── checkpoint_latest.pt          # 最新 checkpoint
    ├── checkpoint_epoch_5.pt         # 周期性备份
    ├── checkpoint_epoch_10.pt
    ├── best_model.pt                 # 最佳模型
    └── runs/                         # TensorBoard 日志
        └── events.out.tfevents...
```

### Resume 后
```
outputs/
└── pocket_gvp_fusion_v5_20260208_143022/  # 同一目录
    ├── config.yaml
    ├── splits.txt
    ├── checkpoint_latest.pt          # 更新
    ├── checkpoint_epoch_5.pt
    ├── checkpoint_epoch_10.pt
    ├── checkpoint_epoch_15.pt        # 新增
    ├── best_model.pt                 # 可能更新
    └── runs/                         # TensorBoard 覆盖
        └── events.out.tfevents...
```

---

## 🚀 使用方法

### 1. 正常训练
```bash
python scripts/train_v5.py --config configs/pocket_gvp_v5.yaml --amp
```

### 2. 从 latest checkpoint 恢复
```bash
python scripts/train_v5.py \
    --config configs/pocket_gvp_v5.yaml \
    --resume outputs/pocket_gvp_fusion_v5_20260208_143022/checkpoint_latest.pt \
    --amp
```

### 3. 从特定 epoch 恢复
```bash
python scripts/train_v5.py \
    --config configs/pocket_gvp_v5.yaml \
    --resume outputs/pocket_gvp_fusion_v5_20260208_143022/checkpoint_epoch_50.pt \
    --amp
```

### 4. 从 best model 继续训练
```bash
python scripts/train_v5.py \
    --config configs/pocket_gvp_v5.yaml \
    --resume outputs/pocket_gvp_fusion_v5_20260208_143022/best_model.pt \
    --amp
```

---

## ✅ Resume 功能验证清单

- [x] Resume 时使用原始实验目录
- [x] 不创建新的时间戳目录
- [x] 恢复模型权重
- [x] 恢复优化器状态（包括动量）
- [x] 恢复学习率调度器
- [x] 恢复 AMP scaler
- [x] 恢复训练状态（epoch、best_val_loss、patience）
- [x] Checkpoint 保存在原目录
- [x] TensorBoard 在原目录（覆盖模式）
- [x] 打印恢复信息（epoch、lr、best_val_loss）

---

## 💡 最佳实践

### 1. 定期保存实验
如果想保留训练历史，在 resume 前复制整个实验目录：
```bash
cp -r outputs/pocket_gvp_fusion_v5_20260208_143022 \
      outputs/pocket_gvp_fusion_v5_20260208_143022_backup
```

### 2. 使用 latest checkpoint
意外中断后快速恢复：
```bash
--resume outputs/exp_xxx/checkpoint_latest.pt
```

### 3. 回溯到特定 epoch
如果发现过拟合，可以回到之前的 checkpoint：
```bash
--resume outputs/exp_xxx/checkpoint_epoch_50.pt
```

### 4. 调整超参数后继续训练
从 best model 开始，使用新的学习率：
```bash
# 修改 config.yaml 中的 learning_rate
--resume outputs/exp_xxx/best_model.pt
```

---

## 🎯 总结

Resume 功能现在完全符合要求：
- ✅ 继续在原目录运行
- ✅ 完整恢复训练状态
- ✅ TensorBoard 覆盖模式
- ✅ 简单易用

V5 已经完全准备好训练了！🎉
