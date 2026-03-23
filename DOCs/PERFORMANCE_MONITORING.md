# V9m2 性能监控使用指南

## 问题诊断

如果你遇到GPU利用率不稳定（50-90%跳动），这通常表明存在数据IO瓶颈。

## 启用性能监控

在配置文件中添加：

```yaml
# configs/pocket_gvp_v9.yaml

training:
  enable_profiling: true  # 启用性能分析
```

## 日志文件位置

性能监控日志会自动保存到每个实验的输出目录下：

```
outputs/
  └── pocket_gvp_fusion_v9_20260302_123456/
      ├── performance_pocket_gvp_fusion_v9_20260302_123456_20260302_123456.log   # 文本日志
      ├── performance_pocket_gvp_fusion_v9_20260302_123456_20260302_123456.json  # JSON日志（用于绘图）
      ├── best_model.pt
      └── ...
```

- **文本日志 (.log)**: 人类可读的性能报告
- **JSON日志 (.json)**: 结构化数据，可用于后续分析和绘图

## 监控输出示例

```
[Performance Monitor] Epoch 1 Batch 100
================================================================================
Timing (ms):
  data_loading        :  125.34 ±  45.23 (min:   80.12, max:  250.45)
  forward             :   85.67 ±  12.34 (min:   70.23, max:  110.56)
  loss_computation    :   15.23 ±   3.45 (min:   12.34, max:   25.67)
  backward            :   95.45 ±  15.67 (min:   75.23, max:  130.45)
  optimizer_step      :   25.67 ±   5.23 (min:   20.12, max:   35.78)

  Total per batch     :  347.36 ms

Time distribution:
  data_loading        :  36.1% ⚠️  (BOTTLENECK!)
  forward             :  24.7%
  loss_computation    :   4.4%
  backward            :  27.5%
  optimizer_step      :   7.4%

GPU Stats:
  Utilization         :  65.3% ±  15.2% (min:  45.0%, max:  90.0%) ⚠️
  Memory              :  18.5 / 24.0 GB (77.1%)

CPU Stats:
  CPU Usage           :  45.2%
  RAM Usage           :  62.3%

================================================================================
🔍 Bottleneck Diagnosis:
  ⚠️  Data loading is slow (36.1% of total time)
  ⚠️  GPU utilization is unstable (std: 15.2%)

💡 Recommendations:
  1. Increase num_workers (current: 24 → try: 32)
  2. Increase prefetch_factor (current: 2 → try: 4)
  3. Enable pin_memory if not already enabled
  4. Consider caching frequently accessed data
  5. Check if disk I/O is the bottleneck (use iostat)
================================================================================
```

## 诊断指标说明

### 1. 时间分布
- **data_loading > 30%**: 数据加载瓶颈
  - 解决方案：增加num_workers、prefetch_factor
- **forward > 40%**: 模型计算瓶颈
  - 解决方案：启用Flash Attention、减小模型、启用AMP
- **backward > 40%**: 反向传播瓶颈
  - 解决方案：启用gradient checkpointing、减小batch size

### 2. GPU利用率
- **平均值 < 70%**: 数据供给不足
- **标准差 > 10%**: 不稳定，通常是IO瓶颈
- **最小值 < 50%**: 严重的数据饥饿

### 3. 显存使用
- **> 90%**: 接近OOM，考虑减小batch size
- **< 50%**: 可以增大batch size提升效率

## 常见问题和解决方案

### 问题1: 数据加载慢 (data_loading > 30%)

**症状**:
```
data_loading: 150ms (40% of total)
GPU utilization: 60% ± 20%
```

**解决方案**:
```yaml
training:
  num_workers: 32        # 从24增加到32
  prefetch_factor: 4     # 从2增加到4
  pin_memory: true       # 确保启用
```

**进阶优化**:
- 使用SSD存储数据
- 预先将数据加载到内存（如果内存足够）
- 使用更快的数据格式（如HDF5）

### 问题2: 前向传播慢 (forward > 40%)

**症状**:
```
forward: 200ms (50% of total)
GPU utilization: 95% (stable)
```

**解决方案**:
```yaml
model:
  use_flash_attn: true   # 启用Flash Attention

training:
  use_amp: true          # 启用混合精度
```

### 问题3: GPU利用率不稳定

**症状**:
```
GPU utilization: 65% ± 25% (min: 30%, max: 95%)
```

**原因**: 数据供给不稳定

**解决方案**:
1. 增加prefetch_factor
2. 增加num_workers
3. 检查磁盘IO: `iostat -x 1`
4. 检查CPU瓶颈: `htop`

### 问题4: 显存不足

**症状**:
```
GPU Memory: 23.5 / 24.0 GB (98%)
CUDA out of memory error
```

**解决方案**:
```yaml
training:
  batch_size: 16         # 从24减小到16

model:
  use_gradient_checkpointing: true  # 启用梯度检查点
```

## 高级诊断

### 使用DataLoaderProfiler

```python
from src.utils.performance_monitor import DataLoaderProfiler

# 在训练前分析DataLoader
profiler = DataLoaderProfiler(train_loader, num_batches=50)
stats = profiler.profile()
profiler.print_report()
```

输出示例:
```
DataLoader Profiling Report
================================================================================
Batches profiled: 50
Total time: 12.5s

Per-batch timing:
  Mean: 250.3ms
  Std:   45.2ms
  Min:  180.5ms
  Max:  350.8ms

Recommendations:
  - Batch time variance is high (std: 45.2ms)
  - Consider increasing num_workers from 24 to 32
  - Current prefetch_factor: 2, try increasing to 4
```

### 使用nvidia-smi监控

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 记录GPU使用历史
nvidia-smi dmon -s pucvmet -c 1000 > gpu_monitor.log
```

### 使用iostat监控磁盘IO

```bash
# 监控磁盘IO
iostat -x 1

# 关注指标:
# - %util: 磁盘利用率 (>80%表示瓶颈)
# - await: 平均等待时间 (>10ms表示慢)
```

## 性能优化检查清单

### 数据加载优化
- [ ] num_workers >= CPU核心数的50%
- [ ] prefetch_factor >= 2
- [ ] pin_memory = true
- [ ] 数据存储在SSD上
- [ ] 使用高效的数据格式

### 模型优化
- [ ] 启用Flash Attention
- [ ] 启用AMP (混合精度)
- [ ] 考虑gradient checkpointing (如果显存紧张)
- [ ] 使用torch.compile (PyTorch 2.0+)

### 训练优化
- [ ] batch size尽可能大 (不OOM的前提下)
- [ ] gradient accumulation (模拟更大batch)
- [ ] 使用DDP多GPU训练

### 系统优化
- [ ] 使用高速存储 (NVMe SSD)
- [ ] 充足的RAM (避免swap)
- [ ] CPU性能足够 (数据预处理)

## 性能基准

### 理想状态
```
GPU Utilization: 90-95% (stable, std < 5%)
Data loading: < 20% of total time
Forward: 30-40% of total time
Backward: 30-40% of total time
Optimizer: < 10% of total time
```

### 需要优化
```
GPU Utilization: < 80% or std > 10%
Data loading: > 30% of total time
```

## 故障排除流程

1. **启用性能监控**
   ```yaml
   training:
     enable_profiling: true
   ```

2. **运行1个epoch，观察输出**

3. **识别瓶颈**
   - 数据加载慢 → 增加num_workers/prefetch_factor
   - GPU利用率低 → 数据IO问题
   - GPU利用率高但慢 → 模型计算问题

4. **应用优化**

5. **重新测试，对比改进**

## 联系支持

如果问题仍未解决，请提供：
- 性能监控输出
- 配置文件
- `nvidia-smi` 输出
- `iostat -x` 输出
- 系统信息 (CPU, RAM, 存储类型)

---

## 分析JSON日志

JSON日志文件包含所有性能数据的结构化记录，可用于绘图和深度分析。

### 读取JSON日志

```python
import json
import matplotlib.pyplot as plt

# 读取日志
with open('outputs/your_exp/performance_*.json', 'r') as f:
    data = json.load(f)

records = data['records']

# 提取数据
epochs = [r['epoch'] for r in records]
gpu_utils = [r['stats']['gpu_util_mean'] for r in records]
data_loading_times = [r['stats']['data_loading_mean'] for r in records]
forward_times = [r['stats']['forward_mean'] for r in records]

# 绘制GPU利用率趋势
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(gpu_utils)
plt.xlabel('Batch')
plt.ylabel('GPU Utilization (%)')
plt.title('GPU Utilization Over Time')
plt.grid(True)

# 绘制各阶段耗时
plt.subplot(1, 2, 2)
plt.plot(data_loading_times, label='Data Loading')
plt.plot(forward_times, label='Forward')
plt.xlabel('Batch')
plt.ylabel('Time (ms)')
plt.title('Stage Timing')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('performance_analysis.png')
```

### 快速分析脚本

创建 `scripts/analyze_performance.py`:

```python
#!/usr/bin/env python3
import json
import sys
from pathlib import Path

def analyze_performance(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    records = data['records']

    # 计算平均值
    avg_gpu_util = sum(r['stats']['gpu_util_mean'] for r in records) / len(records)
    avg_data_loading = sum(r['stats']['data_loading_mean'] for r in records) / len(records)
    avg_forward = sum(r['stats']['forward_mean'] for r in records) / len(records)
    avg_backward = sum(r['stats']['backward_mean'] for r in records) / len(records)

    total_time = avg_data_loading + avg_forward + avg_backward

    print(f"Performance Summary for {json_file}")
    print("=" * 60)
    print(f"Average GPU Utilization: {avg_gpu_util:.1f}%")
    print(f"\nTime Distribution:")
    print(f"  Data Loading: {avg_data_loading:.2f}ms ({avg_data_loading/total_time*100:.1f}%)")
    print(f"  Forward:      {avg_forward:.2f}ms ({avg_forward/total_time*100:.1f}%)")
    print(f"  Backward:     {avg_backward:.2f}ms ({avg_backward/total_time*100:.1f}%)")
    print(f"  Total:        {total_time:.2f}ms")

    # 诊断
    print(f"\nDiagnosis:")
    if avg_gpu_util < 70:
        print("  ⚠️  Low GPU utilization - likely I/O bottleneck")
    if avg_data_loading / total_time > 0.3:
        print("  ⚠️  Data loading is slow (>30% of time)")
    if avg_gpu_util > 85:
        print("  ✓  Good GPU utilization")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <json_file>")
        sys.exit(1)

    analyze_performance(sys.argv[1])
```

使用方法:
```bash
python scripts/analyze_performance.py outputs/your_exp/performance_*.json
```
