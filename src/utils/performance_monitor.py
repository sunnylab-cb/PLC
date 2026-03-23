#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能监控工具 - 用于诊断训练瓶颈

监控指标：
- 数据加载时间
- 前向传播时间
- 反向传播时间
- 优化器更新时间
- GPU利用率
- 显存使用
"""

import time
import torch
import psutil
from collections import defaultdict, deque
from typing import Optional, Dict
from pathlib import Path
import numpy as np
import json
from datetime import datetime

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class PerformanceMonitor:
    """
    性能监控器

    使用方法:
    monitor = PerformanceMonitor(window_size=100)

    for batch in dataloader:
        monitor.start('data_loading')
        # ... 数据处理
        monitor.end('data_loading')

        monitor.start('forward')
        # ... 前向传播
        monitor.end('forward')

        if monitor.should_log(batch_idx):
            monitor.log_stats()
    """

    def __init__(self, window_size: int = 100, log_every: int = 10, device_id: int = 0):
        """
        Args:
            window_size: 滑动窗口大小（用于计算平均值）
            log_every: 每多少个batch打印一次统计
            device_id: GPU设备ID
        """
        self.window_size = window_size
        self.log_every = log_every
        self.device_id = device_id

        # 计时器
        self.timers = defaultdict(lambda: {'start': 0, 'total': 0, 'count': 0})
        self.timing_history = defaultdict(lambda: deque(maxlen=window_size))

        # GPU监控
        self.gpu_util_history = deque(maxlen=window_size)
        self.gpu_mem_history = deque(maxlen=window_size)

        # 初始化NVML
        self.nvml_initialized = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.nvml_initialized = True
            except Exception as e:
                print(f"Warning: Failed to initialize NVML: {e}")

        # 批次计数
        self.batch_count = 0

        # 日志文件
        self.log_file = None
        self.json_log_file = None

    def set_log_file(self, log_dir: str, exp_name: str = "training"):
        """
        设置日志文件路径

        Args:
            log_dir: 日志目录
            exp_name: 实验名称
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 文本日志
        self.log_file = log_dir / f"performance_{exp_name}_{timestamp}.log"

        # JSON日志（用于绘图分析）
        self.json_log_file = log_dir / f"performance_{exp_name}_{timestamp}.json"

        # 初始化JSON日志
        with open(self.json_log_file, 'w') as f:
            json.dump({"metadata": {
                "experiment": exp_name,
                "timestamp": timestamp,
                "device_id": self.device_id
            }, "records": []}, f)

        print(f"Performance logs will be saved to:")
        print(f"  Text: {self.log_file}")
        print(f"  JSON: {self.json_log_file}")

    def start(self, name: str):
        """开始计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timers[name]['start'] = time.time()

    def end(self, name: str):
        """结束计时"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.timers[name]['start']
        self.timers[name]['total'] += elapsed
        self.timers[name]['count'] += 1
        self.timing_history[name].append(elapsed)

    def record_gpu_stats(self):
        """记录GPU统计信息"""
        if self.nvml_initialized:
            try:
                # GPU利用率
                util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
                self.gpu_util_history.append(util.gpu)

                # 显存使用
                mem = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                mem_used_gb = mem.used / 1024**3
                mem_total_gb = mem.total / 1024**3
                self.gpu_mem_history.append((mem_used_gb, mem_total_gb))
            except Exception as e:
                pass
        elif torch.cuda.is_available():
            # Fallback to PyTorch
            mem_allocated = torch.cuda.memory_allocated(self.device_id) / 1024**3
            mem_reserved = torch.cuda.memory_reserved(self.device_id) / 1024**3
            self.gpu_mem_history.append((mem_allocated, mem_reserved))

    def step(self):
        """每个batch后调用"""
        self.batch_count += 1
        self.record_gpu_stats()

    def should_log(self) -> bool:
        """是否应该打印日志"""
        return self.batch_count % self.log_every == 0

    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = {}

        # 计算各阶段平均时间
        for name, history in self.timing_history.items():
            if len(history) > 0:
                stats[f'{name}_mean'] = np.mean(history) * 1000  # ms
                stats[f'{name}_std'] = np.std(history) * 1000
                stats[f'{name}_min'] = np.min(history) * 1000
                stats[f'{name}_max'] = np.max(history) * 1000

        # GPU统计
        if len(self.gpu_util_history) > 0:
            stats['gpu_util_mean'] = np.mean(self.gpu_util_history)
            stats['gpu_util_std'] = np.std(self.gpu_util_history)
            stats['gpu_util_min'] = np.min(self.gpu_util_history)
            stats['gpu_util_max'] = np.max(self.gpu_util_history)

        if len(self.gpu_mem_history) > 0:
            mem_used = [m[0] for m in self.gpu_mem_history]
            stats['gpu_mem_mean'] = np.mean(mem_used)
            stats['gpu_mem_max'] = np.max(mem_used)
            if self.gpu_mem_history[0][1] > 0:  # total available
                stats['gpu_mem_total'] = self.gpu_mem_history[0][1]

        # CPU统计
        stats['cpu_percent'] = psutil.cpu_percent()
        stats['ram_percent'] = psutil.virtual_memory().percent

        return stats

    def log_stats(self, epoch: Optional[int] = None, batch_idx: Optional[int] = None):
        """打印统计信息并写入日志文件"""
        stats = self.get_stats()

        header = f"[Performance Monitor]"
        if epoch is not None:
            header += f" Epoch {epoch}"
        if batch_idx is not None:
            header += f" Batch {batch_idx}"

        # 构建日志内容
        log_lines = []
        log_lines.append(f"\n{header}")
        log_lines.append("=" * 80)

        # 时间统计
        log_lines.append("Timing (ms):")
        timing_keys = [k for k in stats.keys() if k.endswith('_mean') and not k.startswith('gpu')]
        for key in sorted(timing_keys):
            name = key.replace('_mean', '')
            mean = stats.get(f'{name}_mean', 0)
            std = stats.get(f'{name}_std', 0)
            min_val = stats.get(f'{name}_min', 0)
            max_val = stats.get(f'{name}_max', 0)
            log_lines.append(f"  {name:20s}: {mean:7.2f} ± {std:6.2f} (min: {min_val:7.2f}, max: {max_val:7.2f})")

        # 总时间和占比
        total_time = sum(stats.get(f'{k}_mean', 0) for k in timing_keys)
        if total_time > 0:
            log_lines.append(f"\n  {'Total per batch':20s}: {total_time:7.2f} ms")
            log_lines.append(f"\nTime distribution:")
            for key in sorted(timing_keys):
                name = key.replace('_mean', '')
                mean = stats.get(f'{name}_mean', 0)
                pct = (mean / total_time) * 100
                log_lines.append(f"  {name:20s}: {pct:5.1f}%")

        # GPU统计
        if 'gpu_util_mean' in stats:
            log_lines.append(f"\nGPU Utilization:")
            log_lines.append(f"  Mean: {stats['gpu_util_mean']:.1f}%")
            log_lines.append(f"  Std:  {stats['gpu_util_std']:.1f}%")
            log_lines.append(f"  Min:  {stats['gpu_util_min']:.1f}%")
            log_lines.append(f"  Max:  {stats['gpu_util_max']:.1f}%")

            # 利用率诊断
            if stats['gpu_util_mean'] < 70:
                log_lines.append(f"  ⚠️  WARNING: Low GPU utilization! Possible bottlenecks:")
                if stats.get('data_loading_mean', 0) > stats.get('forward_mean', 0):
                    log_lines.append(f"     - Data loading is slower than forward pass")
                    log_lines.append(f"     - Suggestion: Increase num_workers or use faster storage")
                if stats.get('forward_mean', 0) + stats.get('backward_mean', 0) < total_time * 0.5:
                    log_lines.append(f"     - GPU computation time < 50% of total time")
                    log_lines.append(f"     - Suggestion: Increase batch size or use gradient accumulation")

        if 'gpu_mem_mean' in stats:
            log_lines.append(f"\nGPU Memory (GB):")
            log_lines.append(f"  Used: {stats['gpu_mem_mean']:.2f} / {stats.get('gpu_mem_total', 0):.2f}")
            log_lines.append(f"  Peak: {stats['gpu_mem_max']:.2f}")
            if stats.get('gpu_mem_total', 0) > 0:
                util_pct = (stats['gpu_mem_mean'] / stats['gpu_mem_total']) * 100
                log_lines.append(f"  Utilization: {util_pct:.1f}%")

        # CPU/RAM
        log_lines.append(f"\nSystem:")
        log_lines.append(f"  CPU: {stats['cpu_percent']:.1f}%")
        log_lines.append(f"  RAM: {stats['ram_percent']:.1f}%")

        log_lines.append("=" * 80)

        # 打印到控制台
        log_text = '\n'.join(log_lines)
        print(log_text)

        # 写入文本日志
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(log_text + '\n')

        # 写入JSON日志
        if self.json_log_file:
            record = {
                'timestamp': datetime.now().isoformat(),
                'epoch': epoch,
                'batch_idx': batch_idx,
                'stats': stats
            }
            # 读取现有数据
            with open(self.json_log_file, 'r') as f:
                data = json.load(f)
            # 添加新记录
            data['records'].append(record)
            # 写回
            with open(self.json_log_file, 'w') as f:
                json.dump(data, f, indent=2)

    def reset(self):
        """重置所有统计"""
        for timer in self.timers.values():
            timer['total'] = 0
            timer['count'] = 0
        for history in self.timing_history.values():
            history.clear()
        self.gpu_util_history.clear()
        self.gpu_mem_history.clear()
        self.batch_count = 0

    def diagnose_bottleneck(self):
        """诊断性能瓶颈并给出建议"""
        stats = self.get_stats()

        if not stats:
            return

        print("\n" + "=" * 80)
        print("🔍 Bottleneck Diagnosis:")
        print("=" * 80)

        # 计算总时间
        timing_keys = [k for k in stats.keys() if k.endswith('_mean') and not k.startswith('gpu')]
        total_time = sum(stats.get(f'{k}_mean', 0) for k in timing_keys)

        if total_time == 0:
            return

        # 诊断数据加载
        data_loading_time = stats.get('data_loading_mean', 0)
        data_loading_pct = (data_loading_time / total_time) * 100
        if data_loading_pct > 30:
            print(f"  ⚠️  Data loading is slow ({data_loading_pct:.1f}% of total time)")
            print(f"     Current: {data_loading_time:.2f}ms")
            print(f"     💡 Recommendations:")
            print(f"        - Increase num_workers (try +25%)")
            print(f"        - Increase prefetch_factor (try 4 or 6)")
            print(f"        - Check disk I/O with: iostat -x 1")
            print(f"        - Consider using SSD storage")

        # 诊断GPU利用率
        if 'gpu_util_mean' in stats:
            gpu_util = stats['gpu_util_mean']
            gpu_util_std = stats['gpu_util_std']

            if gpu_util < 70:
                print(f"  ⚠️  Low GPU utilization ({gpu_util:.1f}%)")
                print(f"     💡 Recommendations:")
                print(f"        - Likely I/O bottleneck (see data loading above)")
                print(f"        - Increase batch size if memory allows")
                print(f"        - Use gradient accumulation")

            if gpu_util_std > 15:
                print(f"  ⚠️  GPU utilization is unstable (std: {gpu_util_std:.1f}%)")
                print(f"     💡 Recommendations:")
                print(f"        - Data supply is inconsistent")
                print(f"        - Increase prefetch_factor")
                print(f"        - Check for CPU bottleneck with: htop")

            if gpu_util > 85 and gpu_util_std < 10:
                print(f"  ✓  Good GPU utilization ({gpu_util:.1f}% ± {gpu_util_std:.1f}%)")

        # 诊断前向传播
        forward_time = stats.get('forward_mean', 0)
        forward_pct = (forward_time / total_time) * 100
        if forward_pct > 40:
            print(f"  ⚠️  Forward pass is slow ({forward_pct:.1f}% of total time)")
            print(f"     💡 Recommendations:")
            print(f"        - Enable Flash Attention (use_flash_attn: true)")
            print(f"        - Enable AMP (use_amp: true)")
            print(f"        - Consider model optimization")

        # 诊断显存
        if 'gpu_mem_mean' in stats and 'gpu_mem_total' in stats:
            mem_used = stats['gpu_mem_mean']
            mem_total = stats['gpu_mem_total']
            mem_pct = (mem_used / mem_total) * 100

            if mem_pct > 90:
                print(f"  ⚠️  High memory usage ({mem_pct:.1f}%)")
                print(f"     💡 Recommendations:")
                print(f"        - Reduce batch size")
                print(f"        - Enable gradient checkpointing")

            if mem_pct < 50:
                print(f"  💡 Memory underutilized ({mem_pct:.1f}%)")
                print(f"     - Consider increasing batch size for better efficiency")

        print("=" * 80 + "\n")

    def __del__(self):
        """清理NVML"""
        if self.nvml_initialized:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class DataLoaderProfiler:
    """
    DataLoader性能分析器

    专门用于分析数据加载瓶颈
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.fetch_times = deque(maxlen=window_size)
        self.last_fetch_time = None

    def on_batch_start(self):
        """Batch开始前调用"""
        self.last_fetch_time = time.time()

    def on_batch_fetched(self):
        """Batch获取后调用"""
        if self.last_fetch_time is not None:
            elapsed = time.time() - self.last_fetch_time
            self.fetch_times.append(elapsed)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        if len(self.fetch_times) == 0:
            return {}

        times_ms = np.array(self.fetch_times) * 1000
        return {
            'fetch_mean': np.mean(times_ms),
            'fetch_std': np.std(times_ms),
            'fetch_min': np.min(times_ms),
            'fetch_max': np.max(times_ms),
            'fetch_p50': np.percentile(times_ms, 50),
            'fetch_p95': np.percentile(times_ms, 95),
            'fetch_p99': np.percentile(times_ms, 99)
        }

    def log_stats(self):
        """打印统计信息"""
        stats = self.get_stats()
        if not stats:
            return

        print("\n[DataLoader Profiler]")
        print("=" * 60)
        print(f"Batch fetch time (ms):")
        print(f"  Mean:    {stats['fetch_mean']:.2f}")
        print(f"  Std:     {stats['fetch_std']:.2f}")
        print(f"  Min:     {stats['fetch_min']:.2f}")
        print(f"  Max:     {stats['fetch_max']:.2f}")
        print(f"  Median:  {stats['fetch_p50']:.2f}")
        print(f"  P95:     {stats['fetch_p95']:.2f}")
        print(f"  P99:     {stats['fetch_p99']:.2f}")

        # 诊断
        if stats['fetch_max'] > stats['fetch_mean'] * 3:
            print(f"\n⚠️  WARNING: High variance in fetch time!")
            print(f"   Some batches take {stats['fetch_max']/stats['fetch_mean']:.1f}x longer than average")
            print(f"   Suggestion: Check for I/O bottlenecks or slow samples")

        print("=" * 60)
