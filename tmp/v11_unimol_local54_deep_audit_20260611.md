# V11 UniMol / Local54 Deep Audit - 2026-06-11

## Executive Summary

当前最重要的结论不是继续堆新结构，而是先修正 UniMol 训练/评估口径，然后用统一协议确认主线。

- 当前最强候选仍是 `V11m10 local54_interaction_kv + residue_chem_compat + coattn`，已有单 seed val best 约 `0.5986`，高于 `V11m6n grad060_no_gate` 和 `V11m8 all_light`。
- `V11m11 BiMamba` 没有明显超过 V11m10，dependency gap 健康但性能不突出，应降级为 complexity ablation。
- 当前运行中的 `V11m12 UniMol utilization probe` 仍继承 `use_unimol_ligand_input: true`，所以它不是 pure late-branch-only UniMol。它测试的是 direct atom UniMol input + V11m12 late branch 的合并效果。
- 已确认一个 P0 口径风险：如果未来关闭 direct UniMol input、只开 V11m10/V11m12 late UniMol branch，旧训练/评估代码会静默不给模型真实 UniMol embedding，导致 false-negative。

## Literature Anchor

- Uni-Mol 是 ICLR 2023 的 3D molecular representation framework，包含 molecular model 和 pocket model，并面向 molecular property、binding pose、protein-ligand complex 等任务。Source: https://openreview.net/forum?id=6K2RM6wVqKu
- Uni-Mol official repository 说明 Uni-Mol/Uni-Mol2 是分子表示与 3D/atom-graph-geometry 相关的系列方法。Source: https://github.com/deepmodeling/Uni-Mol
- Uni-Mol2 使用 two-track transformer 来整合 atom-level、graph-level、geometry-level 信息。Source: https://arxiv.org/abs/2406.14969
- FABind 使用 ligand-informed pocket prediction，支持 ligand-to-pocket late routing 的设计动机。Source: https://arxiv.org/abs/2310.06763

这些文献支持“把 UniMol/global 作为 late context/memory/gating 信号，而不是继续粗暴 broadcast 到 dense atom features”的方向；但它们不支持在 UniMol embedding 没有正确传入模型时下负结论。

## P0 Finding: UniMol Late Branch False-Negative Risk

### 现象

旧 `scripts/infer/infer_v11_internal_ligdep.py` 只在以下条件为真时传入 UniMol：

```python
if getattr(model, "use_unimol_ligand_input", False):
    inputs["unimol_embed"] = unimol_embed
```

并且 `main()` 只用同一条件加载 cache：

```python
needs_unimol = bool(getattr(model, "use_unimol_ligand_input", False))
```

这会漏掉：

- `V11m10 use_ligand_context_prior=True` 且 `ligand_context_prior_sources` 包含 `unimol`
- `V11m12 use_unimol_memory_kv`
- `V11m12 use_unimol_group_gate`
- `V11m12 use_unimol_atom_gate`
- `V11m12 use_unimol_residue_compat`

### 训练侧风险

`scripts/train/train_v11m3.py` 主流程旧逻辑只在 `use_unimol_ligand_input` 为真时加载 UniMol cache。若 pure late branch 配置为：

```yaml
use_unimol_ligand_input: false
use_unimol_memory_kv: true
```

旧训练会令 `unimol_cache={}`，`get_batch_unimol()` 返回 `None`，模型内部退化为 zero UniMol。训练不会报错，但 UniMol branch 永远看不到真实 UniMol。

### 已实施修复

- `scripts/infer/infer_v11_internal_ligdep.py`
  - 新增 `model_needs_unimol(model)`
  - `predict_v11_logits()` 改为使用该函数决定是否传 `unimol_embed`
  - `main()` 改为使用该函数决定是否加载 UniMol cache
- `scripts/infer/sweep_v11_class0_bias.py`
  - calibration sweep 复用 `model_needs_unimol(model)`
- `scripts/train/train_v11m3.py`
  - 新增 `config_needs_unimol(model_cfg)`
  - cache loading 覆盖 direct input、V11m10 context prior、V11m12 UniMol branches
- `scripts/train/train_v11m12.py`
  - 日志增加 `direct_unimol=...`，避免误读当前 run 是否 pure late branch

## Current V11m12 Interpretation

当前 `scripts/slurm/run_v11m12_unimol_util_probe.slurm` 继承 `configs/pocket_gvp_v11m6n_hpc.yaml`，而 base config 显式启用：

```yaml
use_unimol_ligand_input: true
d_unimol: 512
d_unimol_ligand_proj: 64
```

Slurm 只关闭了旧 context prior：

```python
model["use_ligand_context_prior"] = False
model["ligand_context_prior_sources"] = []
```

因此当前 V11m12 不是 pure late-branch-only。后续 `zero_unimol_only` drop 代表 direct atom UniMol input + V11m12 late branch 的合并贡献。

## Recommended Next Experiments

### P0: Pure Late-Branch UniMol Warm-Start Probe

动机：当前 V11m12 从 scratch + zero-init late branch，不能充分测试 UniMol 是否有用。更合理的是从 V11m10 best warm-start，冻结 base，只训 UniMol branch。

实现注意：不能简单把 `use_unimol_ligand_input=false` 后再从 V11m10 checkpoint warm-start，因为这会改变 ligand encoder 输入维度，造成 checkpoint shape mismatch。当前脚本采用更稳的兼容实现：保留 `use_unimol_ligand_input=true` 的张量形状，但在 `train_v11m12.py` 中通过 `training.zero_direct_unimol_ligand_input=true` 把 `unimol_ligand_alpha` 清零并冻结。因此 direct atom UniMol path 实际关闭，V11m12 late branch 仍能收到 UniMol embedding。

代码落点：

- `src/models/pocket_gvp_fusion_v11m12.py`
- `scripts/train/train_v11m12.py`
- `scripts/slurm/run_v11m12_unimol_warmstart_probe.slurm`

矩阵：

| variant | direct path | branch | init | freeze |
|---|---:|---|---|---|
| `u_rescompat_lr2e4/lr5e4` | alpha zeroed | residue-UniMol compatibility | V11m10 best | freeze base |
| `u_groupgate_lr2e4/lr5e4` | alpha zeroed | UniMol group gate | V11m10 best | freeze base |
| `u_mem1b2_lr2e4/lr5e4` | alpha zeroed | 1 memory token, score bias -2 | V11m10 best | freeze base |
| `u_mem4b2_lr2e4/lr5e4` | alpha zeroed | 4 memory tokens, score bias -2 | V11m10 best | freeze base |
| `u_res_group_lr2e4/lr5e4` | alpha zeroed | rescompat + groupgate | V11m10 best | freeze base |

判断标准：

- val/test F1 比 V11m10 best 提升 `>=0.003-0.005`
- `zero_unimol_only` drop `>=0.01`
- `Health/unimol_*_abs_mean` 非零且稳定
- `real-shuffle` gap 不低于 `~0.075`

失败判据：

- warm-start/freeze 后 `zero_unimol_only` drop 仍 `<=0.005`
- test real 无提升或低于 seed std
- health stats 非零但 dependency gap 不增，说明只是 calibration/noise

### P1: Confirm Mainline With Unified Test Protocol

需要统一比较：

- `V11m6n grad060_no_gate`
- `V11m8x/all_light/coattn`
- `V11m10 l54kvcompat/coattn`
- `V11m11 bimamba_ligkv2`
- `V11m12 best`

每个候选至少做：

- `real/shuffle/zero`
- `source_split`
- `feature_split`
- `local54_subgroups`
- `class0 calibration` 只作为 post-hoc，不作为结构贡献

判断标准：

- mean test real 提升
- real-shuffle gap 保持或扩大
- local54/graph drop 与 claim 一致
- seed variance 不覆盖提升

### P1: Local54 Mechanism Map

已有 `scripts/infer/internal_ligand_dependency.py` 支持：

```bash
--ligand_mode custom_local54
--custom_local54_ranges 33:36,38:42
--custom_local54_op zero|shuffle|mean|noise
```

建议直接用现有工具扫：

- broad groups: `base33`, `electronic3`, `sasa_hbond4`, `flex_rot5`, `metal4`, `stereo5`
- promising pairs: `electronic3+sasa_hbond4`, `electronic3+flex_rot5`, `sasa_hbond4+flex_rot5`
- single dims: 54 个 local dims 单独 zero/shuffle

目标：把“local54/graph 是主信号”从粗粒度 source split 推进到可解释 feature claim。

### P2: BiMamba / Deeper Architecture

`V11m11 bimamba_ligkv2` dependency gap 健康，但 val/test 未明显超过 V11m10。除非多 seed 超过 V11m10 `>0.005`，不建议继续扩 Mamba depth/memory。

## Files Inspected

- `src/models/pocket_gvp_fusion_v11m6.py`
- `src/models/pocket_gvp_fusion_v11m8.py`
- `src/models/pocket_gvp_fusion_v11m8x.py`
- `src/models/pocket_gvp_fusion_v11m10.py`
- `src/models/pocket_gvp_fusion_v11m11.py`
- `src/models/pocket_gvp_fusion_v11m12.py`
- `scripts/train/train_v11m3.py`
- `scripts/train/train_v11m4.py`
- `scripts/train/train_v11m10.py`
- `scripts/train/train_v11m11.py`
- `scripts/train/train_v11m12.py`
- `scripts/infer/infer_v11_internal_ligdep.py`
- `scripts/infer/internal_ligand_dependency.py`
- `scripts/infer/sweep_v11_class0_bias.py`
- `scripts/slurm/run_v11m12_unimol_util_probe.slurm`
- `scripts/slurm/run_v11m10_soft_compat_probe.slurm`
- `scripts/slurm/run_v11m11_bimamba_probe.slurm`
- `configs/pocket_gvp_v11m6n_hpc.yaml`
- `tmp/local54_wide_9816067/local54_wide_scan_summary.csv`
