# 特征提取流程对比分析

**对比对象**:
- **PoC版本**: `/home/husrcf/Code/ProtBind/protein-ligand-PoC/scripts/extract_features.py`
- **PLC版本**: `/home/husrcf/Code/ProtBind/esm2_extract_latents.py`

**分析日期**: 2026-02-06

---

## 执行概览

| 维度 | PoC版本 (extract_features.py) | PLC版本 (esm2_extract_latents.py) |
|------|-------------------------------|-----------------------------------|
| **主要功能** | 一站式提取所有特征 | 仅提取ESM2 embeddings |
| **依赖库** | pypropel (自定义库) | esm (官方库) + biotite |
| **处理模式** | 蛋白质+配体+标签 | 仅蛋白质 |
| **并行支持** | 单进程 | 多GPU多进程 |
| **输出格式** | `.npz` (numpy压缩) | `.pt` (PyTorch) |

---

## 1. 架构设计差异

### PoC版本：一体化流程

```
输入: PDB + Ligand (SDF/MOL2)
  ↓
[1] 读取序列
  ↓
[2] 批量提取ESM-2 embeddings (pypropel)
  ↓
[3] 逐个复合物提取:
    - GVP特征 (coords + vectors)
    - 标量特征 (35维)
    - 配体特征 (64维)
    - 距离标签 (3类)
  ↓
输出: {pdb_id}_features.npz
```

**特点**:
- ✅ 一次运行获得所有特征
- ✅ 特征自动对齐
- ❌ 单进程，速度较慢
- ❌ 依赖自定义pypropel库

### PLC版本：专注ESM提取

```
输入: PDB文件
  ↓
[1] 使用ESM官方API加载蛋白质
  ↓
[2] 提取序列并编码
  ↓
[3] 获取ESM-2 embeddings
  ↓
[4] 对齐到PDB残基索引
  ↓
输出: {pdb_id}_protein.pt
```

**特点**:
- ✅ 多GPU并行处理
- ✅ 使用官方ESM库，更稳定
- ✅ 保留完整残基元数据
- ❌ 需要额外脚本提取其他特征

---

## 2. 链处理方式差异

### PoC版本：简单序列提取

```python
# 第307行
seq = ppstr.read_pdb_sequence(prot_path)
```

**处理方式**:
- 读取整个PDB的序列
- 不区分链
- 简单字符串拼接

**问题**:
- ❌ 多链蛋白质可能混淆
- ❌ 不保留链ID信息
- ❌ 插入码(insertion code)丢失

### PLC版本：精确链对齐

```python
# 第82-147行: _extract_complex()
complex_ = ProteinComplex.from_pdb(pdb_path)
mask = complex_.residue_index != -1  # 去除链断裂位置

# 保留完整元数据
chain_id = [chain_lookup[int(x)] for x in chain_num]
res_idx = complex_.residue_index[mask]
ins_code = complex_.insertion_code[mask]
entity_id = complex_.entity_id[mask]
```

**处理方式**:
- 使用ESM官方ProteinComplex API
- 保留链ID、残基索引、插入码
- 处理链断裂(chain breaks)
- 支持单链模式(`--chain`参数)

**优势**:
- ✅ 精确对齐到PDB残基编号
- ✅ 保留完整拓扑信息
- ✅ 支持复杂蛋白质结构
- ✅ 可追溯每个残基来源

---

## 3. ESM模型加载差异

### PoC版本：通过pypropel封装

```python
# 第105行
model, alphabet, batch_converter = ppesm.load_esm_model(model_name, device=device)

# 第112行
emb_list = ppesm.get_esm_embeddings_batch(seqs, model=model, device=device, batch_size=batch_size)
```

**特点**:
- 使用自定义封装
- 批量处理多个序列
- 简化API

**问题**:
- ❌ 依赖pypropel库
- ❌ 封装可能隐藏细节
- ❌ 更新滞后于官方库

### PLC版本：直接使用官方API

```python
# 第45-62行
def _load_model(device: torch.device, model_name: str):
    if model_name == "esm2_t33_650M_UR50D":
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    elif model_name == "esm2_t36_3B_UR50D":
        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
    # ... 支持5种ESM2模型

    model = model.to(device)
    model.eval()
    return model, alphabet

# 第100-107行
batch_converter = alphabet.get_batch_converter()
data = [("protein", sequence)]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_tokens = batch_tokens.to(device)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[model.num_layers])
    emb = results["representations"][model.num_layers][0, 1:-1]  # 去除BOS/EOS
```

**特点**:
- 直接使用ESM官方API
- 支持5种ESM2模型大小
- 显式去除BOS/EOS token
- 完全控制推理过程

**优势**:
- ✅ 与官方库同步更新
- ✅ 更透明的处理流程
- ✅ 灵活选择模型大小
- ✅ 标准化的输出格式

---

## 4. 保存格式差异

### PoC版本：Numpy压缩格式

```python
# 第221-223行
def save_features(features: Dict, output_path: str):
    """Save features to a compressed numpy file."""
    np.savez_compressed(output_path, **features)
```

**输出文件**: `{pdb_id}_features.npz`

**内容结构**:
```python
{
    'pdb_id': str,
    'esm': np.ndarray,              # [N, 1280]
    'gvp_coords': np.ndarray,       # [N, 3]
    'gvp_vectors': np.ndarray,      # [N, 4, 3]
    'protein_scalar': np.ndarray,   # [N, 35]
    'ligand_coords': np.ndarray,    # [M, 3]
    'ligand_features': np.ndarray,  # [M, 64]
    'labels': np.ndarray,           # [N]
    'distances': np.ndarray,        # [N]
    'n_residues': int,
    'n_ligand_atoms': int
}
```

**特点**:
- ✅ 所有特征在一个文件
- ✅ 压缩存储，节省空间
- ❌ 不保留残基元数据
- ❌ 需要numpy加载

### PLC版本：PyTorch格式

```python
# 第254-255行
data["model_name"] = model_name
torch.save(data, out_path)
```

**输出文件**: `{pdb_id}_protein.pt`

**内容结构**:
```python
{
    'embeddings': torch.Tensor,           # [N, 1280]
    'chain_id': List[str],                # ['A', 'A', 'B', ...]
    'chain_id_numeric': List[int],        # [0, 0, 1, ...]
    'residue_index': List[int],           # [1, 2, 3, ...]
    'insertion_code': List[str],          # ['', '', 'A', ...]
    'residue_name_1': List[str],          # ['M', 'A', 'K', ...]
    'residue_name_3': List[str],          # ['MET', 'ALA', 'LYS', ...]
    'sequence': str,                      # 'MAKL...'
    'entity_id': List[int],               # [1, 1, 2, ...]
    'atom37_positions': torch.Tensor,     # [N, 37, 3]
    'atom37_mask': torch.Tensor,          # [N, 37]
    'pdb_path': str,
    'mode': str,                          # 'complex' or 'chain'
    'model_name': str                     # 'esm2_t33_650M_UR50D'
}
```

**特点**:
- ✅ 保留完整残基元数据
- ✅ 包含atom37坐标(37种原子类型)
- ✅ PyTorch原生格式，训练时无需转换
- ✅ 可追溯到原始PDB
- ❌ 文件较大(未压缩)
- ❌ 只包含蛋白质特征

---

## 5. 并行处理能力

### PoC版本：单进程顺序处理

```python
# 第331-346行
for pdb_id, prot_path, lig_path in tqdm(complexes, desc="Extracting features"):
    out_path = os.path.join(output_dir, f"{pdb_id}_features.npz")

    esm_emb = esm_embeddings.get(pdb_id, None)

    features = extract_features_single(
        pdb_id, prot_path, lig_path,
        esm_embedding=esm_emb,
        use_dssp=not args.no_dssp
    )

    if features is not None:
        save_features(features, out_path)
        success_count += 1
```

**特点**:
- 单进程循环处理
- ESM批量提取后，其他特征逐个处理
- 简单但慢

**性能**:
- 处理19,037个复合物估计需要 **数天**

### PLC版本：多GPU多进程并行

```python
# 第359-402行
if args.devices:
    devices = _parse_devices(args.devices)  # 解析多GPU

    # 文件分片
    splits = [[] for _ in devices]
    for i, pdb_path in enumerate(pdb_files):
        splits[i % len(devices)].append(pdb_path)

    # 启动多进程
    ctx = mp.get_context("spawn")
    procs = []
    for dev, files in zip(devices, splits):
        if not files:
            continue
        p = ctx.Process(
            target=_worker_entry,
            args=(files, input_root, output_root, args.model, dev, ...)
        )
        p.start()
        procs.append(p)

    # 等待所有进程完成
    for p in procs:
        p.join()
```

**特点**:
- 支持多GPU并行
- 文件轮询分配到各GPU
- 独立进程避免GIL限制

**性能**:
- 使用2个GPU: 速度提升 **~2倍**
- 使用4个GPU: 速度提升 **~4倍**
- 处理19,037个文件估计需要 **数小时**

**使用示例**:
```bash
# 单GPU
python esm2_extract_latents.py --input New --output esm2_latents --device cuda:0

# 双GPU并行
python esm2_extract_latents.py --input New --output esm2_latents --devices 0,1

# 四GPU并行
python esm2_extract_latents.py --input New --output esm2_latents --devices 0,1,2,3
```

---

## 6. 错误处理和恢复

### PoC版本：基础错误处理

```python
# 第248-251行
if args.resume:
    remaining = []
    for pdb_id, prot_path, lig_path in complexes:
        out_path = os.path.join(output_dir, f"{pdb_id}_features.npz")
        if not os.path.exists(out_path):
            remaining.append((pdb_id, prot_path, lig_path))
```

**特点**:
- 支持`--resume`跳过已处理文件
- 支持`--overwrite`强制重新处理
- 捕获异常但继续处理

**问题**:
- ❌ 失败文件需要手动重新运行
- ❌ 没有失败日志

### PLC版本：详细错误追踪

```python
# 第237-265行
failures = []

for idx, pdb_path in enumerate(pdb_files, start=1):
    rel = pdb_path.relative_to(input_root)
    out_path = (output_root / rel).with_suffix(".pt")

    if out_path.exists() and not overwrite:
        print(f"[{tag} {idx}/{total}] skip {rel}")
        continue

    try:
        if chain:
            data = _extract_chain(model, alphabet, pdb_path, chain)
        else:
            data = _extract_complex(model, alphabet, pdb_path)
        data["model_name"] = model_name
        torch.save(data, out_path)
        print(f"[{tag} {idx}/{total}] ok   {rel}")
    except Exception as exc:
        failures.append((str(pdb_path), str(exc)))
        print(f"[{tag} {idx}/{total}] fail {rel}: {exc}")

if failures:
    print(f"\nFailures ({tag}):")
    for path, err in failures:
        print(f"- {path}: {err}")
```

**特点**:
- 详细的进度标签(`[cuda:0 123/1000]`)
- 收集所有失败文件和错误信息
- 最后统一打印失败列表
- 保留目录结构

**优势**:
- ✅ 清晰的进度追踪
- ✅ 失败原因可追溯
- ✅ 便于批量重试

---

## 7. 特征维度对比

### PoC版本输出维度

| 特征 | 维度 | 说明 |
|------|------|------|
| ESM embeddings | [N, 1280] | ESM-2 650M模型 |
| Protein scalar | [N, 35] | 35维标量特征 |
| GVP coords | [N, 3] | Cα坐标 |
| GVP vectors | [N, 4, 3] | 4个方向向量 |
| Ligand features | [M, 64] | 19D基础 + 45D全局标签 |
| Ligand coords | [M, 3] | 配体原子坐标 |
| Labels | [N] | 3类距离标签 |
| Distances | [N] | 原始距离值 |

**总计**: 蛋白质 1318维/残基 + 配体 64维/原子

### PLC版本输出维度

| 特征 | 维度 | 说明 |
|------|------|------|
| ESM embeddings | [N, 1280] | ESM-2 650M模型 |
| Atom37 positions | [N, 37, 3] | 37种原子类型坐标 |
| Atom37 mask | [N, 37] | 原子存在性mask |
| 元数据 | - | 链ID、残基索引、插入码等 |

**总计**: 仅ESM embeddings + 原子坐标

---

## 8. 数据流程对比

### PoC版本：完整流程

```
PDBbind数据集
  ↓
extract_features.py (一次运行)
  ↓
{pdb_id}_features.npz (所有特征)
  ↓
直接用于训练
```

**优点**:
- ✅ 一步到位
- ✅ 特征自动对齐

**缺点**:
- ❌ 处理慢
- ❌ 难以更新单个特征

### PLC版本：分步流程

```
PDBbind数据集
  ↓
esm2_extract_latents.py (ESM特征)
  ↓
{pdb_id}_protein.pt

  +

其他脚本 (GVP、配体、标签)
  ↓
complete_features/{pdb_id}_protein.pt
ligand_features/{pdb_id}_ligand.pt
residue_ligand_distances_v2.pt
  ↓
PocketDatasetV3加载
  ↓
训练
```

**优点**:
- ✅ 模块化，易于维护
- ✅ 可独立更新特征
- ✅ 并行处理快

**缺点**:
- ❌ 需要多个脚本
- ❌ 需要手动对齐

---

## 9. 关键代码差异

### 序列提取

**PoC版本**:
```python
# 简单读取
seq = ppstr.read_pdb_sequence(prot_path)
```

**PLC版本**:
```python
# 精确提取并处理
complex_ = ProteinComplex.from_pdb(pdb_path)
mask = complex_.residue_index != -1  # 去除链断裂
seq_list = []
for i, aa_idx in enumerate(complex_.sequence):
    if mask[i]:
        if isinstance(aa_idx, int):
            aa = residue_constants.restypes[aa_idx] if aa_idx < len(residue_constants.restypes) else 'X'
        else:
            aa = str(aa_idx) if aa_idx else 'X'
        seq_list.append(aa)
sequence = "".join(seq_list)
```

### Embedding提取

**PoC版本**:
```python
# 批量处理
emb_list = ppesm.get_esm_embeddings_batch(
    seqs,
    model=model,
    device=device,
    batch_size=batch_size
)
```

**PLC版本**:
```python
# 单个处理，更精确
batch_converter = alphabet.get_batch_converter()
data = [("protein", sequence)]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_tokens = batch_tokens.to(device)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[model.num_layers])
    emb = results["representations"][model.num_layers][0, 1:-1]  # 去除BOS/EOS
```

### 残基名称映射

**PoC版本**: 无

**PLC版本**:
```python
# 从PDB读取真实残基名称
def _load_residue_name_map(pdb_path: Path) -> dict[tuple[str, int, str], str]:
    atom_array = PDBFile.read(pdb_path).get_structure(model=1, extra_fields=["b_factor"])
    atom_array = atom_array[bs.filter_amino_acids(atom_array) & ~atom_array.hetero]

    name_map: dict[tuple[str, int, str], str] = {}
    for res in bs.residue_iter(atom_array):
        first = res[0]
        key = (first.chain_id, int(first.res_id), str(first.ins_code).strip())
        name_map[key] = first.res_name
    return name_map

# 使用映射
res_name_3 = []
for c, r, ins, a in zip(chain_id, res_idx, ins_code, aa):
    key = (c, int(r), str(ins).strip())
    res_name_3.append(name_map.get(key, _fallback_res_name_3(a)))
```

---

## 10. 使用场景建议

### 使用PoC版本的场景

✅ **适合**:
- 小规模数据集 (<1000个复合物)
- 需要快速原型开发
- 所有特征一次性提取
- 不需要精确的残基对齐

❌ **不适合**:
- 大规模数据集 (>10,000个)
- 需要多GPU加速
- 需要精确的链和残基信息
- 需要模块化更新特征

### 使用PLC版本的场景

✅ **适合**:
- 大规模数据集 (10,000+)
- 有多GPU资源
- 需要精确的残基元数据
- 需要与其他特征提取流程集成
- 需要追溯到原始PDB结构

❌ **不适合**:
- 需要一站式提取所有特征
- 没有PyTorch环境
- 只需要简单的序列embeddings

---

## 11. 性能对比估算

### 处理19,037个复合物

| 指标 | PoC版本 | PLC版本 (单GPU) | PLC版本 (4 GPU) |
|------|---------|----------------|----------------|
| ESM提取时间 | ~8小时 | ~8小时 | ~2小时 |
| 其他特征时间 | ~40小时 | N/A | N/A |
| **总时间** | **~48小时** | **~8小时** | **~2小时** |
| 并行能力 | 无 | 单GPU | 4x并行 |
| 内存占用 | 中等 | 高 | 高 |
| 磁盘占用 | ~50GB (压缩) | ~80GB (未压缩) | ~80GB |

**注**: 时间估算基于ESM-2 650M模型，实际时间取决于硬件配置。

---

## 12. 推荐方案

### 当前PLC项目推荐流程

```bash
# 步骤1: 提取ESM2 embeddings (多GPU并行)
python /home/husrcf/Code/ProtBind/esm2_extract_latents.py \
    --input data/raw \
    --output data/esm2_latents \
    --model esm2_t33_650M_UR50D \
    --devices 0,1,2,3 \
    --pattern "*_protein.pdb"

# 步骤2: 提取其他特征 (需要单独脚本)
# - complete_features: GVP + 理化特征
# - ligand_features: 配体特征
# - distances: 距离标签

# 步骤3: 使用PocketDatasetV3加载训练
python scripts/train_v3_optimized.py \
    --config configs/pocket_gvp_v2_nested.yaml \
    --amp
```

**优势**:
- ✅ ESM提取速度快 (多GPU)
- ✅ 保留完整元数据
- ✅ 模块化，易于维护
- ✅ PyTorch原生格式

---

## 总结

### PoC版本特点
- 🎯 **定位**: 快速原型，一站式提取
- ⚡ **速度**: 慢 (单进程)
- 📦 **输出**: 压缩npz，所有特征
- 🔧 **依赖**: pypropel自定义库
- 📊 **元数据**: 最小化

### PLC版本特点
- 🎯 **定位**: 生产级，专注ESM
- ⚡ **速度**: 快 (多GPU并行)
- 📦 **输出**: PyTorch格式，仅ESM
- 🔧 **依赖**: ESM官方库
- 📊 **元数据**: 完整保留

### 关键差异总结

| 维度 | PoC | PLC |
|------|-----|-----|
| 特征完整性 | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| 处理速度 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 元数据保留 | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 并行能力 | ⭐ | ⭐⭐⭐⭐⭐ |
| 易用性 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| 可维护性 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**最终建议**: PLC项目应继续使用当前的分步流程，利用`esm2_extract_latents.py`的多GPU并行能力快速提取ESM特征，然后通过其他脚本补充剩余特征。这种模块化设计更适合大规模生产环境。
