# 损坏ESM2文件重新生成指南

## 概述

发现 **134个** ESM2特征文件损坏（EOFError），需要重新生成。

**损坏原因**: 文件在生成过程中被截断或写入不完整

**影响范围**: 仅 `2011-2019` 年份段

---

## 损坏文件完整列表

### 按PDB ID分组（134个）

```
4z0q 4z1e 4z1k 4z1q 4z22 4z2g 4z2i 4z2k 4z2o 4z3v
4z5w 4z6h 4z7f 4z7i 4z7o 4z83 4z88 4z8a 4z90 4z9l
4zae 4zb6 4zba 4zbi 4zcw 4ze6 4zec 4zei 4zfi 4zg7
4zga 4zh2 4zh4 4zhm 4zim 4zj8 4zji 4zjr 4zk5 4zl4
4zlo 4zly 4zme 4znx 4zom 4zpe 4zpg 4zqt 4zs0 4zs3
4zsa 4zsj 5ehr 5ehw 5ei2 5ei4 5ei8 5eif 5eis 5eje
5ejv 5ek0 5ekh 5ekm 5eko 5el2 5eld 5elq 5elw 5elz
5em6 5em8 5ema 5emj 5eml 5en3 5ene 5eng 5enk 5eob
5eok 5eom 5ep7 5epl 5epp 5eps 5eq0 5eqe 5eqq 5er4
5erg 5esq 5etb 5eth 5etk 5etm 5eto 5etq 5ets 5etu
5etx 5eu1 5eue 5evb 5evk 5ew0 5ew9 5ewh 5ewk 5ewy
5exl 5exn 5ey0 5ey8 5eyc 5eyk 5eyr 5ezg 5ezx 5f00
5f02 5f0f 5f1c 5f1j 5f1r 5f1v 5f1z 5f25 5f2f 5f2p
5f2s 5f2w 5f37 5f3c
```

### 完整文件路径

所有文件位于: `data/esm2_latents/2011-2019/{pdb_id}/{pdb_id}_protein.pt`

例如:
```
data/esm2_latents/2011-2019/4z0q/4z0q_protein.pt
data/esm2_latents/2011-2019/4z1e/4z1e_protein.pt
...
data/esm2_latents/2011-2019/5f3c/5f3c_protein.pt
```

---

## 重新生成方法

### 方法1: 使用批处理脚本（推荐）

创建 `regenerate_corrupted_esm2.sh`:

```bash
#!/bin/bash

# 损坏的PDB IDs
CORRUPTED_IDS=(
    4z0q 4z1e 4z1k 4z1q 4z22 4z2g 4z2i 4z2k 4z2o 4z3v
    4z5w 4z6h 4z7f 4z7i 4z7o 4z83 4z88 4z8a 4z90 4z9l
    4zae 4zb6 4zba 4zbi 4zcw 4ze6 4zec 4zei 4zfi 4zg7
    4zga 4zh2 4zh4 4zhm 4zim 4zj8 4zji 4zjr 4zk5 4zl4
    4zlo 4zly 4zme 4znx 4zom 4zpe 4zpg 4zqt 4zs0 4zs3
    4zsa 4zsj 5ehr 5ehw 5ei2 5ei4 5ei8 5eif 5eis 5eje
    5ejv 5ek0 5ekh 5ekm 5eko 5el2 5eld 5elq 5elw 5elz
    5em6 5em8 5ema 5emj 5eml 5en3 5ene 5eng 5enk 5eob
    5eok 5eom 5ep7 5epl 5epp 5eps 5eq0 5eqe 5eqq 5er4
    5erg 5esq 5etb 5eth 5etk 5etm 5eto 5etq 5ets 5etu
    5etx 5eu1 5eue 5evb 5evk 5ew0 5ew9 5ewh 5ewk 5ewy
    5exl 5exn 5ey0 5ey8 5eyc 5eyk 5eyr 5ezg 5ezx 5f00
    5f02 5f0f 5f1c 5f1j 5f1r 5f1v 5f1z 5f25 5f2f 5f2p
    5f2s 5f2w 5f37 5f3c
)

# 配置
PDBBIND_DIR="/path/to/pdbbind"  # 修改为实际路径
OUTPUT_DIR="data/esm2_latents"
LOG_FILE="regeneration.log"

# 计数器
TOTAL=${#CORRUPTED_IDS[@]}
SUCCESS=0
FAILED=0

echo "开始重新生成 $TOTAL 个损坏的ESM2文件..." | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE

# 处理每个PDB ID
for i in "${!CORRUPTED_IDS[@]}"; do
    pdb_id="${CORRUPTED_IDS[$i]}"
    progress=$((i + 1))

    echo "[$progress/$TOTAL] 处理 $pdb_id..." | tee -a $LOG_FILE

    # 删除损坏的文件
    corrupted_file="$OUTPUT_DIR/2011-2019/$pdb_id/${pdb_id}_protein.pt"
    if [ -f "$corrupted_file" ]; then
        echo "  删除损坏文件: $corrupted_file" | tee -a $LOG_FILE
        rm "$corrupted_file"
    fi

    # 重新生成
    if python scripts/extract_esm2_features.py \
        --pdb_id "$pdb_id" \
        --input_dir "$PDBBIND_DIR" \
        --output_dir "$OUTPUT_DIR" \
        >> $LOG_FILE 2>&1; then

        echo "  ✓ 成功" | tee -a $LOG_FILE
        SUCCESS=$((SUCCESS + 1))
    else
        echo "  ✗ 失败" | tee -a $LOG_FILE
        FAILED=$((FAILED + 1))
    fi

    echo "" | tee -a $LOG_FILE
done

echo "完成时间: $(date)" | tee -a $LOG_FILE
echo "" | tee -a $LOG_FILE
echo "总结:" | tee -a $LOG_FILE
echo "  总数: $TOTAL" | tee -a $LOG_FILE
echo "  成功: $SUCCESS" | tee -a $LOG_FILE
echo "  失败: $FAILED" | tee -a $LOG_FILE
```

**使用方法**:
```bash
chmod +x regenerate_corrupted_esm2.sh
./regenerate_corrupted_esm2.sh
```

---

### 方法2: 使用Python脚本

创建 `regenerate_corrupted_esm2.py`:

```python
#!/usr/bin/env python3
"""
重新生成损坏的ESM2特征文件
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

# 损坏的PDB IDs
CORRUPTED_IDS = [
    '4z0q', '4z1e', '4z1k', '4z1q', '4z22', '4z2g', '4z2i', '4z2k', '4z2o', '4z3v',
    '4z5w', '4z6h', '4z7f', '4z7i', '4z7o', '4z83', '4z88', '4z8a', '4z90', '4z9l',
    '4zae', '4zb6', '4zba', '4zbi', '4zcw', '4ze6', '4zec', '4zei', '4zfi', '4zg7',
    '4zga', '4zh2', '4zh4', '4zhm', '4zim', '4zj8', '4zji', '4zjr', '4zk5', '4zl4',
    '4zlo', '4zly', '4zme', '4znx', '4zom', '4zpe', '4zpg', '4zqt', '4zs0', '4zs3',
    '4zsa', '4zsj', '5ehr', '5ehw', '5ei2', '5ei4', '5ei8', '5eif', '5eis', '5eje',
    '5ejv', '5ek0', '5ekh', '5ekm', '5eko', '5el2', '5eld', '5elq', '5elw', '5elz',
    '5em6', '5em8', '5ema', '5emj', '5eml', '5en3', '5ene', '5eng', '5enk', '5eob',
    '5eok', '5eom', '5ep7', '5epl', '5epp', '5eps', '5eq0', '5eqe', '5eqq', '5er4',
    '5erg', '5esq', '5etb', '5eth', '5etk', '5etm', '5eto', '5etq', '5ets', '5etu',
    '5etx', '5eu1', '5eue', '5evb', '5evk', '5ew0', '5ew9', '5ewh', '5ewk', '5ewy',
    '5exl', '5exn', '5ey0', '5ey8', '5eyc', '5eyk', '5eyr', '5ezg', '5ezx', '5f00',
    '5f02', '5f0f', '5f1c', '5f1j', '5f1r', '5f1v', '5f1z', '5f25', '5f2f', '5f2p',
    '5f2s', '5f2w', '5f37', '5f3c'
]

def main():
    # 配置
    pdbbind_dir = "/path/to/pdbbind"  # 修改为实际路径
    output_dir = "data/esm2_latents"
    log_file = "regeneration.log"

    total = len(CORRUPTED_IDS)
    success = 0
    failed = 0
    failed_ids = []

    print(f"开始重新生成 {total} 个损坏的ESM2文件...")
    print(f"开始时间: {datetime.now()}")
    print()

    with open(log_file, 'w') as log:
        log.write(f"开始时间: {datetime.now()}\n\n")

        for i, pdb_id in enumerate(CORRUPTED_IDS, 1):
            print(f"[{i}/{total}] 处理 {pdb_id}...")
            log.write(f"[{i}/{total}] 处理 {pdb_id}...\n")

            # 删除损坏的文件
            corrupted_file = Path(output_dir) / "2011-2019" / pdb_id / f"{pdb_id}_protein.pt"
            if corrupted_file.exists():
                print(f"  删除损坏文件: {corrupted_file}")
                log.write(f"  删除损坏文件: {corrupted_file}\n")
                corrupted_file.unlink()

            # 重新生成
            cmd = [
                "python", "scripts/extract_esm2_features.py",
                "--pdb_id", pdb_id,
                "--input_dir", pdbbind_dir,
                "--output_dir", output_dir
            ]

            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5分钟超时
                )

                if result.returncode == 0:
                    print(f"  ✓ 成功")
                    log.write(f"  ✓ 成功\n")
                    success += 1
                else:
                    print(f"  ✗ 失败: {result.stderr[:100]}")
                    log.write(f"  ✗ 失败: {result.stderr}\n")
                    failed += 1
                    failed_ids.append(pdb_id)

            except subprocess.TimeoutExpired:
                print(f"  ✗ 超时")
                log.write(f"  ✗ 超时\n")
                failed += 1
                failed_ids.append(pdb_id)
            except Exception as e:
                print(f"  ✗ 错误: {e}")
                log.write(f"  ✗ 错误: {e}\n")
                failed += 1
                failed_ids.append(pdb_id)

            log.write("\n")
            print()

        # 总结
        log.write(f"\n完成时间: {datetime.now()}\n")
        log.write(f"\n总结:\n")
        log.write(f"  总数: {total}\n")
        log.write(f"  成功: {success}\n")
        log.write(f"  失败: {failed}\n")

        if failed_ids:
            log.write(f"\n失败的PDB IDs:\n")
            for pdb_id in failed_ids:
                log.write(f"  {pdb_id}\n")

    print(f"完成时间: {datetime.now()}")
    print()
    print("总结:")
    print(f"  总数: {total}")
    print(f"  成功: {success}")
    print(f"  失败: {failed}")

    if failed_ids:
        print(f"\n失败的PDB IDs: {', '.join(failed_ids)}")

    print(f"\n详细日志: {log_file}")

if __name__ == "__main__":
    main()
```

**使用方法**:
```bash
chmod +x regenerate_corrupted_esm2.py
python regenerate_corrupted_esm2.py
```

---

### 方法3: 手动逐个处理

如果需要手动处理单个文件：

```bash
# 1. 删除损坏的文件
rm data/esm2_latents/2011-2019/4z0q/4z0q_protein.pt

# 2. 重新生成
python scripts/extract_esm2_features.py \
    --pdb_id 4z0q \
    --input_dir /path/to/pdbbind \
    --output_dir data/esm2_latents

# 3. 验证
python -c "
import torch
data = torch.load('data/esm2_latents/2011-2019/4z0q/4z0q_protein.pt')
print(f'Shape: {data[\"embeddings\"].shape}')
print('✓ 文件正常')
"
```

---

## 验证重新生成的文件

运行验证脚本：

```python
import torch
from pathlib import Path

corrupted_ids = [
    '4z0q', '4z1e', '4z1k', # ... 完整列表
]

print("验证重新生成的文件...")
success = 0
failed = []

for pdb_id in corrupted_ids:
    file_path = f"data/esm2_latents/2011-2019/{pdb_id}/{pdb_id}_protein.pt"

    try:
        data = torch.load(file_path, weights_only=False)
        if 'embeddings' in data:
            print(f"✓ {pdb_id}: {data['embeddings'].shape}")
            success += 1
        else:
            print(f"✗ {pdb_id}: 缺少embeddings")
            failed.append(pdb_id)
    except Exception as e:
        print(f"✗ {pdb_id}: {e}")
        failed.append(pdb_id)

print(f"\n总结: {success}/{len(corrupted_ids)} 成功")
if failed:
    print(f"失败: {', '.join(failed)}")
```

---

## 注意事项

1. **原始PDB文件**: 确保有原始PDB文件（`{pdb_id}_protein.pdb`）
2. **ESM2模型**: 确保ESM2模型已下载并可用
3. **显存**: ESM2推理需要约8-16GB显存
4. **时间**: 每个文件约需30-60秒，总计约2-3小时
5. **备份**: 建议先备份损坏的文件以便调试

---

## 常见问题

### Q: 为什么只有2011-2019年份段损坏？

A: 可能是该批次生成时出现了系统问题（磁盘空间不足、进程被中断等）

### Q: 可以跳过这些文件吗？

A: 可以，已添加到黑名单。但会损失134个训练样本（约0.7%）

### Q: 重新生成后还是损坏怎么办？

A: 检查原始PDB文件是否完整，或者该PDB可能本身有问题

---

**生成时间**: 2026-02-06
**损坏文件数**: 134
**影响比例**: 0.7%
