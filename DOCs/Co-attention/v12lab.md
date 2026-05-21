# V12 Co-Attention Lab Notes

更新日期：2026-05-21  
資料來源：HPC TensorBoard events，已用 `rsync` 同步到 `outputs/hpc_latest_compare/`。  
記錄目的：追蹤 V12/V12m1/V12m2 co-attention 與 interaction-token 實驗，避免只看單一 epoch。

## 1. 實驗列表

| 實驗 | HPC 路徑 | 本地同步路徑 | 核心設計 |
|---|---|---|---|
| V12 direct cross-attn | `/hpc2hdd/home/shuang886/jhspoolers/PLC/outputs/pocket_gvp_fusion_v12_20260520_154335` | `outputs/hpc_latest_compare/pocket_gvp_fusion_v12_20260520_154335` | Direct P<->L CoAttentionBlock，無 interaction-token bottleneck |
| V12m1 hybrid | `/hpc2hdd/home/shuang886/jhspoolers/PLC/outputs/pocket_gvp_fusion_v12m1_20260520_160725` | `outputs/hpc_latest_compare/pocket_gvp_fusion_v12m1_20260520_160725` | Direct P<->L + interaction tokens hybrid |
| V12m2 bottleneck-only | `/hpc2hdd/home/shuang886/jhspoolers/PLC/outputs/pocket_gvp_fusion_v12m2_20260521_042939` | `outputs/hpc_latest_compare/pocket_gvp_fusion_v12m2_20260521_042939` | 移除 direct P<->L，跨模態只能走 interaction tokens |
| V12m1 calibration | `/hpc2hdd/home/shuang886/jhspoolers/PLC/outputs/pocket_gvp_fusion_calib_v12m1_20260521_042939` | `outputs/hpc_latest_compare/pocket_gvp_fusion_calib_v12m1_20260521_042939` | V12m1 + label smoothing 0.08 + wd 0.08 + focal warmup 10 + classifier dropout 0.05 |

## 2. 主要結論

1. **V12m2 bottleneck-only 是目前最高峰值**：best `Val/F1_c0 = 0.4613 @ epoch 21`，明顯高於 V12/V12m1 的 ~0.44。
2. **V12m2 後期嚴重過擬合/過置信**：latest epoch 99 F1 回落到 `0.4408`，`Val/focal_loss = 1.8264`，train loss 仍降到 `0.0215`。
3. **Interaction-token bottleneck 確實承載主要跨模態梯度**：V12m2 latest `interaction/ESM = 21.97%`，`interaction/GVPproj = 41.03%`，而 direct `co_attn_blocks = 0` 符合 bottleneck-only 設計。
4. **V12 direct P<->L 更不穩**：`p2l_delta_norm` 後期仍高，V12 latest `3.72`，而 V12m1 latest約 `1.09`，V12m2不使用 direct P<->L。
5. **V12m1 calibration 目前尚未超越原 V12m1/V12m2，但方向不壞**：到 epoch 42 latest/best `F1=0.4387`，且目前仍在上升，需繼續觀察。

## 3. 指標總覽

| 實驗 | latest epoch | latest F1 | best F1 | best epoch | latest Dice | latest IoU | precision | recall | latest val loss | latest focal | latest dice loss |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V12 direct | 77 | 0.4318 | 0.4343 | 52 | 0.4295 | 0.2735 | 0.3943 | 0.4854 | 0.7756 | 0.8732 | 0.5478 |
| V12m1 hybrid | 66 | 0.4378 | 0.4412 | 60 | 0.4350 | 0.2779 | 0.3863 | 0.5129 | 0.7409 | 0.8304 | 0.5320 |
| V12m2 bottleneck-only | 99 | 0.4408 | 0.4613 | 21 | 0.4411 | 0.2830 | 0.4477 | 0.4370 | 1.4413 | 1.8264 | 0.5426 |
| V12m1 calibration | 42 | 0.4387 | 0.4387 | 42 | 0.4344 | 0.2774 | 0.3843 | 0.5220 | 0.5862 | 0.6065 | 0.5388 |

### Last-window stability

| 實驗 | last10 F1 mean | last10 F1 std | last5 F1 mean | last5 F1 std | 趨勢判斷 |
|---|---:|---:|---:|---:|---|
| V12 direct | 0.4292 | 0.0024 | 0.4299 | 0.0013 | 平台，低於 best |
| V12m1 hybrid | 0.4357 | 0.0031 | 0.4370 | 0.0019 | 平台，略優於 V12 |
| V12m2 bottleneck-only | 0.4415 | 0.0005 | 0.4412 | 0.0003 | 後期穩定但已過擬合，遠低於早期 best |
| V12m1 calibration | 0.4279 | 0.0062 | 0.4293 | 0.0074 | 仍在上升，epoch 42 創新高 |

## 4. 梯度與信息流

| 實驗 | ligand/ESM | co/ESM | interaction/ESM | token/ESM | ligand/GVPproj | interaction/GVPproj | 解讀 |
|---|---:|---:|---:|---:|---:|---:|---|
| V12 direct | 7.19% | 7.18% | 0.00% | 0.00% | 9.69% | 0.00% | 直接 P<->L 承載跨模態，p2l 擾動較大 |
| V12m1 hybrid | 4.87% | 1.71% | 0.00%* | 0.00%* | 8.54% | 0.00%* | 此 run 啟動早於 interaction grad logging patch，缺 token 梯度記錄 |
| V12m2 bottleneck-only | 5.14% | 0.00% | 21.97% | 2.28% | 9.61% | 41.03% | interaction blocks 是主要跨模態通道，符合設計 |
| V12m1 calibration | 4.04% | 1.63% | 13.72% | 2.50% | 6.21% | 21.09% | hybrid + calibration 中 interaction tokens 也吃到明確梯度 |

`*` V12m1 hybrid 舊 run 沒有 `grad_norm_interaction_blocks/tokens`，不是代表沒有梯度。

### Delta norm

| 實驗 | p2l_delta_norm | l2p_delta_norm | unimol_atom_delta_norm | 解讀 |
|---|---:|---:|---:|---|
| V12 direct | 3.7245 | 1.1766 | 0.4316 | P->L direct 擾動仍大，可能是 val 抖動來源 |
| V12m1 hybrid | 1.0876 | 1.1323 | 0.4454 | direct cross-modal 擾動較溫和 |
| V12m2 bottleneck-only | N/A | N/A | 0.4668 | 無 direct P->L，interaction token 取代跨模態通道 |
| V12m1 calibration | 0.6003 | 1.2832 | 0.4141 | calibration 進一步壓低 P->L direct 擾動 |

## 5. 分實驗解讀

### V12 direct cross-attn

- Best F1：`0.4343 @ epoch 52`。
- Latest F1：`0.4318 @ epoch 77`。
- last10 F1 平均 `0.4292`，已基本平台。
- Val focal 持續惡化：latest `0.8732`，而 train loss 已降到 `0.0764`。
- 結論：V12 direct P<->L 可以工作，但 p2l 擾動偏大，表現不如 V12m1/V12m2。

### V12m1 hybrid

- Best F1：`0.4412 @ epoch 60`。
- Latest F1：`0.4378 @ epoch 66`。
- 相比 V12：更高 recall、更低 val loss/focal，整體略優。
- 舊 run 沒有 interaction-token 梯度記錄；後續新 run 已有 logging patch。
- 結論：hybrid interaction tokens 有效，但仍受 direct path 影響。

### V12m2 bottleneck-only

- Best F1：`0.4613 @ epoch 21`，目前所有實驗最高。
- Latest F1：`0.4408 @ epoch 99`，後期回落到 V12m1 同級。
- Precision latest `0.4477`，recall latest `0.4370`，後期偏保守。
- Interaction blocks 梯度明確：`interaction/ESM = 21.97%`，`interaction/GVPproj = 41.03%`。
- Val focal 過擬合非常嚴重：latest `1.8264`，train loss `0.0215`。
- 結論：bottleneck-only 是正確方向，但需要防止早期 best 後過擬合。V12m2 的 checkpoint selection 應按 F1/Dice，而不是 latest 或 val loss。

### V12m1 calibration

- 到 epoch 42，latest 即 best：F1 `0.4387`。
- Label smoothing/weight decay/dropout 延緩了過擬合，但目前尚未超越原 V12m1/V12m2 best。
- Interaction blocks 梯度可見：`interaction/ESM = 13.72%`。
- latest focal `0.6065`，顯著低於同樣中後期的 V12/V12m1/V12m2，說明 calibration 確實壓制過置信。
- 結論：值得繼續跑。它可能犧牲早期爬升速度，但能改善後期 focal/FP。

## 6. 當前判斷

1. **架構方向**：V12m2 bottleneck-only 明確優於 direct-only/hybrid 的峰值，interaction-token bottleneck 應保留。
2. **訓練問題**：所有模型都有 train loss 持續下降、val focal 持續上升的過擬合/過置信問題，V12m2 最嚴重。
3. **checkpoint 策略**：不能用 latest；應保存並比較 best `Val/f1_c0`、best `Val/dice_c0`，必要時加 early stopping on F1/Dice。
4. **下一輪重點**：不是再加強 direct cross-attn，而是讓 V12m2 保持 early best，同時控制 overconfidence。

## 7. 下一步實驗建議

### 優先實驗 A：V12m2 calibration

目的：保留 V12m2 bottleneck-only 的高峰值，同時抑制 epoch 20 後的 focal 爆炸。

建議 config：

```yaml
experiment:
  name: "pocket_gvp_fusion_v12m2_calib"

model:
  classifier_dropout: 0.05

training:
  label_smoothing: 0.08
  weight_decay: 0.08
  focal_gamma_warmup_epochs: 10
```

預期：峰值可能略慢，但若能把 V12m2 best 附近維持到 e40+，就是目前最有希望的主線。

### 優先實驗 B：V12m2 smaller interaction gate

目的：檢查 V12m2 的 early high peak 是否來自 bottleneck 過強後快速過擬合。

候選：

```yaml
model:
  interaction_gate_init: 0.05
  interaction_gate_max: 0.2
```

與目前 V12m2 對比：目前為 `init=0.08, max=0.3`。

### 優先實驗 C：V12m2 early-stop/checkpoint policy

不一定要重跑，只需在評估時使用：

- best F1 checkpoint：epoch 21
- best Dice checkpoint：epoch 21
- latest checkpoint：epoch 99

若 best checkpoint 測試集顯著優於 latest，說明問題主要是訓練後期過擬合，不是架構錯。

## 8. 建議執行順序

1. 先繼續觀察 `V12m1 calibration` 到至少 epoch 70。
2. 同時開 `V12m2 calibration` 作為下一個主實驗。
3. 如果有第二張卡，再開 `V12m2 smaller gate`。
4. 停止單純 direct P<->L 放大類實驗；V12 的曲線已顯示 direct path 不如 bottleneck 穩。
