---
name: V12 NaN/Inf 訓練不穩定調查
description: V12 training immediately collapses with NaN/Inf — investigation state and next steps
type: project
---

V12 訓練開始即出現 NaN/Inf，collapse 位置在不同 run 間漂移（非固定）。

**已排除**：
- LayerNorm/RMSNorm 對零向量不會 NaN（有 eps）
- fp16 overflow 無法解釋 bf16 也崩（bf16 exponent range ≈ fp32）

**高概率方向（已確認，待 patch 驗證）**：
1. `RoPEGatedMultiHeadAttention`（V11繼承，被 V12 protein/ligand self-attn 使用）沒有：
   - `softmax(.float())` 升精度
   - `attn_logit_clip`
   V12 每個 CoAttentionBlock 都調用 2 次，路徑比 V11 更深，邊界問題被放大。
2. DistanceBias 的 `-inf` mask 若產生全 `-inf` row（sample 有效 key=0 時），softmax NaN 必然觸發。
3. padding query 行的 NaN 輸出/梯度進入後續 LN/殘差，透過 V12 的 AttnRes（6×2 次 RMSNorm+softmax）放大傳播。

**計劃中的 patch（用戶自己修改）**：
1. 給 `RoPEGatedMultiHeadAttention` 加 `attn_logit_clip` + `softmax(.float())`
2. softmax 後對 query padding 位置 `masked_fill(~q_mask, 0)`

**How to apply**: 等用戶 patch 後，若 train-mode probe 穩定 → 根因確認在 self-attn 數值路徑；若仍崩 → 看 FiLM/AttnRes/optimizer param group。
