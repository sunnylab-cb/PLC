# Front protein sequence-context controls

Date: 2026-10-29. User approved MLP/TCN/BiGRU scratch pilots on U.
This is a targeted follow-up to [LocalScan](local_spatial_scan_20261024.md),
not a replacement of the whole protein/ligand architecture.
Deployment and job states: [tracker](../refine-logs/sequence_context_controls_20261029/EXPERIMENT_TRACKER.md).

Submitted20:54 CST: smoke `13400657_[0-2]`, dependent training `13400659_[0-2]`.
Both use `i64m1tga800u`, one GPU/task. At verification smoke was Priority
pending and training Dependency pending, not yet running. Snapshot `b4221db`;
105 CPU tests passed and real production config/initialization parity with G0
was checked on HPC. Existing LocalScan jobs were not cancelled or restarted.

## Question And Controls

Does the already contextualized ESM/GVP representation need the additional
GFGF front stack, or can a smaller pointwise/convolutional/recurrent operator
improve strict VAL performance? LocalScan removed co-attention and late delta
too, so its negative local results cannot isolate this question.

```text
unchanged ESM2 + GVP + physicochemical fusion (G0 width256)
  -> front context: GFGF / pointwise MLP / TCN / BiGRU
  -> unchanged ligand graph/self-attention and V12-style co-attention
  -> unchanged group_rescompat local54 delta and classifier
```

G0 `13312384_0` is the existing seed44 reference; do not restart or cancel it.
Its observed best strict-VAL F1 is0.602434 at epoch52, AUPRC0.633018 and
MCC@0.5 0.601489 at that same checkpoint. This pilot does not use TEST scores.

| Arm | Front operator | Front parameters, width256 | Hypothesis |
| --- | --- | ---: | --- |
| M0_pointwise | Four pre-LN residual MLPs, 256->512->256 | 1,054,208 | Extra front sequence mixing may be unnecessary |
| C0_tcn | Four noncausal depthwise Conv1D + channel MLP blocks | 1,060,352 | Restricted multiscale sequence context may help |
| R0_bigru | Two residual bidirectional GRU blocks, 128 per direction, each with channel MLP | 1,121,280 | Gated state compression may help |

TCN kernel5, dilations1/2/4/8, stride1, one convolution per block: added
sequence receptive field61 within each segment. Each block applies tokenwise
LayerNorm, convolution, channel MLP and dropout before its residual update.
M0 has the same channel-MLP blocks without convolution. R0 has pre-LN GRU,
residual dropout, then pre-LN channel MLP and another residual. All end in
tokenwise LayerNorm. Front dropout follows G0's configured dropout. The new
front blocks do not use G0's stochastic-depth machinery; this is a front-stack
replacement comparison, not an isolated algebraic operator equivalence.
Three new front parameter budgets differ by less than7%; G0 is not exactly
parameter-matched. Total model counts are recorded by smoke.

## Boundary And Numerical Contract

- Require explicit chains; reset at batch/chain boundaries and padding holes.
  Nonconsecutive supplied position IDs also split a segment. Never concatenate
  different chains using artificial glycine/linker residues.
- The current production interface does not carry complete original sequence
  indices. Adjacent native CA jumps above4.5A and missing CA coordinates also
  break segments. This conservative safeguard cannot identify every unresolved
  sequence gap; it must not be described as complete sequence alignment.
- Geometry for segmentation is protein-only. Do not use ligand pose, target
  contacts, pocket labels or a GT-selected chain. Missing-coordinate residues
  remain scored, with sequence mixing isolated around the missing position.
- Length buckets prevent many tiny fragments from all being padded to the
  longest chain. Packed GRU uses true lengths; invalid padding cannot propagate
  through recurrence. TCN zeros padding after every block and before each conv.
- Only GRU recurrent math is FP32 for stability. Other operations retain the
  existing BF16 TRAIN / FP16 VAL policy. No torch.compile. Activation
  checkpointing uses the unchanged outer encoder wrapper.
- Construct the original context first and isolate replacement RNG, preserving
  bitwise initialization of all unchanged modules for seed44. Default configs
  retain the original encoder and checkpoint key schema.
- This is not a no-attention, strictly local, or ligand-free model: frozen ESM
  already supplies context, GVP supplies geometry, and downstream co-attention
  including its inner self-attention remains unchanged.

## Frozen Training And Selection

Reuse G0's frozen expanded joint GLOBAL-SI30 recipe:139158 TRAIN,17395 VAL,
17395 TEST; TRAIN-only normalization; native heavy-atom contact distance <5A,
existing near/far labels and eligibility. Protein cap1800 and ligand cap256
unchanged. All three start from scratch with fully trainable parameters.

Keep batch16/accum1, optimizer/loss/auxiliaries/source dropout/LR schedule,
seed44 and max120epochs. F1 early stopping, patience30. Save only best-F1
and best finite-loss checkpoints, no periodic/latest/independent ranking
checkpoints. The full source/config/input hashes are frozen before smoke.

Select using the existing legacy VAL F1; also report pooled AUPRC/AUROC,
MCC/F1 at0.5 and the combined VAL gate, source-separated and sample metrics,
Top20/Top-L, throughput and peak memory. Confirm a promising result with VAL
real/shuffle/zero and additional seeds before a claim. No TEST or external
benchmark iteration. A small single-seed difference is not a final win.

CPU checks cover segment isolation, geometry/position breaks, padding,
empty masks, 61-position receptive field, state restoration, unchanged
initialization, full co-attention/local54 backward, AMP and checkpoint parity.
Real GPU smoke requires production TRAIN100, synthetic cap1800 backward,
real VAL finite logits/ranking and strict checkpoint load. Smoke weights are
discarded. Full scratch jobs require all three matching successful smokes.
