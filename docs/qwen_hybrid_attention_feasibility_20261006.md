# Qwen-style Gated DeltaNet hybrid for PLC: feasibility and pilot results

2026-10-06. Initially an analysis-only proposal. The user subsequently approved
a conservative trial; the two-arm implementation and submission are recorded
below. Existing assignment-head jobs were not modified.

## Approved trial: execution and evidence

### 2026-10-09: external benchmarks completed

All four12956268 tasks COMPLETED0:0, not OOM. Durations0/1/2/3:
00:02:29 /00:04:00 /00:54:51 /00:42:46. Final task finished16:59:56 CST.
These job times are not a controlled speed comparison (shared GPUs/I/O).
Evidence synced to `tmp/protein_hybrid_benchmark_12956268`, excluding large
prediction caches and QC JSONL (retained remotely).

All completion markers passed. Across both arms and decisions, verified equal
sample IDs, protein/ligand keys, residue/atom counts and true-contact counts.
COACH:494 rows/299 proteins; HOLO:7245 rows/3430 proteins. No skipped rows,
non-finite evaluated logits or ligand-key mismatches. Atom counts uncapped.
Explicit existing input repair affected0 COACH rows/10 HOLO rows in each arm.

All F1 below are per-row contact-class0 sample means, not pooled global F1.
BP is conditional on successful current-pocket localization; it is not an
unconditional site detection score. LAA covers289 COACH/5621 HOLO rows.

| Dataset | Decision | Arm | Sample F1 | LAA | PDR | BP F1 | BP rows |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| COACH420 | argmax | FFFF | 0.447019 | 0.186851 | 0.514170 | 0.718823 | 157 |
| COACH420 | argmax | GFGF | 0.434975 | 0.173010 | 0.518219 | 0.715029 | 156 |
| COACH420 | fixed VAL gate | FFFF | 0.449124 | 0.186851 | 0.518219 | 0.722187 | 156 |
| COACH420 | fixed VAL gate | GFGF | 0.443525 | 0.173010 | 0.493927 | 0.721041 | 150 |
| HOLO4K | argmax | FFFF | 0.324018 | 0.135385 | 0.418357 | 0.698847 | 1622 |
| HOLO4K | argmax | GFGF | 0.323810 | 0.148550 | 0.441684 | 0.704604 | 1726 |
| HOLO4K | fixed VAL gate | FFFF | 0.327646 | 0.137698 | 0.418496 | 0.703281 | 1646 |
| HOLO4K | fixed VAL gate | GFGF | 0.330157 | 0.146059 | 0.443616 | 0.712820 | 1715 |

| Dataset, fixed VAL gate | FFFF | GFGF |
| --- | ---: | ---: |
| COACH protein_mean_of_ligand_mean F1 | 0.534223 | 0.536114 |
| COACH protein_mean_of_ligand_max F1 | 0.606052 | 0.608624 |
| COACH protein-balanced LAA | 0.234624 | 0.204906 |
| HOLO protein_mean_of_ligand_mean F1 | 0.439984 | 0.447219 |
| HOLO protein_mean_of_ligand_max F1 | 0.589082 | 0.600459 |
| HOLO protein-balanced LAA | 0.171788 | 0.182507 |

Fixed-gate mean sample AP: COACH0.501223 ->0.496020; HOLO0.367155 ->0.369843.
HOLO mean predicted contacts24.0039 ->23.8073, p50/p90 24/36 ->23/35;
precision/recall0.317014/0.357293 ->0.318399/0.359779. No gross contact-count
collapse. Argmax HOLO F1 is effectively unchanged; calibration matters.

Paired protein-key bootstrap of protein-balanced metrics (5000 resamples,
RNG20261009, COACH then HOLO, sorted keys; keep all ligand rows of each protein
together). Fixed-gate GFGF-minus-FFFF F1 delta: COACH+0.0018908,
95%CI[-0.0103526,+0.0141540]; HOLO+0.0072345,[+0.0042876,+0.0102025].
HOLO protein-balanced LAA delta+0.0107191,[+0.0009627,+0.0205632];
PDR delta+0.0200782,[+0.0043285,+0.0354142]. Conditional on this training seed
and these checkpoints, not a sequence-cluster bootstrap or seed-variance result.

Fixed-gate sample F1 stratified by number of ligand rows per protein:

| Dataset | Ligand count | Rows | FFFF | GFGF |
| --- | --- | ---: | ---: | ---: |
| COACH | 1 | 200 | 0.609106 | 0.618359 |
| COACH | 2-3 | 160 | 0.418268 | 0.409830 |
| COACH | >=4 | 134 | 0.247187 | 0.222812 |
| HOLO | 1 | 1619 | 0.597624 | 0.616401 |
| HOLO | 2-3 | 2639 | 0.352242 | 0.347145 |
| HOLO | >=4 | 2987 | 0.159584 | 0.159998 |

Interpretation: modest HOLO gains in localization/assignment metrics, but its
F1 improvement is driven by single-ligand proteins; multi-ligand F1 is not
consistently better. Multiple ligand rows do not imply distinct binding sites.
COACH does not corroborate a broad improvement. Together with the smaller
internal real-shuffle gap, this does not establish a general ligand-assignment
solution. Keep GFGF as a candidate, not a new mainline; if extended, matched
additional seeds are more informative than a larger architecture sweep.
No new jobs submitted during this result check.

### 2026-10-09: external benchmarks submitted to emergency_gpu

User approved external evaluation. Submitted **12956268_[0-3%2]** at15:59:40
CST; initial status PENDING Resources (start time unknown), no dependency.
Each task requests one A800 GPU,4 CPUs,64GB and8h; at most two concurrent.
Task mapping: 0=FFFF/COACH420,1=GFGF/COACH420,2=FFFF/HOLO4K,3=GFGF/HOLO4K.
Script: `scripts/slurm/run_protein_hybrid_bestf1_benchmark.slurm`.

Same immutable code_v1 and best-F1 weights as the completed internal TEST.
Checkpoint hashes, runner hash and TRAIN normalization stats hash passed;
CPU input preflight passed one row per dataset (not a full-dataset audit).
Expected per-ligand rows: COACH420494, HOLO4K7245, with no silent skipping.
Both argmax and fixed checkpoint VAL gates; no benchmark threshold fitting.
Key-aligned canonical protein coverage capped1800, max_atoms=-1, explicit
existing sasa_zero repair, FP16 inference matching the internal TEST protocol.
This is not a precision-matched rerun of historical BF16 benchmarks; the main
comparison is the matched FFFF/GFGF pair in this submission.

Outputs: `benchmark_bestf1_12956268/{FFFF,GFGF}/{coach420,holo4k}/` under the
hybrid experiment root, logs `logs/benchmark_12956268_<task>.{out,err}`.
Saves sample/protein-balanced metrics, LAA/PDR/BP with coverage, input QC,
contact counts and per-row CSVs; fixed-gate probabilities are also exported.
Completion requires all expected IDs, no atom truncation, finite logits and
identical input axes between decision rules. BP remains conditional on current
pocket localization, not an unconditional pocket-recognition F1.
No benchmark accuracy is claimed at submission; no training was changed.

### 2026-10-09: best-F1 strict TEST completed

12954704_0 and _1 both COMPLETED 0:0, in 00:03:25 and 00:09:58.
Both started at 11:17:28 CST; the final task finished at 11:27:26.
Local evidence: `tmp/protein_hybrid_test_12954704/{FFFF,GFGF}/`.
Both complete markers are present. All 12 per-sample CSVs contain the same647
unique IDs and matching label/residue counts (416304 residues). Shuffle ligand
and UniMol donor mappings match across arms and decision protocols, with no
self-donors. Checkpoint selection remained best VAL F1; no TEST gate tuning.

The following is pooled/global contact-class0 F1, NOT sample-mean F1 or the
broader binary-contact metric:

| Decision | Arm | Real | Shuffle | Zero | Real-shuffle | Real-zero |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Argmax | FFFF | 0.5270152 | 0.4736526 | 0.4698100 | 0.0533626 | 0.0572052 |
| Argmax | GFGF | 0.5312108 | 0.4894646 | 0.4908584 | 0.0417462 | 0.0403524 |
| Fixed VAL gate | FFFF | 0.5276439 | 0.4764451 | 0.4768149 | 0.0511988 | 0.0508290 |
| Fixed VAL gate | GFGF | 0.5366858 | 0.4927917 | 0.4994869 | 0.0438941 | 0.0371989 |

GFGF real gain: +0.0041956 argmax, +0.0090419 fixed VAL gate. Argmax real
precision/recall changes from0.511745/0.543225 to0.541390/0.521407; fixed-gate
precision/recall changes from0.505139/0.552247 to0.518794/0.555856.
Sample-mean real F1: argmax0.515386 ->0.516412; fixed gate0.515506 ->0.522808.

Paired sample bootstrap (10000 resamples, RNG seed20261009, same647 IDs):

| Sample-mean metric: GFGF minus FFFF | Delta | 95% percentile CI |
| --- | ---: | --- |
| Argmax real F1 | +0.0010264 | [-0.0101077, +0.0115840] |
| Fixed-gate real F1 | +0.0073022 | [-0.0037059, +0.0177482] |
| Argmax real-shuffle gap | -0.0089578 | [-0.0176520, -0.0006069] |
| Fixed-gate real-shuffle gap | -0.0054286 | [-0.0138522, +0.0026329] |

These intervals are conditional on one training seed and one shuffle mapping;
they are not sequence-cluster bootstraps or estimates of training-seed variance.
Real F1 has a modest positive point estimate, but no clear sample-mean advantage.
Shuffle/zero improve more than real: this does NOT support stronger ligand
assignment. A stronger protein/context prior or reduced ligand reliance is a
plausible interpretation, not a proven mechanism. Positive real-shuffle gaps
still support correct-ligand utility. A smaller zero gap alone is not a failure.

Retain as a candidate rather than promote to mainline. If pursued, prioritize
matched additional seeds or frozen-protocol COACH/HOLO evaluation over larger
hybrid architecture sweeps. No external benchmark or new training was submitted
in this check. Earlier sections below preserve submission/training-time evidence.

### 2026-10-09: best-F1 strict TEST submitted

User explicitly chose to keep training and early stopping unchanged and evaluate
the best VAL-F1 checkpoints now. Submitted **12954704_[0-1%2]** to
`i64m1tga800ue`, one GPU/task, at most two concurrent. Task0=FFFF epoch33;
task1=GFGF epoch32. No additional training or optimizer resume.

Script: `scripts/slurm/run_protein_hybrid_bestf1_test.slurm`. Both checkpoint/
config/split/hash preflights passed on HPC; bash syntax and embedded Python
compilation passed locally. Uses the unchanged `code_v1` model/eval source and
isolated FLA/einops dependencies. Expected strict TEST size647.

Each task runs whole-model real/shuffle/zero for legacy argmax and fixed
checkpoint VAL-gate decisions. Gates: FFFF0.4393550157546997;
GFGF0.34393057227134705. No TEST threshold sweep or checkpoint selection.
Shuffle seed42, batch4, all per-sample rows saved, complete-ID/no-self-donor
checks, and matching donors across decision rules. Existing evaluation FP16
autocast is retained for both arms, matching the training validation path.
As previously agreed for TEST, max_atoms=-1; protein cap remains1800. This
inference atom-cap change applies equally to both checkpoints.

Checkpoint SHA256:
- FFFF: `21cb445cf6b75cfa6a34da8132a1da97abe4fdd4384bbb21472993ac7039af55`
- GFGF: `1186945af7a3d603031eb031597afb734e2106bd596aa4959a4de148f11aeb7e`

Results will be under the experiment root `test_bestf1_12954704/{FFFF,GFGF}`;
logs `logs/test_12954704_{0,1}.{out,err}`. The complete marker is written only
after both decision protocols cover all647 TEST rows. No TEST result claimed
at submission time. This is not a COACH/HOLO benchmark submission.

### 2026-10-09: both training arms completed

12947231 GPU smoke COMPLETED 0:0 in 00:06:17 (gpu1-25).
12947232_0 FFFF COMPLETED 0:0 in 04:05:23 (gpu1-11).
12947232_1 GFGF COMPLETED 0:0 in 04:25:08 (gpu1-25).
Both trained 35 epochs and stopped by the existing loss-based patience30, not
OOM, timeout or a runtime error. Queue empty. No new jobs submitted in this check.
Lightweight logs/TB/configs: `tmp/protein_hybrid_gdn_results_20261009`.
Remote best-F1 checkpoint metadata independently agrees with TB.

| Strict VAL metric | FFFF | GFGF |
| --- | ---: | ---: |
| Best trainer F1_c0 | 0.5298548 | 0.5359956 |
| Best epoch (one-based) | 33 | 32 |
| Precision at best F1 | 0.5124356 | 0.5402535 |
| Recall at best F1 | 0.5519667 | 0.5339808 |
| AUPRC, also run maximum | 0.5445033 | 0.5462875 |
| AUROC at best-F1 checkpoint | 0.9482505 | 0.9464824 |
| Global binary F1 at 0.5 | 0.5304587 | 0.5362874 |
| Global binary MCC at 0.5 | 0.5152892 | 0.5217544 |
| Global binary F1 at VAL-selected gate | 0.5338611 | 0.5392231 |
| VAL-selected gate | 0.4393550 | 0.3439306 |
| Last F1_c0 | 0.5196 | 0.5061 |
| Tail-five mean F1_c0 | 0.5215673 | 0.5209200 |
| Top-L/20 precision, best-F1 checkpoint | 0.4580420 | 0.4585608 |
| Top-L/10 precision, best-F1 checkpoint | 0.3311893 | 0.3274577 |
| Top-L/5 precision, best-F1 checkpoint | 0.2098608 | 0.2075843 |

Unless otherwise stated, metrics describe each arm's best-F1 checkpoint. Trainer
F1_c0, global binary F1 and the previous assignment-head sample-mean F1 have
different aggregation/decision rules; do not directly compare their headlines.
These are VAL-selected single-seed results, not TEST or external benchmarks.

GFGF best F1 improves by 0.0061408 and AUPRC by 0.0017842, with higher precision
and lower recall. This is a weak positive pilot signal, not a stable superiority
claim: tail-five mean is slightly lower, last epoch worse, AUROC at the selected
checkpoint lower, and top-L rankings do not improve consistently. No whole-model
real/shuffle/zero experiment has yet been run on these checkpoints.

Training loss falls from about 0.729 to 0.075 in both arms, while validation loss
is minimized around epoch5 and then rises. Nevertheless best F1/AUPRC occurs
at epoch32/33, near the stopping boundary. The trainer saves metric-best
checkpoints but only val loss resets patience. At stopping, LR is still about
8.43e-5 (initial1e-4, planned120 epochs). Thus completed jobs are not evidence
that F1/AUPRC optimization has converged. Rising loss may reflect worsening
confidence/generalization even as ranking improves; do not simply label this
undertraining or promise that longer runs fix it.

Recorded pre-clipping gradient norms (2240 snapshots/arm, none non-finite):

| Norm | FFFF median / p95 | GFGF median / p95 |
| --- | ---: | ---: |
| Total | 1.08171 / 2.28516 | 1.18612 / 2.58838 |
| Protein context encoder | 0.41391 / 0.81284 | 0.54533 / 1.11544 |
| GVP encoder | 0.15643 / 0.33827 | 0.16924 / 0.36996 |
| Cross-attention | 0.08800 / 0.20269 | 0.08165 / 0.19313 |

No observed gradient explosion/NaN. GDN protein-context gradients are higher,
not disconnected. GPU smoke matched the FP32 oracle at output relative RMSE
0.00600 and input-gradient relative RMSE0.00659; all checked parameter-gradient
relative errors were below0.0075. This establishes implementation viability,
not biological accuracy.

Short real-batch smoke (B4,L379): peak allocated3.3742 vs3.3753 GiB; warmed
mean step0.2260 vs0.2398 sec, so no measured memory gain and slight overhead.
GDN first full-model step took112.36 sec for kernel compilation. Production
median time to batch640 after epoch5 was398.8 vs382.4 sec, but on different
nodes; total GFGF job was longer. Do not claim a controlled speedup or broad
memory saving from these mixed measurements. Slurm MaxRSS is CPU memory,
not GPU VRAM.

Next priorities: matched checkpoint ligand-dependency evaluation before
promoting the hybrid; if continuing training studies, use the same explicitly
chosen AUPRC/F1-based stopping rule or fixed budget for BOTH arms. Do not
silently extend only the promising checkpoint or treat a single-seed maximum
as a confirmed improvement. No expanded seed/architecture matrix is warranted
yet. Earlier submission and repair records below are historical.

2026-10-08 update: the original smoke failed after 34 seconds with
`ModuleNotFoundError: No module named 'einops'` while importing FLA. NOT OOM,
not a failed numerical comparison, and not training divergence. Original
training array 12931199 was dependency-cancelled without running. The local
CPU tests did not import the production CUDA backend, so did not catch this.

Repaired the isolated dependency directory with `einops==0.8.1`, without
upgrading shared packages. Wheel SHA256:
`919387eb55330f5757c6bea9165c5ff5cfe63a642682ea788a6d472576d81737`.
Verified `einops`, torch2.9.1+cu128, Triton3.5.1 and importing
`fla.ops.gated_delta_rule.chunk_gated_delta_rule` on the login node. Its CPU
fallback warning is expected without an allocated GPU and is not a GPU test.
Source snapshot, model configs, data and acceptance thresholds are unchanged.

Retry: smoke **12947231** (PENDING Priority); FFFF/GFGF scratch array
**12947232_[0-1%2]** (PENDING afterok:12947231). Both -ue, one GPU/task.
There is still NO hybrid training accuracy, gradient, throughput or VRAM result.
The following table preserves original submission-time status, not current state.

| Job | Work | Last checked status |
| --- | --- | --- |
| 12931196 | FLA/reference GPU and real-TRAIN forward/backward smoke | PENDING Priority |
| 12931199_0 | FFFF scratch, seed44 | PENDING afterok:12931196 |
| 12931199_1 | GFGF scratch, seed44 | PENDING afterok:12931196 |

All use `i64m1tga800ue`, one GPU/task. The training array allows at most two
concurrent tasks and cancels on invalid dependency. The GPU smoke has a 45-minute
process limit; its JSON success marker is also required by the training script.
No GPU correctness, speed, memory or accuracy result is claimed while pending.
GGGF and new seeds were NOT submitted.

Experiment root: `/hpc2hdd/home/shuang886/PLC/experiments/protein_hybrid_gdn_20261006`.
Source snapshot: `code_v1`; generated configurations and provenance: `prepared`;
logs: `logs`; future training artifacts: `outputs`. No jhspoolers writes.
Slurm entry: `scripts/slurm/run_protein_hybrid_gdn.slurm`.

Implementation is default-off in `RoPEProteinContextEncoder`; existing FFFF
state-dict keys and initialization are preserved. G layers replace only front
attention mixers 0 and 2, retaining their LayerNorm/SwiGLU/residual paths. They
use shared-weight forward/backward GDN, mean direction fusion, segmented short
convolution, L2-normalized Q/K, learned decay/update gates and output gating.
Both recurrent state and convolution reset at chain/batch boundaries and padding
holes. F layers keep protein-internal distance bias; all Co-attention, GVP,
ligand and local54 paths are unchanged. This is a biological adaptation, not
the complete Qwen architecture or a pretrained Qwen weight transfer.

Production uses isolated `fla-core==0.4.1` chunk kernels. The FP32 Python
recurrence is bounded to tiny test inputs and never used by the training configs.
FLA wheel SHA256: `93c6afe4c80fc7bc705fa8aeea6a46d2cf2d77383f9619a41863c7114c801bab`.
No shared-environment upgrades and no `torch.compile`. The fixed four-tap
convolution is vectorized over residues; its overhead is included in smoke timing.

Both arms derive from the strict merged base44 config: TRAIN 10317, VAL 647,
seed44, random initialization, batch16, LR1e-4, 120 epochs maximum, patience30,
sequence cap1800, atom cap256, existing loss and precision settings. Periodic
checkpoint interval is 120 to avoid recreating the old storage problem; best
F1/AUPRC/AUROC saving remains in the shared trainer. TEST/COACH/HOLO are not used
for this trial's selection.

Known source-verified label-axis corrections for 10 TRAIN/VAL IDs are materialized
into a private label copy for BOTH arms. Shared features/labels are not changed.
Consequently the primary control is the newly trained FFFF, not the historical
checkpoint's legacy label-prefix metrics. This is not a new full label audit.
Strict split SHA256: `170a4e6364c19662ad922ace7957e1733409dfe146ebdb74b15c95130bb622bc`.
Config comparison differs only in experiment name and `protein_hybrid_pattern`.
Checksum rsync dry-run found no source-file differences after deployment.

Local BIO tests: 8 passed (6 new GDN tests and 2 existing UniMol-config tests).
Actual-config CPU construction also verified 650 common state tensors are
identically initialized. Parameter counts: FFFF 84,755,558; GFGF 84,784,118
(+28,560, approximately +0.034%).
They cover chain isolation, padding, reversal, gradients, defaults and preserved
initialization. GPU smoke additionally checks FLA vs reference outputs/gradients,
full-model common tensors, real-sample BF16 backward and existing FP16 validation
forward, plus parameter count/step time/peak memory. These tests do not establish
improved ligand assignment; that still requires held-out evaluation.

## Verified design and mechanism

Qwen3-Next uses three Gated DeltaNet layers for each full gated-attention layer;
the released config has `full_attention_interval=4`. These are interleaved
serial token-mixing layers, not four branches summed within one layer.
[Official Qwen announcement](https://qwen.ai/blog?id=qwen3-next),
[released config](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json).
The same 3:1 design is documented for Qwen3.5 in its
[Transformers implementation documentation](https://huggingface.co/docs/transformers/main/model_doc/qwen3_5).

Gated DeltaNet combines forgetting with a key-specific prediction-error update.
Using a value-by-key state convention, a simplified recurrence is:

```
S_decay = alpha_t * S_previous
error   = v_t - S_decay @ k_t
S_t     = S_decay + beta_t * outer(error, k_t)
output  = S_t @ q_t
```

This is learned associative memory, not removal of softmax from the existing
attention code. The paper includes normalized Q/K, short convolutions and a
chunk-parallel algorithm; its demonstrated benefits concern language/retrieval
tasks, not protein pocket prediction. Full attention supplies an explicit
token-addressing path alongside compressed recurrent memory.
[Yang, Kautz and Hatamizadeh, ICLR 2025](https://arxiv.org/html/2412.06464v3).

## Current model, verified against the deployed config

Reference: strict merged-split base44 checkpoint under
`outputs/pocket_gvp_fusion_seed44_chain_si30_biolipmerge_scratch_seed44_12222454_1_v11m13_20260826_055836`.

- `d_model=512`, 8 heads; protein GVP 3 layers.
- Front protein context encoder: 4 noncausal self-attention layers with
  per-layer protein-internal distance bias, gated output and SwiGLU.
- Ligand graph encoder: 3 layers, followed by 2 ligand self-attention layers.
- 4 V12-style CoAttention blocks, each containing protein self-attention,
  ligand self-attention and bidirectional cross-attention.
- Local54 group K/V and residue-chem compatibility remain active; protein
  gradient scale 0.6; no UniMol input or global delta gate.
- Sequence cap 1800, atom cap 256, alternating-ends atom crop.
- Front self-attention uses SDPA; no torch.compile.

Code anchors: `src/models/pocket_gvp_fusion_v11.py:422` (front encoder),
`src/models/pocket_gvp_fusion_v11.py:188` (noncausal SDPA),
`src/models/pocket_gvp_fusion_v12.py:805` (inner protein self-attention),
`src/models/pocket_gvp_fusion_v11m8x.py:184` (CoAttention wrapper).
The wrapper currently supplies no coordinates/position IDs to CoAttention;
inner protein distance bias is disabled in this config. It is enabled in the
front encoder. Chain-aware recurrent variants would require explicit chain
metadata, not inference from an arbitrary packed sequence order.

## Proposed first probes

Only replace front protein token mixers; preserve FFNs, residual interfaces,
GVP, ligand encoder, all CoAttention/cross-attention and local54 pathways.
Let F be existing full noncausal distance-biased attention and G be a proposed
bidirectional, chain-segmented Gated DeltaNet mixer (not an off-the-shelf causal
Qwen layer).

| Arm | Four front mixers | Purpose |
| --- | --- | --- |
| Control | F F F F | Same-config reference |
| hybrid_gdn2_full2 | G F G F | Retain more explicit geometric interaction |
| hybrid_gdn3_full1 | G G G F | Test Qwen-inspired 3:1 ratio |

The biological adaptations below are project hypotheses, not published proof:

- Run forward/reverse recurrences within each valid protein chain and fuse
  them; reverse valid residues rather than padded tensor positions. No recurrent
  state or convolution history should cross chain or batch boundaries.
- Keep GVP and geometry-biased F layers. Ordinary DeltaNet does not support an
  arbitrary dense residue-pair distance bias as a drop-in score modification.
  Replacing F removes that layer's explicit pairwise geometry path.
- Preserve full cross-attention and local54 readout: these provide explicit
  atom-residue evidence. Do not concatenate protein residues and ligand atoms
  into a single causal stream as an implicit substitute.
- Do not first apply recurrence to RDKit atom enumeration: atom ordering is
  not a biological sequence. Current ligand self-attention deliberately
  neutralizes sequential RoPE; a recurrence would reintroduce ordering effects.

For a clean first comparison use strict split seed44, identical feature caps,
loss, precision, effective batch and training budget, with all arms trained
from scratch rather than resuming old optimizer states into new blocks. Reuse
an existing baseline only when its full training protocol truly matches. Log
parameter counts and actual memory/throughput; bidirectionality has a cost.
Confirm seeds43/45 only after a promising pilot. No F1 gain is forecast.

## Cost and scope limits

The full model executes eight protein self-attention blocks: four before and
four inside CoAttention. Changing three front layers leaves five protein full
attention layers, as well as ligand attention and all cross-attention. It is
not a 75% whole-model memory reduction and the remaining full layers retain a
quadratic sequence-length term. At cap 1800, a fused full-attention kernel may
still outperform an inefficient recurrent implementation; profile rather than
extrapolating Qwen's long-context autoregressive speedups.

Use maintained chunk-parallel FLA kernels if compatible with the HPC environment,
not a Python token loop for production. Their Triton kernel compilation is
separate from the prohibited model-wide `torch.compile`, but first-use compile
cost and forward/backward support still need a bounded timing check.
[Official implementation and kernel guidance](https://github.com/NVlabs/GatedDeltaNet).

A later, separate variant could replace only the protein self-attention inside
three CoAttention blocks while retaining every cross update. That targets
ligand-conditioned context but changes a different pathway and must not be
combined with the front swap in the first comparison.

Evaluate full VAL F1/AUPRC, fixed and VAL-calibrated classification, protein-
balanced metrics, count distributions, memory/time and sequence-length buckets.
If promising, freeze selection and evaluate TEST ligand dependency and external
multi-site LAA. These mixers can change context aggregation and computational
cost; they do not create missing ligand-to-distinct-pocket supervision. Past
BiMamba results are a caution against assuming that another sequence mixer
alone solves HOLO4K assignment, not a refutation of this specific hybrid.
