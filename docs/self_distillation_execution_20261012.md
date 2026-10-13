# GFGF self-distillation: first scratch generation

Latest status, 2026-10-12: smoke **12985072 COMPLETED**; students
**12985073_[0-1] COMPLETED**, both with normal early stopping at epoch 35.
Best strict-VAL F1 is **0.530556 / 0.529393**, versus reference **0.535996**.
Student TEST/benchmark evaluation has not run. See the
[completed result analysis](self_distillation_results_20261012.md).
The submission/recovery chronology below describes earlier checks, not
the current queue state.

Date: 2026-10-12. User approved the pilot. Follow-up: the original GPU smoke
failed its lambda-zero weight-equivalence assertion after 3m02s (exit 1), NOT
an OOM. The two dependent training tasks were automatically cancelled before
starting. Outputs are empty: there are no student VAL/TEST results yet.
Diagnostic job 12984888 completed and showed the unchanged base itself fails
the old near-bitwise post-Adam tolerance. Initial weights/RNG match exactly,
and KD0 differences match base-repeat noise. Corrected smoke 12984984 and
dependent training 12984985_[0-1] were submitted in an isolated `_r1` root.
That retry was stopped by an overly tight aggregate-loss tolerance as well;
the active retry is now smoke 12985072 and training 12985073_[0-1] in `_r2`.
See [measured cause, updated checks and retry status](self_distillation_smoke_recovery_20261012.md).

Original submission record (historical, not current state):

| Stage | Slurm ID | State |
|---|---|---|
| Unified TRAIN/VAL evaluation and teacher cache | 12977489 | COMPLETED, exit 0, 22m33s |
| Real B16 GPU smoke, including lambda-zero equivalence | 12977763 | PENDING, Resources, no scheduler ETA |
| T=2, lambda=.05 scratch student | 12977764_0 | PENDING, afterok smoke |
| T=2, lambda=.10 scratch student | 12977764_1 | PENDING, afterok smoke |

Each new task requests **one GPU**, 8 CPUs, 100G RAM on `i64m1tga800ue`.
Array concurrency is at most two tasks, not two GPUs within one training run.
Smoke has a 2h allocation and a 90m process timeout; training has a 36h limit.
No existing job was cancelled. No Star job, download, or shared-code edit.
Submission does not mean the students have started or improved.

## Completed measurement before the pilot

Both sides below use the same frozen best-F1 teacher, eval mode, FP16 forward,
FP32 metrics, identical preprocessing, and no threshold search on TRAIN.
Contact F1 is **global class-0 argmax F1**, not legacy batch-mean TB F1.
Loss is the common hard-label Focal/Tversky objective, without TRAIN-only CNC,
label smoothing, cluster weighting, or auxiliary losses.

| Metric | Full TRAIN (10317 rows) | Strict VAL (647 rows) |
|---|---:|---:|
| Global contact F1 | 0.919765 | 0.535538 |
| Contact AUPRC | 0.979861 | 0.546287 |
| Contact AUROC | 0.999250 | 0.946483 |
| Contact precision | 0.864615 | 0.538876 |
| Contact recall | 0.982430 | 0.532241 |
| Sample-macro contact F1 | 0.922786 | 0.515832 |
| NLL | 0.075080 | 0.300241 |
| Common loss, sample mean | 0.065693 | 1.454576 |
| Wrong-prediction confidence, mean | 0.675306 | 0.736787 |
| Entropy T=1 / T=2 / T=4 | 0.0881 / 0.2685 / 0.5931 | 0.1669 / 0.3903 / 0.6863 |

TRAIN mixes 5160 PDBbind and 5157 BioLiP rows; VAL contains PDBbind only.
This is a genuine generalization gap but not a controlled estimate of pure
overfitting alone: source/sequence-cluster distribution shift also matters.
The gap is not explained away by source composition: PDBbind-only TRAIN
global F1 is 0.970511, AUPRC 0.994948, sample-macro F1 0.971877.
BioLiP TRAIN sample-macro F1 is 0.873667.

The four previous controls also retain large gaps. Compare **the same 2048
source-stratified TRAIN rows** against full VAL, not full-baseline TRAIN with
subset-control TRAIN:

| Fixed best-F1 weights | TRAIN subset F1 | VAL F1 | TRAIN subset AUPRC | VAL AUPRC |
|---|---:|---:|---:|---:|
| GFGF reference | 0.920346 | 0.535538 | 0.979711 | 0.546287 |
| Product readout | 0.900715 | 0.523754 | 0.965950 | 0.530375 |
| Additive control | 0.916882 | 0.526746 | 0.979420 | 0.536480 |
| CNC off | 0.922004 | 0.535743 | 0.987957 | 0.545379 |
| Residue classmass | 0.918892 | 0.531300 | 0.979068 | 0.539931 |

CNC-off's +0.000205 global F1 is negligible and AUPRC is lower; this is not
evidence that it solves generalization. Global and legacy batch-mean rankings
can differ slightly. The teacher remains the preselected baseline, not a
post-hoc replacement chosen from this table.

## Exact experimental change

Teacher: GFGF V11m13 seed44 `best_f1_c0_model.pt`, epoch 32.
Students: identical GFGF/local54/co-attention/group-rescompat architecture,
84,784,118 trainable parameters, seed44, from random initialization. The same
seed policy isolates the training objective; weights are **not** loaded from
the teacher and no layers are frozen.

| Variant | Temperature | Final KL weight |
|---|---:|---:|
| `gfgf_sd_t2_lam005` | 2 | 0.05 |
| `gfgf_sd_t2_lam010` | 2 | 0.10 |

```text
L_total = L_original_hard_and_aux
          + lambda(epoch) * T^2 / B
            * sum_b [keep_ligand_b * mean_valid_residues KL(teacher_T || student_T)]

e1-5: lambda=0
e6-10: 20%, 40%, 60%, 80%, 100% of final lambda
e11+: full lambda
```

Dropped rows contribute zero rather than renormalizing the remaining rows.
Teacher outputs are cached for TRAIN real pairs only. The three ligand-source
dropout events are recorded with unchanged RNG draws and arithmetic. Any
source-dropped row is excluded from KL, while its original hard loss remains.
Padding is removed before FP32 softmax/KL; teacher logits are detached.
No real/shuffle/zero consistency penalty is imposed.

Original data, max_seq_len=1800, max_atoms=256 alternating-ends crop, B16,
LR=1e-4, WD=.05, ordinal smoothing=.05, Focal/Tversky, CNC, original auxiliary
losses, schedule, early stopping, and best-F1/AUPRC/AUROC saves are unchanged.
`torch.compile` remains disabled. There is no TEST/benchmark target use.

TB adds `KD/lambda`, hard-plus-original-aux loss, raw/weighted KL, eligible and
dropped rows, and KL contributions/counts by true class. Global TRAIN/VAL
metrics above are not comparable to the new combined training loss.

## Validation and execution safeguards

53 local CPU tests passed across self-distillation, existing transfer training,
and shared TRAIN/VAL evaluation. Python compilation and Slurm shell syntax
also passed. Local optional torch-scatter/torch-sparse extensions emitted
load warnings; full GPU behavior is deliberately checked on HPC, not inferred
from CPU tests. Unrelated pre-existing `package.py` whitespace was not edited.

Remote preflight passed: exact checkpoint, config, strict split and label-overlay
hashes; 10317 unique TRAIN cache rows; no VAL/TEST IDs; model/data config match;
finite [valid_residues,3] logits. Each used batch additionally checks labels,
length, coordinates and chain-ID axis hashes. No silent trimming/remapping.

GPU smoke checks the actual production train_epoch with three positive-KL
steps per arm, BF16 backward/finite gradients, FP16 eval, standard checkpoint
reconstruction, unchanged parameter count, and an actual lambda-zero optimizer
step against the untouched training loop. Training cannot start until the
smoke artifact matches the frozen source/config and teacher-cache hashes.

## Decision and artifacts

Primary screen: strict-VAL best F1 versus legacy reference 0.535996, with AUPRC,
MCC and P/R guardrails. A +0.003 improvement is a pilot screen, not significance.
Do not count a smaller train/val gap caused only by lower TRAIN performance,
or lower NLL without better residue prediction, as success. Then evaluate
promising checkpoints with fixed-VAL choices on TEST real/shuffle/zero and
COACH/HOLO. Self-distillation adds no missing distinct-pocket assignment labels
and may copy teacher errors; no gain is claimed before results.

- Remote: `/hpc2hdd/home/shuang886/PLC/experiments/self_distill_gfgf_20261012`.
- Isolated model source: `code_v1`, copied from the completed transfer snapshot,
  with only new KD files overlaid. Shared models/trainers were not modified.
- Prepared configs and hashes: `prepared/`; submissions: `submission.json`.
- Local copy: `tmp/self_distill_gfgf_20261012/`.
- Full unified results: `tmp/train_val_gap_12977489/results_12977489/results.json`.
- Cache on HPC only: `train_val_gap_selfdistill_20261012/results_12977489/baseline/teacher_train_logits.pt`.
- Entry: `scripts/train/train_self_distill.py`; loss: `src/losses/self_distillation.py`.
- Slurm: `scripts/slurm/run_self_distill.slurm`; tests: `tests/test_self_distillation.py`.

The [original feasibility note](self_distillation_feasibility_20261012.md)
contains the primary-source literature and the pre-submission rationale.
