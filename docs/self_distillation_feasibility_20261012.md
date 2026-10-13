# Self-distillation after the matched TRAIN/VAL audit

Date: 2026-10-12. This note records the original feasibility review. Update:
evaluation/cache job 12977489 completed. User approved two scratch students;
smoke 12977763 and dependent training 12977764_[0-1] were submitted on
`i64m1tga800ue`. See [execution status and completed gap measurements](self_distillation_execution_20261012.md).
No new model weights or datasets were downloaded.

## What the current evidence does and does not say

The completed seed44 GFGF reference and four LiBRe-transfer controls have
similar online TRAIN accuracy (about 96%) and lower VAL accuracy (about 91%).
VAL Focal loss rises while contact F1 and AUPRC improve through later epochs.
This motivates checking generalization and error confidence. It does not yet
establish a calibrated-probability failure, since TRAIN/VAL losses differed
and the old TB did not contain TRAIN F1/AUPRC.

Job 12977489 removes those measurement differences: fixed best-F1 weights,
eval mode on both sides, FP16 forward plus FP32 metric computation, same
max_seq_len=1800, max_atoms=256 and alternating_ends crop, and hard-label
Focal/Tversky without CNC, smoothing, sample weights or auxiliary terms.
The baseline evaluates all 10317 TRAIN rows and all 647 VAL rows. All four
controls evaluate the same source-stratified 2048 TRAIN rows and all VAL.
Baseline metrics for that same subset are also reported. Sources are read
from the merged dataset's provenance, not inferred from sample-name patterns.

Outputs include global and sample-macro contact F1, global MCC/AP/AUROC,
fixed .5 and stored VAL gate metrics, Top-L, common loss, NLL, Brier, ECE15,
wrong-prediction confidence and class/source breakdowns. No TRAIN threshold
search, TEST inference, or benchmark inference is performed. These global
and sample-macro F1 values must not be confused with legacy batch-mean TB F1.

## Primary-source literature

| Paper | Evidence | Boundary of relevance |
|---|---|---|
| Furlanello et al., ICML 2018, [Born Again Neural Networks](https://proceedings.mlr.press/v80/furlanello18a.html) | Same-architecture students can outperform teachers in the paper's vision/language experiments. | Supports a scratch student with soft targets; does not establish a protein-pocket gain. |
| Mobahi et al., NeurIPS 2020, [Self-Distillation Amplifies Regularization in Hilbert Space](https://proceedings.neurips.cc/paper/2020/hash/2288f691b58edecadcc9a8691762b4fd-Abstract.html) | A restricted function-space analysis describes increased regularization; repeated rounds can move from reduced overfitting to underfitting. | A mechanism hypothesis, not a theorem for our nonlinear GVP/co-attention model. |
| Zhang and Sabuncu, NeurIPS 2020, [Self-Distillation as Instance-Specific Label Smoothing](https://proceedings.neurips.cc/paper_files/paper/2020/hash/1731592aca5fb4d789c4119c65c10b4b-Abstract.html) | Relates soft teacher targets to instance-specific regularization and predictive diversity. | Smoothing logits is not the same as providing new ligand/site assignment labels. |
| Mueller et al., NeurIPS 2019, [When Does Label Smoothing Help?](https://proceedings.neurips.cc/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html) | In their experiments, label-smoothed teachers transfer less effectively despite improved teacher accuracy/calibration. | Our .05 ordinal smoothing plus Focal/Tversky is different; this is a caution, not proof that our teacher is unsuitable. |
| Liang et al., NeurIPS 2021, [R-Drop](https://proceedings.neurips.cc/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html) | Symmetric KL between two dropout passes is an alternative consistency regularizer. | It adds another training forward path and is not the same as fixed-teacher self-distillation. |

These papers justify a controlled experiment, not a predicted numerical gain.
Self-distillation does not add missing same-protein distinct-pocket supervision.
It can preserve a teacher's protein-prior bias as readily as useful chemistry.

## Recommended first pilot (now submitted with smoke dependency)

Keep architecture, strict split, features, optimizer, stopping and best-F1
selection unchanged. Use the GFGF baseline `best_f1_c0_model.pt` as a frozen
teacher, with an independently initialized student of identical architecture.
This is NOT resume or warm-start. Retain the original hard-label objective.

| Arm | Added objective | Purpose |
|---|---|---|
| Existing GFGF reference | None | Matched recipe baseline; confirm a lambda=0 implementation smoke is equivalent. |
| `gfgf_sd_t2_lam005` | Temperature 2, KL weight .05 | Low-strength cached-teacher regularization. |
| `gfgf_sd_t2_lam010` | Temperature 2, KL weight .10 | Check whether stronger imitation helps or copies teacher errors. |

Proposed additional objective:

```text
L = L_existing + lambda(epoch) * T^2 * mean_over_rows(
        mean_over_valid_residues(KL(p_teacher(T) || p_student(T))))

p_teacher(T) = softmax(z_teacher / T), detached
p_student(T) = softmax(z_student / T)
T = 2
lambda(epoch): zero during e1-5, linearly ramp over e6-10, then fixed
```

This is an additive regularizer, not a convex replacement of the existing
Focal/Tversky loss. Log its raw/weighted contribution and per-class contributions
so far residues cannot silently dominate. Keep original label smoothing in
both students for the first comparison; changing that simultaneously would
confound the result. Do not start with repeated generations, hard pseudo-labels,
feature/attention matching, or a larger backbone.

Required safeguards before training:

1. Teacher targets contain TRAIN rows only. Never use VAL/TEST/COACH/HOLO
   pseudo-labels for student optimization. The teacher's VAL selection is
   recorded explicitly and does not turn validation data into training data.
2. Cache valid-residue raw logits, not thresholded labels, with checkpoint,
   split and label-overlay hashes, sample IDs, crop configuration and residue
   coordinate/chain/label-axis hashes. Fail closed on missing or reordered rows.
3. Match the same real protein-ligand pair. Never enforce real/shuffle/zero
   output equality. For training's ligand-source dropout, track dropped rows
   and exclude them from KL; hard-label training remains unchanged. Do not
   force a ligand-missing student view to reproduce real-ligand decisions.
4. Teacher is frozen/eval; student remains trainable from scratch. Use FP32
   KL/log-softmax, detach teacher, mask padding before arithmetic, and preserve
   per-row normalization. Verify lambda=0 leaves model outputs, RNG behavior,
   gradients, parameter count and checkpoint loading unchanged.
5. Inspect teacher entropy at T=1/2/4 and confidence when wrong. Three output
   classes plus a near-one-hot TRAIN teacher may leave little extra information
   beyond hard labels. If that is observed, do not assume another generation
   will fix the gap; consider weaker consistency/regularization instead.

Cached teacher logits avoid loading a second full model for every training
step. They reduce repeated teacher-forward cost, not the cost of student
training. This is safer than doubling live forward graphs under the earlier
memory constraints.

## Decision criteria

Primary: best strict-VAL contact F1 versus the unchanged .535996 reference;
an initial +.003 screen is practical, not statistical significance. Check
AUPRC, MCC, source-stratified scores, P/R and predicted contact extent too.
Do not accept lower ECE/NLL alone as improved pocket prediction. Do not
optimize a smaller train/val gap by making TRAIN worse while VAL stays flat.

Only a promising VAL result proceeds to the same TEST real/shuffle/zero
protocol with its fixed VAL gate, then HOLO/COACH and more seeds. Preserve
real F1 and real-shuffle sensitivity; a smaller zero gap is not required.
There is currently no experimental evidence that self-distillation improves
this project. The completed evaluation supplies the teacher cache and measured
gap; dependent training still needs to establish whether the pilot helps.

## Artifacts and reproducibility

- Runner: `scripts/diagnostics/eval_train_val_gap.py`.
- Slurm: `scripts/slurm/run_train_val_gap.slurm`.
- Tests: `tests/test_train_val_gap.py` (10 CPU tests passed).
- Remote root: `/hpc2hdd/home/shuang886/PLC/experiments/train_val_gap_selfdistill_20261012`.
- Frozen model code: `libre_transfer_20261010/code_v1`, unchanged and verified.
- Results: `results_12977489/`; the baseline TRAIN-only cache is
  `baseline/teacher_train_logits.pt` when its full TRAIN pass finishes.
- Job 12977489 was observed RUNNING on gpu1-35, with finite first-batch
  validation logits. Inspect `complete.json` before claiming all arms finished.
